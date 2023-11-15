import json
import pickle
import sys
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor
import random
from copy import deepcopy
import torch
from torch import nn
import heapq

import argparse

from allennlp.common.checks import check_for_gpu
from allennlp.common.util import import_module_and_submodules
from allennlp.models.archival import load_archive

from transformers import AutoTokenizer, AutoModelWithLMHead
sys.path.append(os.getcwd())

from denoising_event_lm.models.event_lm.seq2seq import get_flatten_varg_toks
from denoising_event_lm.models.event_lm.seq2seq import V_ARGS_string_to_varg_seq


BATCH_SIZE = 16


if __name__ == '__main__':
    random.seed(765)


class RandomBaseline:
    def __init__(self, name, cuda_device):
        pass

    def get_all_candidate_chains(self, input_events, unseen_event):
        """
        input_events (assume in temporal order): [varg_dict]
        unseen_events: varg_dict
        """
        candidates = []
        for insert_pos in range(len(input_events)+1):
            events = deepcopy(input_events)
            events.insert(insert_pos, unseen_event)
            assert len(events) == len(input_events) + 1
            candidates.append(events)
        assert len(candidates) == len(input_events) + 1
        return candidates

    def predict_json(self, inputs):
        """
        Override this function for demo
        Expects JSON object as ``{"example": dict,
                                  "beams": BEAM_int,
                                  "unseen_idx": int}``
        """
        #input_events = V_ARGS_string_to_varg_seq(inputs['events'])
        #unseen_events = V_ARGS_string_to_varg_seq(inputs['unseen_events']) if inputs['unseen_events'] else []
        # get the gold vargs
        example = inputs['example']
        varg_seq = example['varg_seq']
        permutation = example.get('permutation') # indices to the true varg_seq
        aug_metadata = example.get('aug_metadata', None)

        # get the gold chain
        if aug_metadata is not None and 'source_chain' in aug_metadata:
            gold_varg_seq = aug_metadata['source_chain']
        elif permutation is not None:
            order = sorted(list(range(len(varg_seq))), key=lambda x: permutation[x])
            gold_varg_seq = [varg_seq[i] for i in order]

        for varg in gold_varg_seq:
            if not 'Description' in varg:
                varg['Description'] = " ".join(get_flatten_varg_toks(varg))

        # get input and unseen
        #unseen_idx = random.choice(range(len(gold_varg_seq)))
        unseen_idx = int(inputs["unseen_idx"])
        input_events = [gold_varg_seq[i] for i in range(len(gold_varg_seq)) if not i == unseen_idx]
        unseen_event = gold_varg_seq[unseen_idx]
        beams = int(inputs['beams'])

        candidates = self.get_all_candidate_chains(input_events, unseen_event)
        random.shuffle(candidates)

        candidates = candidates[:beams]

        topk_candidates = [(cand, 0.) for cand in candidates]

        def chain_str(chain):
            texts = []
            for varg in chain:
                texts.append("<EVENT> " + " ".join(varg['V_toks']) + " <ARGS> " + varg['Description'])
            return texts
        return {"gold_vargs":       gold_varg_seq,
                "input_vargs":      input_events,
                "unseen_vargs":      [unseen_event],
                "beam_pred":        [{'pred_vargs': cand,
                                      'pred_repr': chain_str(cand),
                                      'score': score,
                                      'pred_is_neg_chain': False} 
                                     for cand, score in topk_candidates],
                "all_beam_scores": [0.0]*(len(input_events)+1),
                "best_pos_score": 0.0
                }


class GPT2baseline:
    def __init__(self, model_name, cuda_device):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, pad_token='<PAD>')   # Initialize tokenizer
        self._model = AutoModelWithLMHead.from_pretrained(model_name)    # Download model and configuration from S3 and cache.
        self._model.resize_token_embeddings(len(self._tokenizer))
        self._model.to(cuda_device)
        self._cuda_device = cuda_device
        self._pad_token_id = self._tokenizer.pad_token_id

    def get_all_candidate_chains(self, input_events, unseen_event):
        """
        input_events (assume in temporal order): [varg_dict]
        unseen_events: varg_dict
        """
        candidates = []
        for insert_pos in range(len(input_events)+1):
            events = deepcopy(input_events)
            events.insert(insert_pos, unseen_event)
            assert len(events) == len(input_events) + 1
            candidates.append(events)
        assert len(candidates) == len(input_events) + 1
        return candidates

    def get_topk_candidates(self, candidates, scores, beams):
        min_heap = []
        for i, (cand, score) in enumerate(zip(candidates, scores)):
            if len(min_heap) < beams:
                heapq.heappush(min_heap, (score, i))
            else:
                heapq.heappushpop(min_heap, (score, i))
        topk = [None]*len(min_heap)
        while len(min_heap) > 0:
            score, i = heapq.heappop(min_heap)
            topk[len(min_heap)] = (candidates[i], score)
        return topk

    def _batch_json_to_instance(self, hotpot_dict_instances_list):
        target_str_list = []
        for instance in hotpot_dict_instances_list:
            source_varg_seq = instance['source_varg_seq']
            target_varg_seq = instance['target_varg_seq']
            target_str = " . ".join([varg['Description'] for varg in target_varg_seq]) + ' . '
            target_str_list.append(target_str)

        target_encodes = self._tokenizer.batch_encode_plus(target_str_list, padding=True, return_tensors='pt')

        return target_encodes

    def predict_json(self, inputs):
        """
        Override this function for demo
        Expects JSON object as ``{"example": dict,
                                  "beams": BEAM_int,
                                  "unseen_idx": int}``
        """
        #input_events = V_ARGS_string_to_varg_seq(inputs['events'])
        #unseen_events = V_ARGS_string_to_varg_seq(inputs['unseen_events']) if inputs['unseen_events'] else []
        # get the gold vargs
        example = inputs['example']
        varg_seq = example['varg_seq']
        permutation = example.get('permutation') # indices to the true varg_seq
        aug_metadata = example.get('aug_metadata', None)

        # get the gold chain
        if aug_metadata is not None and 'source_chain' in aug_metadata:
            gold_varg_seq = aug_metadata['source_chain']
        elif permutation is not None:
            order = sorted(list(range(len(varg_seq))), key=lambda x: permutation[x])
            gold_varg_seq = [varg_seq[i] for i in order]

        for varg in gold_varg_seq:
            if not 'Description' in varg:
                varg['Description'] = " ".join(get_flatten_varg_toks(varg))

        # get input and unseen
        #unseen_idx = random.choice(range(len(gold_varg_seq)))
        unseen_idx = int(inputs["unseen_idx"])
        input_events = [gold_varg_seq[i] for i in range(len(gold_varg_seq)) if not i == unseen_idx]
        unseen_event = gold_varg_seq[unseen_idx]
        beams = int(inputs['beams'])

        candidates = self.get_all_candidate_chains(input_events, unseen_event)
        random.shuffle(input_events)
        candidate_scores = []
        for b_start in range(0, len(candidates), BATCH_SIZE):
            batch_cands = candidates[b_start:b_start+BATCH_SIZE]
            batch_json_dict = [{'source_varg_seq': input_events, 'target_varg_seq': cand} for cand in batch_cands]
            outs = self.predict_batch_json(batch_json_dict)
            candidate_scores += [o['seq_score'] for o in outs]
            
        topk_candidates = self.get_topk_candidates(candidates, candidate_scores, beams)

        def chain_str(chain):
            texts = []
            for varg in chain:
                texts.append("<EVENT> " + " ".join(varg['V_toks']) + " <ARGS> " + varg['Description'])
            return texts
        return {"gold_vargs":       gold_varg_seq,
                "input_vargs":      input_events,
                "unseen_vargs":      [unseen_event],
                "beam_pred":        [{'pred_vargs': cand,
                                      'pred_repr': chain_str(cand),
                                      'score': score,
                                      'pred_is_neg_chain': False} 
                                     for cand, score in topk_candidates],
                "all_beam_scores": sorted(candidate_scores, reverse=True),
                "best_pos_score": max([score for i, score in enumerate(candidate_scores)])
                }

    def predict_batch_json(self, inputs_list):
        """
        Serve as the substitute for the original ``predict_json``
        This function is only used for computing the score of a target events chain conditioned on input events
        """
        instances = self._batch_json_to_instance(inputs_list)
        return self.compute_sequence_scores(instances)

    def compute_sequence_scores(self, instances):
        with torch.no_grad():
            assert len(instances) == 2
            input_tok_ids = instances['input_ids'].to(self._cuda_device)
            attention_mask = instances['attention_mask'].to(self._cuda_device)
            batch_size = input_tok_ids.size(0)

            output_dict = {}

            # compute loss under teacher forcing, which should be the sequence score
            # create labels
            labels = input_tok_ids.clone().detach()
            labels[input_tok_ids == self._pad_token_id] = -100
            # shift so that tokens < n predict n
            labels = labels[..., 1:].contiguous()

            # prediction_scores, cache, all_dec_hiddens, all_dec_attns, encoder_outputs; if exists
            seq2seq_outputs = self._model(
                input_ids=input_tok_ids,
                attention_mask=attention_mask,
                labels=None, # don't compute loss,
                use_cache=False
            )
            # shape: (batch_size, length-1, vocab_size), shift so that tokens < n predict n
            logits = seq2seq_outputs[0][:, :-1, :].contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction='none')
            # shape: (batch_size*length,)
            label_len = labels.size(1)
            neg_logprob = loss_fct(logits.view(batch_size*label_len, self._model.config.vocab_size),
                                   labels.view(batch_size*label_len))
            # shape: (batch_size,)
            seq_scores = -torch.sum(neg_logprob.view(batch_size, label_len), dim=-1)

            # shape: (batch_size,)
            seq_len = torch.sum((labels != -100).float(), dim=-1)

            output_dict['seq_score'] = seq_scores / seq_len

            instance_separated_output = [
                {} for _ in instances
            ]
            for name, output in list(output_dict.items()):
                output = output.detach().cpu().numpy()
                for instance_output, batch_element in zip(instance_separated_output, output):
                    instance_output[name] = batch_element
            return instance_separated_output


class InfillingGPT2baseline(GPT2baseline):
    def get_all_candidate_chains(self, input_events, unseen_event):
        """
        input_events (assume in temporal order): [varg_dict]
        unseen_events: varg_dict
        """
        candidates = []
        for insert_pos in range(len(input_events)+1):
            prefix_events = input_events[:insert_pos]
            suffix_events = input_events[insert_pos:]
            rotate_events = suffix_events + prefix_events + [unseen_event]
            assert len(rotate_events) == len(input_events) + 1
            candidates.append((rotate_events, insert_pos))
        assert len(candidates) == len(input_events) + 1
        return candidates

    def _batch_json_to_instance(self, hotpot_dict_instances_list):
        target_str_list = []
        for instance in hotpot_dict_instances_list:
            source_varg_seq = instance['source_varg_seq']
            target_varg_seq = instance['target_varg_seq']
            insert_pos = instance['insert_pos']
            assert insert_pos < len(target_varg_seq)

            # last one is the unseen event
            rotate_split_idx = (len(target_varg_seq) - 1) - insert_pos
            varg_seq_part1 = target_varg_seq[:rotate_split_idx]
            varg_seq_part2 = target_varg_seq[rotate_split_idx:]

            target_str = ""
            if len(varg_seq_part1) > 0:
                for varg in varg_seq_part1:
                    target_str += varg['Description'] + " . "
                target_str += self._tokenizer.eos_token
            assert len(varg_seq_part2) > 0
            for varg in varg_seq_part2:
                target_str += varg['Description'] + " . "
            target_str_list.append(target_str)

        target_encodes = self._tokenizer.batch_encode_plus(target_str_list, padding=True, return_tensors='pt')

        return target_encodes

    def predict_json(self, inputs):
        """
        Override this function for demo
        Expects JSON object as ``{"example": dict,
                                  "beams": BEAM_int,
                                  "unseen_idx": int}``
        """
        #input_events = V_ARGS_string_to_varg_seq(inputs['events'])
        #unseen_events = V_ARGS_string_to_varg_seq(inputs['unseen_events']) if inputs['unseen_events'] else []
        # get the gold vargs
        example = inputs['example']
        varg_seq = example['varg_seq']
        permutation = example.get('permutation') # indices to the true varg_seq
        aug_metadata = example.get('aug_metadata', None)

        # get the gold chain
        if aug_metadata is not None and 'source_chain' in aug_metadata:
            gold_varg_seq = aug_metadata['source_chain']
        elif permutation is not None:
            order = sorted(list(range(len(varg_seq))), key=lambda x: permutation[x])
            gold_varg_seq = [varg_seq[i] for i in order]

        for varg in gold_varg_seq:
            if not 'Description' in varg:
                varg['Description'] = " ".join(get_flatten_varg_toks(varg))

        # get input and unseen
        #unseen_idx = random.choice(range(len(gold_varg_seq)))
        unseen_idx = int(inputs["unseen_idx"])
        input_events = [gold_varg_seq[i] for i in range(len(gold_varg_seq)) if not i == unseen_idx]
        unseen_event = gold_varg_seq[unseen_idx]
        beams = int(inputs['beams'])

        candidates = self.get_all_candidate_chains(input_events, unseen_event)
        random.shuffle(input_events)
        candidate_scores = []
        for b_start in range(0, len(candidates), BATCH_SIZE):
            batch_cands = candidates[b_start:b_start+BATCH_SIZE]
            batch_json_dict = [{'source_varg_seq': input_events, 'target_varg_seq': cand, "insert_pos": insert_pos} for cand, insert_pos in batch_cands]
            outs = self.predict_batch_json(batch_json_dict)
            candidate_scores += [o['seq_score'] for o in outs]
            
        topk_candidates = self.get_topk_candidates(candidates, candidate_scores, beams)

        def chain_str(chain):
            texts = []
            for varg in chain:
                texts.append("<EVENT> " + " ".join(varg['V_toks']) + " <ARGS> " + varg['Description'])
            return texts
        def get_orig_chain(chain, insert_pos):
            assert insert_pos < len(chain)
            # last one is the unseen event
            rotate_split_idx = (len(chain) - 1) - insert_pos
            varg_seq_suffix = chain[:rotate_split_idx]
            varg_seq_prefix = chain[rotate_split_idx:]

            return varg_seq_prefix + varg_seq_suffix
        return {"gold_vargs":       gold_varg_seq,
                "input_vargs":      input_events,
                "unseen_vargs":      [unseen_event],
                "beam_pred":        [{'pred_vargs': get_orig_chain(cand, insert_pos),
                                      'pred_repr': chain_str(get_orig_chain(cand, insert_pos)),
                                      'score': score,
                                      'pred_is_neg_chain': False} 
                                     for (cand, insert_pos), score in topk_candidates],
                "all_beam_scores": sorted(candidate_scores, reverse=True),
                "best_pos_score": max([score for i, score in enumerate(candidate_scores)])
                }


def predict_on_unseen_events(data, predictor, file=sys.stdout):
    def chain_str(chain):
        texts = []
        for varg in chain:
            if not 'Description' in varg:
                varg['Description'] = " ".join(get_flatten_varg_toks(varg))
            texts.append("<EVENT> " + " ".join(varg['V_toks']) + " <ARGS> " + varg['Description'])
        return texts
    output = predictor.predict_json(data)
    print('---'*3, file=file)
    print("gold_repr:", file=file)
    for r in chain_str(output['gold_vargs']):
        print(r, file=file)
    print(file=file)
    print('---'*3, file=file)
    print("input_repr:", file=file)
    for r in chain_str(output['input_vargs']):
        print(r, file=file)
    print(file=file)
    print('---'*3, file=file)
    print("unseen_repr:", file=file)
    for r in chain_str(output['unseen_vargs']):
        print(r, file=file)
    print('---'*3, file=file)
    print("Max: {:.4f} - Min: {:.4f} - Mean: {:.4f} - Std: {:.4f} - Best POS: {:.4f}".format(np.max(output['all_beam_scores']), np.min(output['all_beam_scores']), np.mean(output['all_beam_scores']), np.std(output['all_beam_scores']), output["best_pos_score"]), file=file)
    beam_matches = []
    for b_idx, pred in enumerate(output['beam_pred']):
        match =  [varg['Description'] for varg in pred['pred_vargs']] == [varg['Description'] for varg in output['gold_vargs']]
        beam_matches.append(match)
        print("Beam {:d} (gold: {} - score: {:.4f})".format(b_idx, match, pred['score']), file=file)
        for r in pred['pred_repr']:
            print(r, file=file)
        print(file=file)
    return beam_matches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test the predictor above')

    parser.add_argument('--archive-path', type=str, required=True, help='path to trained archive file')
    parser.add_argument('--predictor', type=str, required=True, help='name of predictor')
    parser.add_argument('--weights-file', type=str,
                        help='a path that overrides which weights file to use')
    parser.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')
    parser.add_argument('-o', '--overrides', type=str, default="",
                        help='a JSON structure used to override the experiment configuration')
    parser.add_argument('--include-package',
                        type=str,
                        action='append',
                        default=[],
                        help='additional packages to include')
    parser.add_argument('--input-path', type=str, nargs='+', help='input data')
    parser.add_argument('--beams', type=int, help='beam size', default=1)
    parser.add_argument('--num_instances', type=int, default=-1,
                        help='number of instances to process')
    parser.add_argument('--chain_len_min', type=int, help='minimum length of chains', default=0)
    parser.add_argument('--baseline', type=str, default=None, choices=["random", "gpt2", "gpt2-medium", "gpt2-large", "infilling-gpt2", "infilling-gpt2-medium", "infilling-gpt2-large"],
                        help='use baseline model instead')

    args = parser.parse_args()

    # Load modules
    for package_name in args.include_package:
        import_module_and_submodules(package_name)

    check_for_gpu(args.cuda_device)
    if args.baseline is None:
        archive = load_archive(args.archive_path,
                               weights_file=args.weights_file,
                               cuda_device=args.cuda_device,
                               overrides=args.overrides)

        predictor = Predictor.from_archive(archive, args.predictor)
    else:
        baselines = {
            'gpt2': GPT2baseline,
            'gpt2-medium': GPT2baseline,
            'gpt2-large': GPT2baseline,
            'infilling-gpt2': InfillingGPT2baseline,
            'infilling-gpt2-medium': InfillingGPT2baseline,
            'infilling-gpt2-large': InfillingGPT2baseline,
            'random': RandomBaseline,
        }
        predictor = baselines[args.baseline](args.baseline[10:] if args.baseline.startswith("infilling") else args.baseline, args.cuda_device)

    if args.input_path is None:
        test_json = {
            "events": V_ARGS_string_to_varg_seq("<EVENT> died <ARGS> Durer's father died in 1502 <EVENT> died <ARGS> Durer's mother died in 1513"),
            "unseen_events": V_ARGS_string_to_varg_seq("<EVENT> became <ARGS> Durer's mother became depressed"),
            'beams': args.beams,
        }
        ans = V_ARGS_string_to_varg_seq("<EVENT> died <ARGS> Durer's father died in 1502 <EVENT> became <ARGS> Durer's mother became depressed <EVENT> died <ARGS> Durer's mother died in 1513"),
        '''
        test_json = {
            'events': V_ARGS_string_to_varg_seq('<EVENT> slipped <ARGS> She slipped on a crack that was on one of the concrete tiles <EVENT> fell <ARGS> She fell on the ground'),
            'unseen_events': V_ARGS_string_to_varg_seq('<EVENT> scattered <ARGS> her belongings scattered'),
            'beams': args.beams,
        }
        ans = V_ARGS_string_to_varg_seq('<EVENT> slipped <ARGS> She slipped on a crack that was on one of the concrete tiles <EVENT> fell <ARGS> She fell on the ground <EVENT> scattered <ARGS> her belongings scattered'),
        '''
        '''
        test_json = {
            'events': V_ARGS_string_to_varg_seq('<EVENT> suffered <ARGS> Therefore Louis suffered from cedar allergies for three long months <EVENT> move <ARGS> Louis move to a different state <EVENT> moved <ARGS> Louis just moved to Texas'),
            'unseen_events': V_ARGS_string_to_varg_seq(''),
            'beams': args.beams,
        }
        ans = None
        '''
        '''
        test_json = {
            'events': V_ARGS_string_to_varg_seq('<EVENT> used <ARGS> Once it was cut down we used a cart to bring it to the car <EVENT> bring <ARGS> we bring it to the car <EVENT> cut <ARGS> it cut down'),
            'unseen_events': V_ARGS_string_to_varg_seq('<EVENT> took <ARGS> Then we took the tree home in our trunk'),
            'beams': args.beams,
        }
        ans = None
        '''
        predict_on_unseen_events(test_json, ans, predictor)
    else:
        data = []
        for path_regex in args.input_path:
            for path in sorted(glob.glob(path_regex)):
                with open(path, 'rb') as f:
                    data += pickle.load(f)

        print(len(data))
        seen_ch_id = set()
        unique_data = []
        for d in data:
            ch_id = d['chain_id'] if not 'aug_metadata' in d or d['aug_metadata'] is None else d['aug_metadata']['source_chain_id']
            if not ch_id in seen_ch_id:
                unique_data.append(d)
                seen_ch_id.add(ch_id)
        data = unique_data
        print(len(data))

        if args.num_instances > 0:
            data = data[:args.num_instances]

        if args.chain_len_min > 0:
            data = [d for d in data if len(d['varg_seq']) >= args.chain_len_min]

        total_top2_correct = 0.
        total_correct = 0.
        total_examples = 0

        unseen_idxs = [random.choice(range(len(d["varg_seq"]))) for d in data]
        print(len(data))
        print(sum([1 / len(d["varg_seq"]) for d in data]) / len(data))
        random.seed(444)
        for d_idx, d in enumerate(tqdm(data)):
            d = deepcopy(d)
            test_json = {
                "example": d,
                "beams": args.beams,
                "unseen_idx": unseen_idxs[d_idx]
            }
            beam_matches = predict_on_unseen_events(test_json, predictor)
            total_examples += 1
            total_top2_correct += int(any(beam_matches[:2]))
            total_correct += int(beam_matches[0])
        print("Avg EM: {:.4f} ({:.4f} / {:d})".format(total_correct / total_examples, total_correct, total_examples))
        print("Avg top2 EM: {:.4f} ({:.4f} / {:d})".format(total_top2_correct / total_examples, total_top2_correct, total_examples))
