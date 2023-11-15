import json
import sys
import os
import glob
import pandas as pd
import numpy as np
import torch
from torch import nn
import random
from itertools import combinations, permutations
from copy import deepcopy
from allennlp.predictors.predictor import Predictor
import heapq

import argparse

from allennlp.common.checks import check_for_gpu
from allennlp.common.util import import_module_and_submodules
from allennlp.models.archival import load_archive

from transformers import AutoTokenizer, AutoModelWithLMHead
sys.path.append(os.getcwd())

from denoising_event_lm.models.event_lm.seq2seq import get_flatten_varg_toks
from denoising_event_lm.models.event_lm.seq2seq import V_ARGS_string_to_varg_seq

if __name__ == '__main__':
    random.seed(2020)

BATCH_SIZE = 16

class Randombaseline:
    def __init__(self, model_name, cuda_device):
        pass

    def get_all_candidate_chains(self, input_events, unseen_events, constraints):
        """
        input_events: [varg_dict]
        unseen_events: [varg_dict]
        constraints: [(varg_dict, varg_dict), ...]
        """
        all_events = input_events + unseen_events # list of varg dict
        all_events_dict = {str(e): i for i, e in enumerate(all_events)}
        if not len(all_events_dict) == len(all_events):
            raise ValueError("duplicate events: {}".format(repr(all_events)))
        candidates = []
        if_fulfill_constraints = []
        constraints = [(all_events_dict[str(e1)], all_events_dict[str(e2)]) for e1, e2 in constraints]
        for p in permutations(range(len(all_events))):
            if len(constraints) == 0 or check_chain_fulfill_constraints(p, constraints):
                if_fulfill_constraints.append(True)
            else:
                if_fulfill_constraints.append(False)
            candidates.append([all_events[i] for i in p])
        return candidates, if_fulfill_constraints

    def predict_json(self, inputs):
        """
        Override this function for demo
        Expects JSON object as ``{"events": INPUTS_str,
                                  "unseen_events": UNSEEN_str,
                                  "constraints": CONSTS_str,
                                  "beams": BEAM_int,
                                  "feed_unseen": FEED_UNSEEN_bool,
                                  "perm_normalize": PERM_NORMALIZE_bool,
                                  "unigram_normalize": UNIGRAM_NORMALIZE_bool}``
        """
        input_events = V_ARGS_string_to_varg_seq(inputs['events'], add_event_sep_entry=False)
        unseen_events = V_ARGS_string_to_varg_seq(inputs['unseen_events'], add_event_sep_entry=False) if inputs['unseen_events'] else []
        constraints = get_constraints(inputs['constraints']) if inputs['constraints'] else []
        beams = int(inputs['beams'])
        feed_unseen = bool(distutils.util.strtobool(inputs['feed_unseen'])) if type(inputs['feed_unseen']) == str else bool(inputs['feed_unseen'])
        perm_normalize = bool(distutils.util.strtobool(inputs['perm_normalize'])) if type(inputs['perm_normalize']) == str else bool(inputs['perm_normalize'])
        unigram_normalize = bool(distutils.util.strtobool(inputs['unigram_normalize'])) if type(inputs['unigram_normalize']) == str else bool(inputs['unigram_normalize'])

        candidates, if_fulfill_constraints = self.get_all_candidate_chains(input_events, unseen_events, constraints)
        random.shuffle(candidates)

        candidates = candidates[:beams]
        topk_candidates = [(cand, 0.) for cand in candidates]

        def chain_str(chain):
            texts = []
            for varg in chain:
                texts.append("<EVENT> " + " ".join(varg['V_toks']) + " <ARGS> " + varg['Description'])
            return texts
        return {"input_repr":       chain_str(input_events),
                "unseen_repr":     chain_str(unseen_events),
                "beam_pred":        [{'pred_repr': chain_str(cand),
                                      'score': score,
                                      'pred_is_neg_chain': False} 
                                     for cand, score in topk_candidates],
                "all_beam_scores": [0.0]*len(candidates),
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

    def get_all_candidate_chains(self, input_events, unseen_events, constraints):
        """
        input_events: [varg_dict]
        unseen_events: [varg_dict]
        constraints: [(varg_dict, varg_dict), ...]
        """
        all_events = input_events + unseen_events # list of varg dict
        all_events_dict = {str(e): i for i, e in enumerate(all_events)}
        if not len(all_events_dict) == len(all_events):
            raise ValueError("duplicate events: {}".format(repr(all_events)))
        candidates = []
        if_fulfill_constraints = []
        constraints = [(all_events_dict[str(e1)], all_events_dict[str(e2)]) for e1, e2 in constraints]
        for p in permutations(range(len(all_events))):
            if len(constraints) == 0 or check_chain_fulfill_constraints(p, constraints):
                if_fulfill_constraints.append(True)
            else:
                if_fulfill_constraints.append(False)
            candidates.append([all_events[i] for i in p])
        return candidates, if_fulfill_constraints

    def get_topk_candidates(self, candidates, if_fulfill_constraints, scores, beams):
        min_heap = []
        for i, (cand, ful, score) in enumerate(zip(candidates, if_fulfill_constraints, scores)):
            if ful:
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
        Expects JSON object as ``{"events": INPUTS_str,
                                  "unseen_events": UNSEEN_str,
                                  "constraints": CONSTS_str,
                                  "beams": BEAM_int,
                                  "feed_unseen": FEED_UNSEEN_bool,
                                  "perm_normalize": PERM_NORMALIZE_bool,
                                  "unigram_normalize": UNIGRAM_NORMALIZE_bool}``
        """
        input_events = V_ARGS_string_to_varg_seq(inputs['events'], add_event_sep_entry=False)
        unseen_events = V_ARGS_string_to_varg_seq(inputs['unseen_events'], add_event_sep_entry=False) if inputs['unseen_events'] else []
        constraints = get_constraints(inputs['constraints']) if inputs['constraints'] else []
        beams = int(inputs['beams'])
        feed_unseen = bool(distutils.util.strtobool(inputs['feed_unseen'])) if type(inputs['feed_unseen']) == str else bool(inputs['feed_unseen'])
        perm_normalize = bool(distutils.util.strtobool(inputs['perm_normalize'])) if type(inputs['perm_normalize']) == str else bool(inputs['perm_normalize'])
        unigram_normalize = bool(distutils.util.strtobool(inputs['unigram_normalize'])) if type(inputs['unigram_normalize']) == str else bool(inputs['unigram_normalize'])

        candidates, if_fulfill_constraints = self.get_all_candidate_chains(input_events, unseen_events, constraints)
        random.shuffle(input_events)
        candidate_scores = []
        for b_start in range(0, len(candidates), BATCH_SIZE):
            batch_cands = candidates[b_start:b_start+BATCH_SIZE]
            batch_json_dict = [{'source_varg_seq': input_events, 'target_varg_seq': cand} for cand in batch_cands]
            outs = self.predict_batch_json(batch_json_dict)
            candidate_scores += [o['seq_score'] for o in outs]
            
        topk_candidates = self.get_topk_candidates(candidates, if_fulfill_constraints, candidate_scores, beams)

        def chain_str(chain):
            texts = []
            for varg in chain:
                texts.append("<EVENT> " + " ".join(varg['V_toks']) + " <ARGS> " + varg['Description'])
            return texts
        return {"input_repr":       chain_str(input_events),
                "unseen_repr":     chain_str(unseen_events),
                "beam_pred":        [{'pred_repr': chain_str(cand),
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


DIE_EVENT_TEMPLATES = ["<EVENT> died <ARGS> Tom died {}", "<EVENT> died <ARGS> Jake died {}", "<EVENT> died <ARGS> Mike died {}", "<EVENT> died <ARGS> David died {}"]
LIFE_EVENT_TEMPLATES = ["<EVENT> went <ARGS> Durer went to a supermarket {}", "<EVENT> bought <ARGS> Durer bought a book at a shop {}", "<EVENT> took <ARGS> Durer took a photo in front of a museum {}", "<EVENT> jogged <ARGS> Durer jogged in a park {}"]

def get_random_years(n):
    samples = random.sample(range(1000, 2100), n)
    orders = sorted(range(n), key=lambda x: int(samples[x]))
    return ["in "+str(s) for s in samples], orders

def get_random_months(n):
    MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    MONTHS = {k: v for k, v in zip(MONTHS, range(12))}
    samples = random.sample(MONTHS.keys(), n)
    orders = sorted(range(n), key=lambda x: MONTHS[samples[x]])
    return ["in "+s for s in samples], orders

def get_random_weekdays(n):
    WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    WEEKDAYS = {k: v for k, v in zip(WEEKDAYS, range(7))}
    samples = random.sample(WEEKDAYS.keys(), n)
    orders = sorted(range(n), key=lambda x: WEEKDAYS[samples[x]])
    return ["on "+s for s in samples], orders

def get_random_daytimes(n):
    DAYTIMES = ["in the morning", "in the afternoon", "in the evening", "at night"]
    DAYTIMES = {k: v for k, v in zip(DAYTIMES, range(4))}
    samples = random.sample(DAYTIMES.keys(), n)
    orders = sorted(range(n), key=lambda x: DAYTIMES[samples[x]])
    return samples, orders

def get_random_hours_12(n):
    hour_samples = [str(h) for h in random.sample(range(1, 13), n)]
    min_samples = ["{:02d}".format(m) for m in random.sample(range(0, 60), n)]
    half_samples = random.choices(['am', 'pm'], k=n)
    samples = ["at {}:{} {}".format(h, m, half) for h, m, half in zip(hour_samples, min_samples, half_samples)]
    orders =  sorted(range(n), key=lambda x: (int(hour_samples[x])%12)*60+int(min_samples[x])+12*60*(half_samples[x]=='pm'))
    return samples, orders

def get_random_hours_24(n):
    hour_samples = [str(h) for h in random.sample(range(0, 24), n)]
    min_samples = ["{:02d}".format(m) for m in random.sample(range(0, 60), n)]
    samples = ["at {}:{}".format(h, m) for h, m in zip(hour_samples, min_samples)]
    orders =  sorted(range(n), key=lambda x: int(hour_samples[x])*60+int(min_samples[x]))
    return samples, orders

def get_random_hours_mixed(n):
    is_12 = random.choices([0, 1], k=n)
    samples = []
    total_mins = []
    for i in range(n):
        if is_12[i] == 1:
            h = random.choice(range(1, 13))
            m = "{:02d}".format(random.choice(range(0, 60)))
            half = random.choice(['am', 'pm'])
            sample = "at {}:{} {}".format(h, m, half)
            tot_min = (int(h)%12)*60+int(m)+12*60*(half=='pm')
        else:
            h = random.choice(range(0, 24))
            m = "{:02d}".format(random.choice(range(0, 60)))
            sample = "at {}:{}".format(h, m)
            tot_min = int(h)*60+int(m)
        samples.append(sample)
        total_mins.append(tot_min)
    orders = sorted(range(n), key=lambda x: total_mins[x])
    return samples, orders


def get_before_pairs(seq):
    pairs = set((i, j) for i, j in combinations(seq, 2))
    return pairs


def pairwise_accuracy(pred_seq, gold_seq):
    """ pairwise accuracy
    """
    gold_before_pairs = get_before_pairs(gold_seq) # in a chain, there are only `before` relations
    pred_before_pairs = get_before_pairs(pred_seq)
    corr = len(gold_before_pairs & pred_before_pairs)
    pair_acc = corr / len(gold_before_pairs)
    return pair_acc


def predict_on_unseen_events(data, predictor, print_input_repr=True, file=sys.stdout):
    orders = data[1]
    data = data[0]
    output = predictor.predict_json(data)
    pred_chains = [r for r in output['beam_pred'][0]['pred_repr']]
    gold_chains = [output['input_repr'][o] for o in orders]
    em = pred_chains == gold_chains
    pair_acc = pairwise_accuracy(pred_chains, gold_chains)
    print('---'*3, file=file)
    print("EM", em)
    print("Pair Acc", pair_acc)
    if print_input_repr:
        print("input events:", file=file)
        for r in output['input_repr']:
            print(r, file=file)
        print(file=file)
    print("gold:", file=file)
    for o in orders:
        print(output['input_repr'][o], file=file)
    print(file=file)
    print("predictions:", file=file)
    print("Max: {:.4f} - Min: {:.4f} - Mean: {:.4f} - Std: {:.4f}".format(np.max(output['all_beam_scores']), np.min(output['all_beam_scores']), np.mean(output['all_beam_scores']), np.std(output['all_beam_scores'])), file=file)
    for b_idx, pred in enumerate(output['beam_pred']):
        print("Beam {:d} (score: {:.4f})".format(b_idx, pred['score']), file=file)
        for r in pred['pred_repr']:
            print(r, file=file)
        print(file=file)
    print('---'*3, file=file)
    return output['beam_pred'][0]['score'], em, pair_acc


def create_time_data(args):
    if args.eval_type == 'year':
        sample_func = get_random_years
    elif args.eval_type == 'month':
        sample_func = get_random_months
    elif args.eval_type == 'weekday':
        sample_func = get_random_weekdays
    elif args.eval_type == 'hour':
        if args.hour_type == '12':
            sample_func = get_random_hours_12
        elif args.hour_type == '24':
            sample_func = get_random_hours_24
        elif args.hour_type == 'mixed':
            sample_func = get_random_hours_mixed
        else:
            raise ValueError("Unknown hour type: "+args.hour_type)
    elif args.eval_type == 'daytime':
        sample_func = get_random_daytimes
    else:
        raise ValueError("Unknown eval type: "+args.eval_type)

    if args.event_type == 'die':
        templates = DIE_EVENT_TEMPLATES
    elif args.event_type == 'life':
        templates = LIFE_EVENT_TEMPLATES
    else:
        raise ValueError("Unknown event type: "+args.event_type)

    data = []
    for i in range(args.num_data):
        samples, orders = sample_func(args.num_events)
        input_events = [e.format(t) for e, t in zip(templates[:args.num_events], samples)]
        data.append(
            ({
                "events": " ".join(input_events),
                "unseen_events": "",
                "constraints": "",
                "beams": args.beams,
                'feed_unseen': args.feed_unseen,
                'perm_normalize': args.perm_normalize,
                'unigram_normalize': args.unigram_normalize
            }, orders)
        )

    return data


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
    parser.add_argument('--beams', type=int, help='beam size', default=1)
    parser.add_argument('--feed-unseen', action='store_true', help='whether to feed unseen events as inputs', default=False)
    parser.add_argument('--perm_normalize', action='store_true', help='whether normalize scores by permutations', default=False)
    parser.add_argument('--unigram_normalize', action='store_true', help='whether unigram normalization', default=False)
    parser.add_argument('--eval-type', required=True, type=str, help='time expression to evaluate', choices=['year', 'month', 'weekday', 'date', 'hour', 'daytime'])
    parser.add_argument('--hour-type', type=str, default='12', help='in 12/24 hours format', choices=['12', '24', 'mixed'])
    parser.add_argument('--event-type', type=str, default='die', help='event type', choices=['die', 'life'])
    parser.add_argument('--num_events', type=int, default=3, help='number of events to input', choices=[2, 3, 4])
    parser.add_argument('--num_data', type=int, default=10, help='number of data to evaluate')
    parser.add_argument('--baseline', type=str, default=None, choices=["gpt2", "gpt2-medium", "gpt2-large", 'random'],
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
                'random': Randombaseline
            }
        predictor = baselines[args.baseline](args.baseline[10:] if args.baseline.startswith("infilling") else args.baseline, args.cuda_device)

    data = create_time_data(args)

    total_em = 0
    total_pair_acc = 0.0
    for d in data:
        d = deepcopy(d)
        score, em, pair_acc = predict_on_unseen_events(d, predictor)
        total_em += em
        total_pair_acc += pair_acc
    print("Avg EM: {:.4f} - Avg Pairwise Acc: {:.4f}".format(total_em / len(data), total_pair_acc / len(data)))
