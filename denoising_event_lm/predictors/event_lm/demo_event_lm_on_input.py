from overrides import overrides
from allennlp.common.util import JsonDict, sanitize
import json
import pickle
from allennlp.data import DatasetReader, Instance
from allennlp.predictors.predictor import Predictor
from allennlp.models import Model
import numpy as np
import torch
import time
import random
from itertools import permutations
import heapq

from typing import List, Callable
import argparse
import os
import sys
from distutils.util import strtobool

from allennlp.common.checks import check_for_gpu
from allennlp.common.util import import_module_and_submodules
from allennlp.models.archival import load_archive
from allennlp.data.tokenizers import SpacyTokenizer

sys.path.append(os.getcwd())
from denoising_event_lm.models.event_lm.seq2seq import V_ARGS_string_to_varg_seq


DEBUG = False

BATCH_SIZE = 16

WORD_TOKENIZER = SpacyTokenizer(pos_tags=False, keep_spacy_tokens=False)

with open(os.path.join(os.getcwd(), "released_data_models/data/KB_all_docs_vocab.pkl"), 'rb') as f:
    VOCAB_CNTS = pickle.load(f)
    VOCAB_CNTS = {k: v for k, v in VOCAB_CNTS.items() if v > 0}
    TOTAL_VOCAB_CNTS = sum(v for k, v in VOCAB_CNTS.items())

random.seed(2020)


def get_varg_toks(varg, arg_order='text'):
    varg_toks = [varg['V_toks']] + varg['ARGS_toks']
    varg_span = [varg['V_span']] + varg['ARGS_span']
    varg_type = ['V'] + [arg_type for arg_type in varg['ARGS_type']]
    assert len(varg_toks) == len(varg_span) and len(varg_toks) == len(varg_type)
    indices = list(range(len(varg_toks)))
    if arg_order == 'srl':
        pass
    elif arg_order == 'text':
        indices = sorted(indices, key=lambda x: varg_span[x])
    varg_toks = [varg_toks[i] for i in indices]
    varg_type = [varg_type[i] for i in indices]
    return varg_toks


def get_constraints(constraints):
    constraints = constraints.split("<CONST>")[1:]
    pairs = []
    for const in constraints:
        e1, *e2 = const.split("<BEFORE>")
        e1 = V_ARGS_string_to_varg_seq(e1.strip(), add_event_sep_entry=False)
        e2 = V_ARGS_string_to_varg_seq(" <BEFORE> ".join(e2).strip(), add_event_sep_entry=False)
        assert len(e1) == 1
        assert len(e2) == 1
        pairs.append((e1[0], e2[0]))
    return pairs


def check_chain_fulfill_constraints(events, constraints):
    def fulfill_constraint(e1, e2):
        for e in events:
            if e == e1:
                return True
            elif e == e2:
                return False
    return all(fulfill_constraint(e1, e2) for e1, e2 in constraints)


@Predictor.register('demo_denoising_event_lm_on_input_predictor')
class DemoDenosingEventLMonInputPredictor(Predictor):
    @overrides
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        """
        Override the original init function to load the dataset to memory for demo
        """
        self._model = model
        self._dataset_reader = dataset_reader
        self._dataset_reader._event_del_prob = 0.
        self._dataset_reader._event_keep_num = 0

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

    @overrides
    def _json_to_instance(self, hotpot_dict_instance: JsonDict) -> Instance:
        source_varg_seq = hotpot_dict_instance['source_varg_seq']
        target_varg_seq = hotpot_dict_instance['target_varg_seq']

        #random.shuffle(source_varg_seq)
        varg2eventtag = self._dataset_reader.get_varg2eventtag(source_varg_seq)
        source_str = self._dataset_reader._get_source_strings(source_varg_seq, varg2eventtag)
        source_str = self._dataset_reader._source_prefix + source_str
        source_encodes = self._dataset_reader.tokenize_text(source_str)

        target_str = self._dataset_reader._get_target_strings(target_varg_seq, varg2eventtag)
        if len(self._dataset_reader._neg_chain_identifiers) > 0:
            target_str = self._dataset_reader._pos_chain_prefix + target_str
        target_str = self._dataset_reader._target_prefix + target_str + self._dataset_reader._target_suffix
        target_encodes = self._dataset_reader.tokenize_text(target_str)

        instance = self._dataset_reader.text_to_instance(source_encodes, target_encodes)
        return instance

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
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
        start_time = time.time()
        input_events = V_ARGS_string_to_varg_seq(inputs['events'], add_event_sep_entry=False)
        unseen_events = V_ARGS_string_to_varg_seq(inputs['unseen_events'], add_event_sep_entry=False) if inputs['unseen_events'] else []
        constraints = get_constraints(inputs['constraints']) if inputs['constraints'] else []
        beams = int(inputs['beams'])
        feed_unseen = bool(distutils.util.strtobool(inputs['feed_unseen'])) if type(inputs['feed_unseen']) == str else bool(inputs['feed_unseen'])
        perm_normalize = bool(distutils.util.strtobool(inputs['perm_normalize'])) if type(inputs['perm_normalize']) == str else bool(inputs['perm_normalize'])
        unigram_normalize = bool(distutils.util.strtobool(inputs['unigram_normalize'])) if type(inputs['unigram_normalize']) == str else bool(inputs['unigram_normalize'])
        if DEBUG:
            print("input events:")
            print(input_events)
            print("unseen events:")
            print(unseen_events)
            print("constraints:")
            print(constraints)

        candidates, if_fulfill_constraints = self.get_all_candidate_chains(input_events, unseen_events, constraints)
        candidate_scores = []
        for b_start in range(0, len(candidates), BATCH_SIZE):
            batch_cands = candidates[b_start:b_start+BATCH_SIZE]
            if feed_unseen:
                batch_json_dict = [{'source_varg_seq': input_events+unseen_events, 'target_varg_seq': cand} for cand in batch_cands]
            else:
                batch_json_dict = [{'source_varg_seq': input_events, 'target_varg_seq': cand} for cand in batch_cands]
            outs = self.predict_batch_json(batch_json_dict)
            if unigram_normalize:
                # normalize by all p(w)
                norms = []
                for cand in batch_cands:
                    tokens = [str(tok) for varg in cand for tok in WORD_TOKENIZER.tokenize(varg['Description'])]
                    norm = sum(np.log(VOCAB_CNTS.get(tok, 1) / TOTAL_VOCAB_CNTS) for tok in tokens)
                    norms.append(norm)
                candidate_scores += [o['seq_score'] - norm for o, norm in zip(outs, norms)]
            else:
                candidate_scores += [o['seq_score'] for o in outs]

        if perm_normalize:
            candidate_scores = [np.exp(s) for s in candidate_scores]
            norm_const = sum(candidate_scores)
            candidate_scores = [s / norm_const for s in candidate_scores]
        else:
            norm_const = 1.
            
        topk_candidates = self.get_topk_candidates(candidates, if_fulfill_constraints, candidate_scores, beams)

        if DEBUG:
            print("fin time:", time.time() - start_time)
        def chain_str(chain):
            texts = []
            for varg in chain:
                texts.append("<EVENT> " + " ".join(varg['V_toks']) + " <ARGS> " + varg['Description'])
            return texts
        return {"input_repr":       chain_str(input_events),
                "unseen_repr":      chain_str(unseen_events),
                "beam_pred":        [{'pred_repr': chain_str(cand),
                                      'score': score,
                                      'pred_is_neg_chain': False} 
                                     for cand, score in topk_candidates],
                "all_beam_scores": sorted(candidate_scores, reverse=True),
                "best_pos_score": max([score for i, score in enumerate(candidate_scores) if if_fulfill_constraints[i]])
                }

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.compute_sequence_scores([instance])[0]
        return sanitize(outputs)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.compute_sequence_scores(instances)
        return sanitize(outputs)

    def _predict_json(self, inputs: JsonDict) -> JsonDict:
        """
        Serve as the substitute for the original ``predict_json``
        This function is only used for computing the score of a target events chain conditioned on input events
        """
        instance = self._json_to_instance(inputs)
        return self.predict_instance(instance)

    def predict(self, hotpot_dict_instance: JsonDict) -> JsonDict:
        """
        Expects JSON that has the same format of instances in Hotpot dataset
        This function is only used for computing the score of a target events chain conditioned on input events
        hotpot_dict_instance: 
        {
            'source_varg_seq': [varg_dict],
            'target_varg_seq': [varg_dict]
        }
        """
        return self._predict_json(hotpot_dict_instance)


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

    args = parser.parse_args()

    # Load modules
    for package_name in args.include_package:
        import_module_and_submodules(package_name)

    check_for_gpu(args.cuda_device)
    archive = load_archive(args.archive_path,
                           weights_file=args.weights_file,
                           cuda_device=args.cuda_device,
                           overrides=args.overrides)

    predictor = Predictor.from_archive(archive, args.predictor)

    test_json = {
        'events': '<EVENT> slipped <ARGS> She slipped on a crack that was on one of the concrete tiles <EVENT> fell <ARGS> She fell on the ground <EVENT> scattered <ARGS> her belongings scattered',
        'unseen_events': '',
        'constraints': '',
        'beams': 4,
        "feed_unseen": False,
        "perm_normalize": False,
        "unigram_normalize": False
    }
    output = predictor.predict_json(test_json)
    print('---'*3)
    print("input_repr:")
    for r in output['input_repr']:
        print(r)
    print('---'*3)
    print("unseen_repr:")
    for r in output['unseen_repr']:
        print(r)
    print('---'*3)
    for b_idx, pred in enumerate(output['beam_pred']):
        print("Beam {:d} ({:.4f}, {:})".format(b_idx, pred['score']), "POSITIVE" if pred['pred_is_neg_chain'] else "NEGATIVE")
        for r in pred['pred_repr']:
            print(r)
        print()
    print("\n\n")
