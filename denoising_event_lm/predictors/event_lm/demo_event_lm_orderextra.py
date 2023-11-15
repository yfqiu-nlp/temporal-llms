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
from copy import deepcopy

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
from denoising_event_lm.models.event_lm.seq2seq import get_flatten_varg_toks


DEBUG = False

BATCH_SIZE = 16

WORD_TOKENIZER = SpacyTokenizer(pos_tags=False, keep_spacy_tokens=False)

with open(os.path.join(os.getcwd(), "released_data_models/data/KB_all_docs_vocab.pkl"), 'rb') as f:
    VOCAB_CNTS = pickle.load(f)
    VOCAB_CNTS = {k: v for k, v in VOCAB_CNTS.items() if v > 0}
    TOTAL_VOCAB_CNTS = sum(v for k, v in VOCAB_CNTS.items())

#random.seed(2020)


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


@Predictor.register('demo_denoising_event_lm_orderextra_predictor')
class DemoDenosingEventLMOrderExtraPredictor(Predictor):
    @overrides
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        """
        Override the original init function to load the dataset to memory for demo
        """
        self._model = model
        self._dataset_reader = dataset_reader
        self._dataset_reader._event_del_prob = 0.
        self._dataset_reader._event_del_all_prob = 0.0
        self._dataset_reader._allow_empty_events = False
        self._dataset_reader._event_keep_num = -1
        self._dataset_reader._neg_chain_len_min = None

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
        Expects JSON object as ``{"example": dict,
                                  "beams": BEAM_int,
                                  "unseen_idx": int}``
        """
        start_time = time.time()
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
        if DEBUG:
            print("input events:")
            print(input_events)
            print("unseen events:")
            print(unseen_events)

        candidates = self.get_all_candidate_chains(input_events, unseen_event)
        random.shuffle(input_events)
        candidate_scores = []
        for b_start in range(0, len(candidates), BATCH_SIZE):
            batch_cands = candidates[b_start:b_start+BATCH_SIZE]
            batch_json_dict = [{'source_varg_seq': input_events, 'target_varg_seq': cand} for cand in batch_cands]
            outs = self.predict_batch_json(batch_json_dict)
            candidate_scores += [o['seq_score'] for o in outs]
            
        topk_candidates = self.get_topk_candidates(candidates, candidate_scores, beams)

        if DEBUG:
            print("fin time:", time.time() - start_time)
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
