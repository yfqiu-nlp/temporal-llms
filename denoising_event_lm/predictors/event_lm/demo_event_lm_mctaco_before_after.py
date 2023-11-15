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


DEBUG = False

BATCH_SIZE = 16

if __name__ == '__main__':
    random.seed(2020)


@Predictor.register('demo_denoising_event_lm_mctaco_before_after_predictor')
class DemoDenosingEventLMonInputPredictor(Predictor):
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

    def get_all_candidate_chains(self, input_events, unseen_events):
        """
        input_events: [varg_dict]
        unseen_events: [varg_dict]
        """
        all_events = input_events + unseen_events # list of varg dict
        candidates = []
        for p in permutations(range(len(all_events))):
            candidates.append([all_events[i] for i in p])
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
        Expects JSON object as ``{"events": List[ varg_dict ],
                                  "cand_event": varg_dict,
                                  "beams": BEAM_int,
                                  "feed_unseen": FEED_UNSEEN_bool}``
        """
        start_time = time.time()
        input_events = inputs['events']
        cand_event = inputs['cand_event']
        beams = int(inputs['beams'])
        feed_unseen = bool(distutils.util.strtobool(inputs['feed_unseen'])) if type(inputs['feed_unseen']) == str else bool(inputs['feed_unseen'])

        candidates = self.get_all_candidate_chains(input_events, [cand_event])
        candidate_scores = []
        if feed_unseen:
            input_events = input_events + [cand_event]
        #random.shuffle(input_events)
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
        return {"input_vargs":       input_events,
                "unseen_vargs":      [cand_event],
                "beam_pred":        [{'pred_vargs': cand,
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
