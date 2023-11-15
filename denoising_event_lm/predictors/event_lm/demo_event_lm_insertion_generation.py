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
from typing import List, Callable
import sys
import os
from copy import deepcopy
from lemminflect import getLemma, getAllInflections

from denoising_event_lm.predictors.data_visualization.temporal_data_visualizer import get_doc2event_map, get_digraph_template

sys.path.append(os.getcwd())
from denoising_event_lm.models.event_lm.seq2seq import get_flatten_varg_toks


if __name__ == '__main__':
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


@Predictor.register('demo_denoising_event_lm_insertion_generation_predictor')
class DemoDenosingEventLMInsertionGenerationPredictor(Predictor):
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

    @overrides
    def _json_to_instance(self, hotpot_dict_instance: JsonDict) -> Instance:
        example = hotpot_dict_instance['example'] # dict
        replace_idx = hotpot_dict_instance.get('replace_idx', None) # int
        insert_idx = hotpot_dict_instance.get('insert_idx', None) # int, if gold: A B C D, insert_idx: 2 => A B E C D
        force_suffix = hotpot_dict_instance['force_suffix'] # bool
        override_decode_kwargs = hotpot_dict_instance['override_decode_kwargs'] # dict

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

        # get the input varg seq
        if replace_idx is not None:
            input_varg_seq = [gold_varg_seq[i] for i in range(len(gold_varg_seq)) if not i == replace_idx]
            insert_idx = replace_idx
            num_output_events = len(gold_varg_seq)
        elif insert_idx is not None:
            input_varg_seq = gold_varg_seq
            num_output_events = len(gold_varg_seq) + 1

        target_prefix_varg_seq = [input_varg_seq[i] for i in range(len(input_varg_seq)) if i < insert_idx]
        target_suffix_varg_seq = [input_varg_seq[i] for i in range(len(input_varg_seq)) if i >= insert_idx]

        varg2eventtag = self._dataset_reader.get_varg2eventtag(input_varg_seq)
        source_str = self._dataset_reader._get_source_strings(input_varg_seq, varg2eventtag)

        source_str = self._dataset_reader._source_prefix + source_str
        source_encodes = self._dataset_reader.tokenize_text(source_str)


        # get the target prefix str
        target_prefix_str = self._dataset_reader._get_target_strings(target_prefix_varg_seq, varg2eventtag) + self._dataset_reader._event_sep
        if len(self._dataset_reader._neg_chain_identifiers) > 0:
            target_prefix_str = self._dataset_reader._pos_chain_prefix + target_prefix_str

        # preprocess target prefix strings. Not adding target_suffix since they are decoding prefix
        if len(target_prefix_str) > 0:
            target_prefix_str = self._dataset_reader._target_prefix + target_prefix_str
        else:
            # remove the trailing space
            target_prefix_str = self._dataset_reader._target_prefix[:-1]
        # tokenize the target prefix.
        target_prefix_encodes = self._dataset_reader.tokenize_text(target_prefix_str)

        # remove eos from target prefix
        assert target_prefix_encodes["input_ids"][-1] == self._dataset_reader._tokenizer.eos_token_id
        target_prefix_encodes["input_ids"] = target_prefix_encodes["input_ids"][:-1]
        assert all(not tok_id == self._dataset_reader._tokenizer.eos_token_id for tok_id in target_prefix_encodes["input_ids"])
        if self._dataset_reader._return_token_type_ids:
            target_prefix_encodes["token_type_ids"] = target_prefix_encodes["token_type_ids"][:-1]
        if self._dataset_reader._return_attention_mask:
            target_prefix_encodes["attention_mask"] = target_prefix_encodes["attention_mask"][:-1]


        # get the target suffix str
        target_suffix_str = self._dataset_reader._get_target_strings(target_suffix_varg_seq, varg2eventtag)

        # tokenize the target suffix. # TODO: ingore ``self._dataset_reader._target_suffix`` for now
        target_suffix_encodes = self._dataset_reader.tokenize_text(target_suffix_str)

        # remove sos from target suffix
        assert target_suffix_encodes["input_ids"][0] == self._dataset_reader._tokenizer.bos_token_id
        target_suffix_encodes["input_ids"] = target_suffix_encodes["input_ids"][1:]
        assert all(not tok_id == self._dataset_reader._tokenizer.bos_token_id for tok_id in target_suffix_encodes["input_ids"]), target_suffix_encodes["input_ids"]
        if self._dataset_reader._return_token_type_ids:
            target_suffix_encodes["token_type_ids"] = target_suffix_encodes["token_type_ids"][1:]
        if self._dataset_reader._return_attention_mask:
            target_suffix_encodes["attention_mask"] = target_suffix_encodes["attention_mask"][1:]


        # get bad_verbs_ids
        bad_verbs_ids = [self._dataset_reader._tokenizer.encode(bad_word, add_prefix_space=False, add_special_tokens=False)[:1] 
                         for bad_word in [w[0] for varg in input_varg_seq for pos, w in getAllInflections(getLemma(varg['V_toks'][0], "VERB")[0], "VERB").items()]]
        bad_verbs_ids += [self._dataset_reader._tokenizer.encode(bad_word, add_prefix_space=True, add_special_tokens=False)[:1] 
                         for bad_word in [w[0] for varg in input_varg_seq for pos, w in getAllInflections(getLemma(varg['V_toks'][0], "VERB")[0], "VERB").items()]]

        additional_metadata = {
            "source_str": self._dataset_reader._tokenizer.decode(source_encodes["input_ids"]),
            "target_prefix_str": self._dataset_reader._tokenizer.decode(target_prefix_encodes["input_ids"]),
            "target_suffix_str": self._dataset_reader._tokenizer.decode(target_suffix_encodes["input_ids"]),
            "input_varg_seq": input_varg_seq,
            "target_prefix_varg_seq": target_prefix_varg_seq,
            "target_suffix_varg_seq": target_suffix_varg_seq,
            "gold_varg_seq": gold_varg_seq,
            "bad_verbs_ids": bad_verbs_ids,
            "ban_bad_verbs_event_idxs": [insert_idx],
            "num_output_events": num_output_events,
            "target_suffix_encodes": target_suffix_encodes if force_suffix else None,
            "target_suffix_start_event_idx": insert_idx+1 if force_suffix else None,
            "override_decode_kwargs": override_decode_kwargs
        }

        instance = self._dataset_reader.text_to_instance(source_encodes, target_prefix_encodes, additional_metadata)
        return instance

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        """
        Override this function for demo
        Expects JSON object as ``{"dataset": d,
                                  "instance_idx": idx}``
        """
        start_time = time.time()
        events, chains = self.demo_dataset[inputs['dataset']]
        idx = int(inputs['instance_idx']) % len(chains)
        replace_idx = int(inputs['replace_idx']) if 'replace_idx' in inputs else None
        insert_idx = int(inputs['insert_idx']) if 'insert_idx' in inputs else None
        instance = chains[idx]
        events = events[instance['doc_id']]
        output = self.predict({"example": instance,
                               "replace_idx": replace_idx,
                               "insert_idx": insert_idx,
                               "force_suffix": True,
                               "override_decode_kwargs": {}})
        print("pred time:", time.time() - start_time)
        pred_str            = output['prediction_str']
        pred_varg_seq       = output['prediction_varg_seq']
        beam_pred_str       = output['beam_prediction_str']
        beam_pred_varg_seq  = output['beam_prediction_varg_seq']
        source_str          = output['source_str']
        decode_prefix_str   = output['target_str']
        gold_varg_seq       = output['gold_varg_seq']
        input_varg_seq       = output['input_varg_seq']
        print("source_str:")
        print(source_str)
        print("decode_prefix_str:")
        print(decode_prefix_str)
        print("pred_str:")
        print(pred_str)

        for varg in gold_varg_seq:
            varg['Description'] = " ".join([tok for arg in get_varg_toks(varg) for tok in arg])
        for varg in input_varg_seq:
            varg['Description'] = " ".join([tok for arg in get_varg_toks(varg) for tok in arg])

        doc_text = events['text']
        doc_toks = events['tokens']
        eiid2events = events['eiid2events']
        events_edges = events['events_edges']
        vague_edges = events['vague_edges']
        all_doc_spans, doc_sp2eiid = get_doc2event_map(len(doc_toks), eiid2events, events_edges)
        graph_template = get_digraph_template(eiid2events, events_edges, [], [], [])
        print("fin time:", time.time() - start_time)
        def chain_str(chain):
            texts = []
            for varg in chain:
                texts.append("<EVENT> " + " ".join(varg['V_toks']) + " <ARGS> " + varg['Description'])
            return texts
        return {"eiids":            instance['eiids'],
                "chain_repr":       chain_str(gold_varg_seq),
                "input_repr":       chain_str(input_varg_seq),
                "beam_pred":        [{'pred_eiids': [""] * len(pred_varg_seq),
                                      'pred_repr': chain_str(pred_varg_seq),
                                      'prob': None} 
                                     for pred_varg_seq in beam_pred_varg_seq],
                "doc_id":           instance['doc_id'],
                "_id":              str(instance['doc_idx'])+'_#'+str(instance['_id'].split('#')[-1]),
                "doc_text":         doc_text,
                "doc_tokens":       doc_toks,
                "all_doc_spans":    all_doc_spans,
                "doc_sp2eiid":      doc_sp2eiid,
                "eiid2events":      eiid2events,
                "events_edges":     events_edges,
                "vague_edges":      vague_edges,
                "graph_template":   graph_template,
                }

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.generate_extra_events([instance])[0]
        return sanitize(outputs)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.generate_extra_events(instances)
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
            'example': dict with the same structure defined in the dataset_readder,
            'num_input_events': int,
            'keep_idxs': int,
            'generate_following_events': bool,
            'override_decode_kwargs': dict
        }
        """
        return self._predict_json(hotpot_dict_instance)
