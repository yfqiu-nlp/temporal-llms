from overrides import overrides
from allennlp.common.util import JsonDict
import json
import pickle
import glob
import math
from allennlp.data import DatasetReader, Instance
from allennlp.predictors.predictor import Predictor
from allennlp.models import Model
import numpy as np
import torch
import time
import traceback
from graphviz import Digraph


def load_data(path):
    try:
        with open(path) as f:
            dataset = json.load(f)
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
    return dataset


def get_doc2event_map(doc_len, eiid2events, events_edges):
    e_in_graph = set([eiid for eiid in events_edges.keys()]) | set([eiid for ends in events_edges.values() for eiid in ends])
    e_spans = [(e['tok_start'], e['tok_end'], eiid) for eiid, e in eiid2events.items() if eiid in e_in_graph]
    e_spans = sorted(e_spans)
    all_spans = []
    sp2eiid = []
    last_end = -1
    for s, e, eiid in e_spans:
        if s > last_end+1:
            all_spans.append((last_end+1, s-1))
            sp2eiid.append("")
        all_spans.append((s, e))
        sp2eiid.append(eiid)
        last_end = e
    if doc_len > last_end+1:
        all_spans.append((last_end+1, doc_len-1))
        sp2eiid.append("")
    return all_spans, sp2eiid


def get_digraph_template(eiid2events, events_edges, vague_edges, includes_edges, is_included_edges):
    g = Digraph()
    for start, ends in events_edges.items():
        for end in ends:
            g.edge(("[%s]\n" % start)+eiid2events[start]['event'],
                   ("[%s]\n" % end)+eiid2events[end]['event'])
    for start, end in vague_edges:
        g.edge(("[%s]\n" % start)+eiid2events[start]['event'],
               ("[%s]\n" % end)+eiid2events[end]['event'], label='vague', dir='none')
    for start, end in includes_edges:
        g.edge(("[%s]\n" % start)+eiid2events[start]['event'],
               ("[%s]\n" % end)+eiid2events[end]['event'], label='includes', dir='none')
    for start, end in is_included_edges:
        g.edge(("[%s]\n" % start)+eiid2events[start]['event'],
               ("[%s]\n" % end)+eiid2events[end]['event'], label='is_included', dir='none')
    return g.source


@Predictor.register('temporal_data_visualizer')
class TempDataVisualizer(Predictor):
    @overrides
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        """
        Override the original init function to load the dataset to memory for demo
        """
        # load the coref model and dataset_reader
        self._model = model
        self._dataset_reader = dataset_reader

        tb_dense_train = load_data('/scratch/cluster/j0717lin/data/TB_Dense/vis_data/tb_dense.json')
        matres_tb_train = load_data('/scratch/cluster/j0717lin/data/MATRES/vis_data/timebank.json')
        matres_ac_train = load_data('/scratch/cluster/j0717lin/data/MATRES/vis_data/aquaint.json')
        '''
        UDS_T_train = load_data('/scratch/cluster/j0717lin/data/DecompTime/vis_data/en-ud-train.json')
        UDS_T_dev = load_data('/scratch/cluster/j0717lin/data/DecompTime/vis_data/en-ud-dev.json')
        '''
        UDS_T_train = load_data('/scratch/cluster/j0717lin/data/DecompTime/vis_data_nocycle/en-ud-train.json')
        #UDS_T_dev = load_data('/scratch/cluster/j0717lin/data/DecompTime/vis_data_nocycle/en-ud-dev.json')
        UDS_T_dev = load_data('/scratch/cluster/j0717lin/temporal/UDST_test_avgconf/en-ud-dev.json')
        #UDS_T_dev = load_data('/scratch/cluster/j0717lin/temporal/UDST_test_maj/en-ud-dev.json')
        MEANTIME_trial = load_data('/scratch/cluster/j0717lin/data/SemEval-2015_task4/vis_data/meantime_trial.json')
        MEANTIME = load_data('/scratch/cluster/j0717lin/data/SemEval-2015_task4/vis_data/meantime.json')
        self.demo_dataset = {'tb_dense_full': tb_dense_train,
                             'tb_dense': tb_dense_train,
                             'matres_tb_train': matres_tb_train,
                             'matres_ac_train': matres_ac_train,
                             'UDS_T_train': UDS_T_train,
                             'UDS_T_dev': UDS_T_dev,
                             'MEANTIME_trial': MEANTIME_trial,
                             'MEANTIME': MEANTIME
                             }

    @overrides
    def _json_to_instance(self, hotpot_dict_instance: JsonDict) -> Instance:
        processed_instance = self._dataset_reader.process_raw_instance(hotpot_dict_instance)
        instance = self._dataset_reader.text_to_instance(*processed_instance)
        return instance

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        """
        Override this function for demo
        Expects JSON object as ``{"dataset": d,
                                  "instance_idx": idx}``
        """
        start_time = time.time()
        dataset = self.demo_dataset[inputs['dataset']]
        idx = int(inputs['instance_idx']) % len(dataset)
        d = dataset[idx]
        doc_text = d['text']
        doc_toks = d['tokens']
        eiid2events = d['eiid2events']
        events_edges = d['events_edges']
        vague_edges = d['vague_edges'] if inputs['dataset'].endswith('full') else []
        includes_edges = d.get('includes_edges', []) if inputs['dataset'].endswith('full') else []
        is_included_edges = d.get('is_included_edges', []) if inputs['dataset'].endswith('full') else []
        all_doc_spans, doc_sp2eiid = get_doc2event_map(len(doc_toks), eiid2events, events_edges)
        graph_template = get_digraph_template(eiid2events, events_edges, vague_edges, includes_edges, is_included_edges)
        # get coref predictions
        sents_sp = [(d['sents_tok_offset'][i], d['sents_tok_offset'][i+1]-1) for i in range(len(d['sents_tok_offset'])-1)] + \
            [(d['sents_tok_offset'][-1], len(doc_toks)-1)]
        sents = [doc_toks[s:e+1] for s, e in sents_sp]
        coref_ins = self._dataset_reader.text_to_instance(sents)
        coref_out = self.predict_instance(coref_ins)
        pred_clusters = coref_out['pred_clusters']
        return {"doc_text":         doc_text,
                "doc_tokens":       doc_toks,
                "all_doc_spans":    all_doc_spans,
                "doc_sp2eiid":      doc_sp2eiid,
                "eiid2events":      eiid2events,
                "events_edges":      events_edges,
                "vague_edges":      vague_edges,
                "graph_template":   graph_template,
                "pred_clusters":    pred_clusters,
                }

    def _predict_json(self, inputs: JsonDict) -> JsonDict:
        """
        Serve as the substitute for the original ``predict_json``
        """
        instance = self._json_to_instance(inputs)
        return self.predict_instance(instance)

    def predict(self, hotpot_dict_instance: JsonDict) -> JsonDict:
        """
        Expects JSON that has the same format of instances in Hotpot dataset
        """
        return self._predict_json(hotpot_dict_instance)
