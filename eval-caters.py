from denoising_event_lm.training.metrics import metric_map
import sys
from denoising_event_lm.utils.constants import EVENT_TAG, POINTER_EVENT_TAGS, ARGS_TAG
from denoising_event_lm.data.dataset_readers.event_lm.event_seq2seq_transformer_reader import EventSeq2SeqTransformerReader
from denoising_event_lm.models.event_lm.seq2seq import EventLMTransformerSeq2Seq
import re
from scipy.optimize import linear_sum_assignment
from allennlp.training.metrics import Average
import json
import pandas
import pickle
from rouge_score import rouge_scorer
rouge_scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)

from copy import deepcopy

def normalize_arg_type(arg_type):
    if arg_type[0] in ['R', 'C']:
        return arg_type[2:]
    else:
        return arg_type

def get_flatten_varg_toks(varg):
    varg_toks = [varg['V_toks']] + varg['ARGS_toks']
    varg_span = [varg['V_span']] + varg['ARGS_span']
    varg_type = ['V'] + [normalize_arg_type(arg_type) for arg_type in varg['ARGS_type']]
    assert len(varg_toks) == len(varg_span) and len(varg_toks) == len(varg_type)
    indices = list(range(len(varg_toks)))
    # sort pred/args by their textual order
    indices = sorted(indices, key=lambda x: varg_span[x])
    varg_toks = [varg_toks[i] for i in indices]
    varg_type = [varg_type[i] for i in indices]
    flatten_toks = []
    for i, toks in enumerate(varg_toks):
        flatten_toks.extend(toks)
    return flatten_toks

def V_ARGS_string_to_varg_seq(varg_string, add_event_sep_entry=True):
    #vargs = varg_string.split(EVENT_TAG)[1:]
    regex = rf"({'|'.join([EVENT_TAG]+POINTER_EVENT_TAGS)})(.*?)(?={'|'.join(['$',EVENT_TAG]+POINTER_EVENT_TAGS)})"
    vargs = [(x.group(1), x.group(2)) for x in re.finditer(regex, varg_string)]
    varg_seq = []
    for event_sep, varg in vargs:
        v, *desc = varg.split(ARGS_TAG)
        desc = f" {ARGS_TAG} ".join(desc)
        if add_event_sep_entry:
            varg_seq.append(
                {
                    "V_toks": [v.strip()],
                    "Description": desc.strip(),
                    "EVENT_SEP": event_sep
                }
            )
        else:
            varg_seq.append(
                {
                    "V_toks": [v.strip()],
                    "Description": desc.strip()
                }
            )
    return varg_seq

def get_event_matching(varg_seq_a, varg_seq_b):
    # get description if needed: ARG0 Pred ARG1 ...
    if len(varg_seq_a) > 0 and not 'Description' in varg_seq_a[0]:
        for varg in varg_seq_a:
            varg['Description'] = " ".join(get_flatten_varg_toks(varg))
    if len(varg_seq_b) > 0 and not 'Description' in varg_seq_b[0]:
        for varg in varg_seq_b:
            varg['Description'] = " ".join(get_flatten_varg_toks(varg))

    # miximum weighted bipartite matching
    if len(varg_seq_a) > 0 and len(varg_seq_b) > 0:
        scores = [[0 for j in range(len(varg_seq_b))] for i in range(len(varg_seq_a))]
        for i in range(len(varg_seq_a)):
            for j in range(len(varg_seq_b)):
                e_sep_a = varg_seq_a[i]['EVENT_SEP']
                v_a = " ".join(varg_seq_a[i]['V_toks'])
                desc_a = varg_seq_a[i]['Description']
                e_sep_b = varg_seq_b[j]['EVENT_SEP']
                v_b = " ".join(varg_seq_b[j]['V_toks'])
                desc_b = varg_seq_b[j]['Description']
                scores[i][j] = float(e_sep_a == e_sep_b) * float(v_a == v_b) * rouge_scorer.score(desc_a, desc_b)['rougeLsum'].fmeasure
        rows, cols = linear_sum_assignment(scores, maximize=True)
        total_score = sum(scores[i][j] for i, j in zip(rows, cols)) / len(rows)
    else:
        rows, cols = [], []
        total_score = 0
    # build seq representations
    repr_a = list(range(len(varg_seq_a)))
    repr_b = list(range(len(varg_seq_a), len(varg_seq_a)+len(varg_seq_b)))
    for i, j in zip(rows, cols):
        if scores[i][j] > 0:
            repr_b[j] = repr_a[i]
    return repr_a, repr_b, total_score

output_path = sys.argv[1]
model_name = sys.argv[2]

import pandas as pd

golds = pd.read_csv("./dataset/caters/caters.csv", sep='\t')['target'].values.tolist()

with open(output_path+"/caters-test-"+model_name+".out", "r") as f:
    predictions = [d.strip() for d in f.readlines()]

assert len(golds) == len(predictions)

print(golds[:3])
print(predictions[:3])
    
pairwise_acc = metric_map['chain_pairwise_accuracy']()
desc_rouge = Average()
    
for prediction_str, gold_str in zip(predictions, golds):
    prediction_varg_seq = V_ARGS_string_to_varg_seq(prediction_str)
    gold_varg_seq = V_ARGS_string_to_varg_seq(gold_str)

    if gold_varg_seq is not None:
        pred_seq, gold_seq, matching_score = get_event_matching(prediction_varg_seq, gold_varg_seq)
        pairwise_acc(pred_seq, gold_seq)
        desc_rouge(matching_score)

result_dict={}
print("PAcc=",pairwise_acc.get_metric())
print("Desc_ROUGE=",desc_rouge.get_metric())

result_dict['pairwise_acc'] = pairwise_acc.get_metric()
result_dict['desc_rouge'] = desc_rouge.get_metric()

with open(output_path+'/caters-'+model_name+'.json', 'w+') as f:
    json.dump(result_dict, f)