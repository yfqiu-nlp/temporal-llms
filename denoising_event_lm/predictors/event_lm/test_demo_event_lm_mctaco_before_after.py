import json
import pickle
import sys
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor
from copy import deepcopy
import torch
from torch import nn
import heapq

import argparse

import allennlp
from allennlp.common.checks import check_for_gpu
if allennlp.__version__ == '0.8.5':
    from allennlp.common.util import import_submodules as import_module_and_submodules
elif allennlp.__version__ == '1.1.0':
    from allennlp.common.util import import_module_and_submodules
from allennlp.models.archival import load_archive


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


def chain_str(chain):
    texts = []
    for varg in chain:
        if not 'Description' in varg:
            varg['Description'] = " ".join(get_flatten_varg_toks(varg))
        texts.append("<EVENT> " + " ".join(varg['V_toks']) + " <ARGS> " + varg['Description'])
    return texts


def check_chain_fulfill_constraints(events, constraints):
    def fulfill_constraint(e1, e2):
        for e in events:
            if e == e1:
                return True
            elif e == e2:
                return False
    return all(fulfill_constraint(e1, e2) for e1, e2 in constraints)


def predict_on_unseen_events(data, predictor, args, file=sys.stdout):
    question_event_in_context = data['question_event_in_context']
    question_event_in_context_idx = data['question_event_in_context_idx']
    assert data['context_events'][question_event_in_context_idx] == question_event_in_context
    assert data['temp_rel'] in {'BEFORE', 'AFTER'}
    if data['temp_rel'] == 'BEFORE':
        constraints = [(data['candidate_event'], question_event_in_context)]
    elif data['temp_rel'] == 'AFTER':
        constraints = [(question_event_in_context, data['candidate_event'])]

    test_json = {
        'events': data['context_events'],
        'cand_event': data['candidate_event'],
        'beams': args.beams,
        'feed_unseen': args.feed_unseen
    }

    output = predictor.predict_json(test_json)
    print('---'*3, file=file)
    print('##Context##', file=file)
    print(data['context'], file=file)
    print(file=file)
    print('##Question##', file=file)
    print(data['question'], file=file)
    print(file=file)
    print('##Candidate##', file=file)
    print(data['candidate'], file=file)
    print(file=file)
    print("##Relation##", file=file)
    print("[Candidate]", data['temp_rel'], "[Question]", file=file)
    print(file=file)
    print('---'*3, file=file)
    print("input_repr:", file=file)
    for r in chain_str(output['input_vargs']):
        print(r, file=file)
    print(file=file)
    print('---'*3, file=file)
    print("question_repr:", file=file)
    for r in chain_str([question_event_in_context]):
        print(r, file=file)
    print(file=file)
    print('---'*3, file=file)
    print("cand_repr:", file=file)
    for r in chain_str(output['unseen_vargs']):
        print(r, file=file)
    print(file=file)
    print('---'*3, file=file)
    print("Max: {:.4f} - Min: {:.4f} - Mean: {:.4f} - Std: {:.4f} - Best POS: {:.4f}".format(np.max(output['all_beam_scores']), np.min(output['all_beam_scores']), np.mean(output['all_beam_scores']), np.std(output['all_beam_scores']), output["best_pos_score"]), file=file)
    beam_matches = []
    for b_idx, pred in enumerate(output['beam_pred']):
        if "EVENT_SEP" in pred['pred_vargs'][0]:
            for v in pred['pred_vargs']:
                v.pop("EVENT_SEP")
        assert question_event_in_context in pred['pred_vargs']
        assert data['candidate_event'] in pred['pred_vargs']
        match = check_chain_fulfill_constraints(pred['pred_vargs'], constraints)
        beam_matches.append(match)
        print("Beam {:d} (gold: {} - score: {:.4f})".format(b_idx, match, pred['score']), file=file)
        for r in chain_str(pred['pred_vargs']):
            print(r, file=file)
        print(file=file)
    print("\n\n", file=file)
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
    parser.add_argument('--feed-unseen', action='store_true', help='whether to feed unseen events as inputs', default=False)

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

    data = []
    for path_regex in args.input_path:
        for path in sorted(glob.glob(path_regex)):
            with open(path, 'r') as f:
                data += json.load(f)
    if args.num_instances > 0:
        data = data[:args.num_instances]
    print("Num Instances:", len(data))

    total_confusion = {
        "gold BEFORE": {
            "pred BEFORE": 0.,
            "pred AFTER": 0.
        },
        "gold AFTER": {
            "pred BEFORE": 0.,
            "pred AFTER": 0.
        }
    }
    total_correct = 0.
    total_examples = 0

    for d_idx, d in enumerate(tqdm(data)):
        beam_matches = predict_on_unseen_events(d, predictor, args)
        if beam_matches[0]:
            pred_temp_rel = d['temp_rel']
        else:
            if d['temp_rel'] == 'BEFORE':
                pred_temp_rel = 'AFTER'
            else:
                pred_temp_rel = 'BEFORE'
        total_confusion['gold '+d['temp_rel']]['pred '+pred_temp_rel] += 1
        total_correct += int(beam_matches[0])
        total_examples += 1
    assert sum(pv for gk, gv in total_confusion.items() for pk, pv in gv.items()) == total_examples
    assert sum(pv for gk, gv in total_confusion.items() for pk, pv in gv.items() if gk[5:] == pk[5:]) == total_correct
    print("Acc: {:.4f} ({:.4f} / {:d})".format(total_correct / total_examples, total_correct, total_examples))
    # BEFORE f1
    if sum(pv for pk, pv in total_confusion['gold BEFORE'].items()) > 0:
        recl = total_confusion['gold BEFORE']['pred BEFORE'] / sum(pv for pk, pv in total_confusion['gold BEFORE'].items())
    else:
        recl = 0.
    if sum(gv['pred BEFORE'] for gk, gv in total_confusion.items()) > 0:
        prec = total_confusion['gold BEFORE']['pred BEFORE'] / sum(gv['pred BEFORE'] for gk, gv in total_confusion.items())
    else:
        prec = 0.
    if prec + recl > 0:
        before_f1 = (2 * prec * recl) / (prec + recl)
    else:
        before_f1 = 0.
    print("BEFORE P: {:.4f} - R: {:.4f} - F1: {:.4f}".format(prec, recl, before_f1))
    # AFTER f1
    if sum(pv for pk, pv in total_confusion['gold AFTER'].items()) > 0:
        recl = total_confusion['gold AFTER']['pred AFTER'] / sum(pv for pk, pv in total_confusion['gold AFTER'].items())
    else:
        recl = 0.
    if sum(gv['pred AFTER'] for gk, gv in total_confusion.items()) > 0:
        prec = total_confusion['gold AFTER']['pred AFTER'] / sum(gv['pred AFTER'] for gk, gv in total_confusion.items())
    else:
        prec = 0.
    if prec + recl > 0:
        after_f1 = (2 * prec * recl) / (prec + recl)
    else:
        after_f1 = 0.
    print("AFTER  P: {:.4f} - R: {:.4f} - F1: {:.4f}".format(prec, recl, after_f1))
    macro_f1 = (before_f1 + after_f1) / 2.
    print("Macro F1: {:.4f})".format(macro_f1))
