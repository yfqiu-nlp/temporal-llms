import json
import sys
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor

import argparse

from allennlp.common.checks import check_for_gpu
from allennlp.common.util import import_module_and_submodules
from allennlp.models.archival import load_archive


def predict_on_unseen_events(data, ans, predictor, print_input_repr=True, file=sys.stdout):
    output = predictor.predict_json(data)
    if print_input_repr:
        print("input_repr:", file=file)
        for r in output['input_repr']:
            print(r, file=file)
        print(file=file)
    print('---'*3, file=file)
    print("candidate_repr:", file=file)
    for r in output['unseen_repr']:
        print(r, file=file)
    print('---'*3, file=file)
    print("Max: {:.4f} - Min: {:.4f} - Mean: {:.4f} - Std: {:.4f} - Best POS: {:.4f}".format(np.max(output['all_beam_scores']), np.min(output['all_beam_scores']), np.mean(output['all_beam_scores']), np.std(output['all_beam_scores']), output["best_pos_score"]), file=file)
    for b_idx, pred in enumerate(output['beam_pred']):
        print("Beam {:d} (gold: {} - score: {:.4f} - pred: {})".format(b_idx, ans, pred['score'], 'no' if pred['pred_is_neg_chain'] else 'yes'), file=file)
        for r in pred['pred_repr']:
            print(r, file=file)
        print(file=file)
    return output['beam_pred'][0]['score'], 'no' if output['beam_pred'][0]['pred_is_neg_chain'] else 'yes', output["best_pos_score"]


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
    parser.add_argument('--feed-unseen', action='store_true', help='whether to feed unseen events as inputs', default=False)
    parser.add_argument('--perm_normalize', action='store_true', help='whether normalize scores by permutations', default=False)
    parser.add_argument('--unigram_normalize', action='store_true', help='whether unigram normalization', default=False)

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

    if args.input_path is None:
        test_json = {
            "events": "<EVENT> died <ARGS> Durer's father died in 1502 <EVENT> died <ARGS> Durer's mother died in 1513",
            "unseen_events": "<EVENT> became <ARGS> Durer's mother became depressed",
            "constraints": "<CONST> <EVENT> died <ARGS> Durer's father died in 1502 <BEFORE> <EVENT> became <ARGS> Durer's mother became depressed",
            'beams': args.beams,
            'feed_unseen': args.feed_unseen,
            'perm_normalize': args.perm_normalize,
            'unigram_normalize': args.unigram_normalize
        }
        '''
        test_json = {
            "events": "<EVENT> estimates <ARGS> The CIA now estimates that it cost al Qaeda about $30 million per year to sustain its activities before 9/11 and that this money was raised almost entirely through donations <EVENT> cost <ARGS> cost al Qaeda about $30 million per year to sustain its activities before 9/11 <EVENT> sustain <ARGS> sustain its activities before 9/11 <EVENT> raised <ARGS> this money raised almost entirely through donations",
            "unseen_events": "",
            "constraints": "",
            'beams': args.beams,
            'feed_unseen': args.feed_unseen,
            'perm_normalize': args.perm_normalize,
            'unigram_normalize': args.unigram_normalize
        }
        '''
        '''
        test_json = {
            'events': '<EVENT> scattered <ARGS> her belongings scattered <EVENT> slipped <ARGS> She slipped on a crack that was on one of the concrete tiles <EVENT> fell <ARGS> She fell on the ground',
            'unseen_events': '',
            'constraints': '',
            'beams': args.beams,
            'feed_unseen': args.feed_unseen,
            'perm_normalize': args.perm_normalize,
            'unigram_normalize': args.unigram_normalize
        }
        '''
        '''
        test_json = {
            'events': '<EVENT> suffered <ARGS> Therefore Louis suffered from cedar allergies for three long months <EVENT> move <ARGS> Louis move to a different state <EVENT> moved <ARGS> Louis just moved to Texas',
            'unseen_events': '',
            'constraints': '',
            'beams': args.beams,
            'feed_unseen': args.feed_unseen,
            'perm_normalize': args.perm_normalize,
            'unigram_normalize': args.unigram_normalize
        }
        '''
        '''
        test_json = {
            'events': '<EVENT> used <ARGS> Once it was cut down we used a cart to bring it to the car <EVENT> bring <ARGS> we bring it to the car <EVENT> cut <ARGS> it cut down',
            'unseen_events': '<EVENT> took <ARGS> Then we took the tree home in our trunk',
            'constraints': '<CONST> <EVENT> bring <ARGS> we bring it to the car <BEFORE> <EVENT> took <ARGS> Then we took the tree home in our trunk',
            'beams': args.beams,
            'feed_unseen': args.feed_unseen,
            'perm_normalize': args.perm_normalize,
            'unigram_normalize': args.unigram_normalize
        }
        test_json = {
            "events": "<EVENT> emerged <ARGS> Islam later emerged as the majority religion during the centuries of Ottoman rule though a significant Christian minority remained <EVENT> remained <ARGS> a significant Christian minority remained",
            "unseen_events": "<EVENT> was <ARGS> christianity was the majority religion",
            "constraints": "<CONST> <EVENT> was <ARGS> christianity was the majority religion <BEFORE> <EVENT> emerged <ARGS> Islam later emerged as the majority religion during the centuries of Ottoman rule though a significant Christian minority remained",
            "beams": args.beams,
            'feed_unseen': args.feed_unseen,
            'perm_normalize': args.perm_normalize,
            'unigram_normalize': args.unigram_normalize
        }
        '''
        '''
        test_json = {
            "events": "<EVENT> emerged <ARGS> Islam later emerged as the majority religion during the centuries of Ottoman rule though a significant Christian minority remained <EVENT> remained <ARGS> a significant Christian minority remained",
            "unseen_events": "<EVENT> ended <ARGS> white-minority rule ended",
            "constraints": "<CONST> <EVENT> ended <ARGS> white-minority rule ended <BEFORE> <EVENT> emerged <ARGS> Islam later emerged as the majority religion during the centuries of Ottoman rule though a significant Christian minority remained",
            "beams": args.beams,
            'feed_unseen': args.feed_unseen,
            'perm_normalize': args.perm_normalize,
            'unigram_normalize': args.unigram_normalize
        }
        test_json = {
            "events": "<EVENT> emerged <ARGS> Islam later emerged as the majority religion during the centuries of Ottoman rule though a significant Christian minority remained <EVENT> remained <ARGS> a significant Christian minority remained",
            "unseen_events": "<EVENT> emerged <ARGS> he emerged as the heir apparent",
            "constraints": "<CONST> <EVENT> emerged <ARGS> he emerged as the heir apparent <BEFORE> <EVENT> emerged <ARGS> Islam later emerged as the majority religion during the centuries of Ottoman rule though a significant Christian minority remained",
            "beams": args.beams,
            'feed_unseen': args.feed_unseen,
            'perm_normalize': args.perm_normalize,
            'unigram_normalize': args.unigram_normalize
        }
        test_json = {
            "events": "<EVENT> estimates <ARGS> The CIA now estimates that it cost al Qaeda about $30 million per year to sustain its activities before 9/11 and that this money was raised almost entirely through donations <EVENT> cost <ARGS> cost al Qaeda about $30 million per year to sustain its activities before 9/11 <EVENT> sustain <ARGS> sustain its activities before 9/11 <EVENT> raised <ARGS> this money raised almost entirely through donations",
            "unseen_events": "<EVENT> given <ARGS> they given a serious blow",
            "constraints": "<CONST> <EVENT> cost <ARGS> cost al Qaeda about $30 million per year to sustain its activities before 9/11 <BEFORE> <EVENT> given <ARGS> they given a serious blow",
            "beams": args.beams,
            'feed_unseen': args.feed_unseen,
            'perm_normalize': args.perm_normalize,
            'unigram_normalize': args.unigram_normalize
        }
        test_json = {
            "events": "<EVENT> tried <ARGS> a couple that the family dog, Mika tried to take Joey from Marsha and eat him <EVENT> take <ARGS> the family dog, Mika take Joey from Marsha <EVENT> eat <ARGS> the family dog, Mika eat him",
            "unseen_events": "<EVENT> gets <ARGS> Mika gets put outside <EVENT> put <ARGS> Mika put outside",
            "constraints": "<CONST> <EVENT> eat <ARGS> the family dog, Mika eat him <BEFORE> <EVENT> gets <ARGS> Mika gets put outside <CONST> <EVENT> eat <ARGS> the family dog, Mika eat him <BEFORE> <EVENT> put <ARGS> Mika put outside",
            "beams": args.beams,
            'feed_unseen': args.feed_unseen,
            'perm_normalize': args.perm_normalize,
            'unigram_normalize': args.unigram_normalize
        }
        test_json = {
            "events": "<EVENT> tried <ARGS> a couple that the family dog, Mika tried to take Joey from Marsha and eat him <EVENT> take <ARGS> the family dog, Mika take Joey from Marsha <EVENT> eat <ARGS> the family dog, Mika eat him",
            "unseen_events": "<EVENT> praised <ARGS> Mika praised by marsha",
            "constraints": "<CONST> <EVENT> eat <ARGS> the family dog, Mika eat him <BEFORE> <EVENT> praised <ARGS> Mika praised by marsha",
            "beams": args.beams,
            'feed_unseen': args.feed_unseen,
            'perm_normalize': args.perm_normalize,
            'unigram_normalize': args.unigram_normalize
        }
        '''
        predict_on_unseen_events(test_json, "UNK", predictor)
    else:
        data = []
        for path_regex in args.input_path:
            for path in sorted(glob.glob(path_regex)):
                with open(path) as f:
                    data += json.load(f)

        pos_scores = []
        neg_scores = []

        #num_p_n_pairs = 0
        #num_corr_p_n_orders = 0
        total_rank_acc = 0.
        total_examples = 0

        result_map = {}
        prediction_count_map = {}
        prediction_map = {}
        gold_count_map = {}

        for d_idx, d in enumerate(tqdm(data)):
            answers = d['answers']
            candidates = d['candidates']
            print("##context##")
            print(d['context'])
            print()
            print("##question##")
            print(d['question'])
            print()
            print("##candidates##")
            for cand, ans in zip(candidates, answers):
                print(cand, ans)
            print()
            print("##predictions##")
            cur_pos_scores = []
            cur_neg_scores = []
            result_map[d_idx] = []
            prediction_count_map[d_idx] = 0.0
            gold_count_map[d_idx] = 0.0
            prediction_map[d_idx] = []
            for cand_idx, (ins, ans) in enumerate(zip(d['lm_instances'], answers)):
                test_json = {
                    "events": ins['events'],
                    "unseen_events": ins['unseen_events'],
                    "constraints": ins['constraints'],
                    "beams": args.beams,
                    'feed_unseen': args.feed_unseen,
                    'perm_normalize': args.perm_normalize,
                    'unigram_normalize': args.unigram_normalize
                }
                score, pred_ans, best_pos_score = predict_on_unseen_events(test_json, ans, predictor, print_input_repr=cand_idx==0)
                if ans == 'yes':
                    #cur_pos_scores.append(score)
                    cur_pos_scores.append(best_pos_score)
                elif ans == 'no':
                    #cur_neg_scores.append(score)
                    cur_neg_scores.append(best_pos_score)
                else:
                    raise ValueError("answer is neither 'yes' nor 'no'")
                assert pred_ans in {'yes', 'no'}
                prediction_map[d_idx].append(pred_ans)
                if pred_ans == "yes":
                    prediction_count_map[d_idx] += 1.0
                if ans == "yes":
                    gold_count_map[d_idx] += 1.0
                result_map[d_idx].append(pred_ans == ans)
            print("\n")
            pos_scores += cur_pos_scores
            neg_scores += cur_neg_scores
            '''
            for p_score in cur_pos_scores:
                for n_score in cur_neg_scores:
                    num_p_n_pairs += 1
                    if p_score > n_score:
                        num_corr_p_n_orders += 1
            '''
            num_p_n_pairs = 0
            num_corr_p_n_orders = 0
            for p_score in cur_pos_scores:
                for n_score in cur_neg_scores:
                    num_p_n_pairs += 1
                    if p_score > n_score:
                        num_corr_p_n_orders += 1
            if num_p_n_pairs > 0:
                total_rank_acc += num_corr_p_n_orders / num_p_n_pairs
                total_examples += 1
        #print("ranking accuracy: {:.4f} ({:d} / {:d})".format(num_corr_p_n_orders / num_p_n_pairs, num_corr_p_n_orders, num_p_n_pairs))
        print("Avg Ranking Accuracy: {:.4f} ({:.4f} / {:d})".format(total_rank_acc / total_examples, total_rank_acc, total_examples))
        print()
        print("###Positive Scores Stat###")
        print(pd.Series(pos_scores).describe())
        print()
        print("###Negative Scores Stat###")
        print(pd.Series(neg_scores).describe())

        total = 0.0
        correct = 0.0
        f1 = 0.0
        binary_total = 0.0
        binary_correct = 0.0
        f1s = []
        for d_idx in result_map:
            val = True
            total += 1.0
            cur_correct = 0.0
            for i, v in enumerate(result_map[d_idx]):
                val = val and v
                if v and prediction_map[d_idx][i] == "yes":
                    cur_correct += 1.0
            if val:
                correct += 1.0
            p = 1.0
            if prediction_count_map[d_idx] > 0.0:
                p = cur_correct / prediction_count_map[d_idx]
            r = 1.0
            if gold_count_map[d_idx] > 0.0:
                r = cur_correct / gold_count_map[d_idx]
            if p + r > 0.0:
                f1 += 2 * p * r / (p + r)

            if p + r > 0.0:
                f1s.append(2 * p * r / (p + r))
            else:
                f1s.append(0.)
            binary_total += len(result_map[d_idx])
            binary_correct += sum(1. if v else 0. for v in result_map[d_idx])

        print("Binary Acc.: {:.4f} ({:.4f} / {:.1f})".format(binary_correct / binary_total, binary_correct, binary_total))
        print("Strict Acc.: {:.4f} ({:.4f} / {:.1f})".format(correct / total, correct, total))
        print("Avg F1: {:.4f} ({:.4f} / {:.1f})".format(f1 / total, f1, total))
