import re
import argparse
import json, pickle, glob
import sys
import os
from collections import defaultdict, deque, Counter
from itertools import permutations
from tqdm import tqdm
import random
import math

from denoising_event_lm.utils.utils import read_data
from denoising_event_lm.data.data_utils.event_lm.creat_lm_data import read_chains, get_neg_seq, print_stat, get_textorder_label, print_stat_with_posneg, print_stat_chainlen
from denoising_event_lm.data.data_utils.event_lm.creat_lm_data import create_seqs_data_from_matched_chains
from denoising_event_lm.data.data_utils.event_lm.creat_lm_data import create_seqs_data_from_KB_chains


def get_textorder_perm(varg_seq):
    perm_ids = sorted(list(range(len(varg_seq))), key=lambda x: varg_seq[x]['tok_start_in_doc'])
    perm = [varg_seq[i] for i in perm_ids]
    assert get_textorder_label(perm) == 'POS'
    return perm, perm_ids


def main(matched_chains_path, KB_events_path, KB_chains_path, output_path, args):
    all_KB_events, all_KB_chains = read_chains(KB_events_path, KB_chains_path, args)
    if matched_chains_path is not None:
        # make KB chains accessible by doc_id
        all_KB_chains = {d['doc_id']: d for d in all_KB_chains}
        # get matched chains and their v-arg toks
        all_matched_chains, _ = read_data(matched_chains_path, args)
        docid2chains = create_seqs_data_from_matched_chains(all_matched_chains, all_KB_events, all_KB_chains, args)
    else:
        # get KB chains and their v-arg toks
        docid2chains = create_seqs_data_from_KB_chains(all_KB_events, all_KB_chains)
    # flatten chain into a list
    all_seqs = [ch for doc_id in docid2chains for ch in docid2chains[doc_id]]
    if args.text_order is not None:
        # get text order data
        all_seqs = [seq for seq in all_seqs if get_textorder_label(seq[0]) == args.text_order]
    if args.num_chain is not None:
        # sample finetune data
        random.seed(args.seed)
        all_seqs = get_finetune_data(all_seqs, args.num_chain)
    # print some stat
    print_stat(all_seqs)
    if args.num_pairwise is not None:
        # creat pairwise chain
        random.seed(args.seed)
        all_seqs = get_pairwise_chain(all_seqs)
        # sample pairwise chain
        all_seqs = get_finetune_data(all_seqs, args.num_pairwise)
        print("Stat of pairwise chains")
        print_stat(all_seqs)
    # creat negative samples
    random.seed(args.seed)
    data = []
    for seq in tqdm(all_seqs):
        text_order = get_textorder_label(seq[0])
        if text_order == 'POS':
            data.append({'varg_seq': seq[0],
                         'label': 'POS',
                         'permutation': list(range(len(seq[0]))),
                         '_id': seq[1]+'_pos'})
        else:
            # get a scrambled chain that is in-text-order
            perm, perm_ids = get_textorder_perm(seq[0])
            data.append({'varg_seq': perm,
                         'label': 'NEG',
                         'permutation': perm_ids,
                         '_id': seq[1]+'_neg'})
    # print data stat with pos-neg instances info
    print_stat_with_posneg(data)
    print_stat_chainlen(data)
    if args.action == 'json':
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)
    elif args.action == 'pickle':
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
    return data


if __name__ == '__main__':
    """
    each true chain -> add the permutation that is in text-order
    """
    parser = argparse.ArgumentParser("KB chain extraction")
    parser.add_argument('--matched_chains', required=False, help='matched chains path')
    parser.add_argument('--seen_matched_chains', required=False, help='seen matched chains path')
    parser.add_argument('--KB_events', required=True, help='KB events path')
    parser.add_argument('--KB_chains', required=True, help='KB chains path')
    parser.add_argument('--output', help='output')
    parser.add_argument('--action', default='pickle', help='action to do', choices=['txt', 'json', 'pickle'])
    parser.add_argument('--start', type=int, help='start idx of data to be processed', default=-1)
    parser.add_argument('--end', type=int, help='end idx of data to be processed', default=-1)
    parser.add_argument('--num_match', type=int, help='number of matched chains to use for each target chain', default=5)
    parser.add_argument('--num_chain', type=str, help='number of true chain to sample', default=None)
    parser.add_argument('--num_pairwise', type=str, default=None,
                        help='number of pairwise chains to sample from all chains')
    parser.add_argument('--text_order', help='get in- or out-of- text order chains', 
                        choices=['POS', 'NEG'] ,default=None)
    parser.add_argument('--seed', type=int, help='random seed', default=2020)
    args = parser.parse_args()

    data = main(args.matched_chains, args.KB_events, args.KB_chains, args.output, args)
