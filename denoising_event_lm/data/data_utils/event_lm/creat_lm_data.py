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


def get_finetune_data(all_seqs, num_finetune):
    assert type(num_finetune) == str, type(num_finetune)
    if not num_finetune.isdigit():
        num_finetune = int(len(all_seqs) * num_finetune)
    else:
        num_finetune = int(num_finetune)
    assert num_finetune > 0, num_finetune
    if num_finetune >= len(all_seqs):
        num_finetune = len(all_seqs)
    print("Sample {:d} data from {:d} data".format(num_finetune, len(all_seqs)))
    if num_finetune >= len(all_seqs):
        return all_seqs
    indices = list(range(len(all_seqs)))
    random.shuffle(indices)
    indices = indices[:num_finetune]
    all_seqs = [all_seqs[i] for i in indices]
    assert len(all_seqs) == num_finetune
    return all_seqs


def get_pairwise_chain(all_seqs):
    pairwise_all_seqs = []
    for varg_seq, seq_id in all_seqs:
        pair_id = 0
        for i in range(len(varg_seq)):
            for j in range(i+1, len(varg_seq)):
                pairwise_all_seqs.append(([varg_seq[i], varg_seq[j]], seq_id+'_pair'+str(pair_id), varg_seq))
                pair_id += 1
    return pairwise_all_seqs


def get_toks_from_span(toks, span):
    s, e = span
    return [toks[p] for p in range(s, e+1)]


def flatten_vargtoks(vargtoks):
    toks = vargtoks['V'][:]
    for arg in vargtoks['ARGS']:
        toks += arg
    return toks


def get_events_vargtoks(d, eiid2events, eiid2srlvid):
    toks = d['tokens']
    sents_tok_offset = d['sents_tok_offset'] + [len(toks)]
    sents_toks = [toks[sents_tok_offset[i]:sents_tok_offset[i+1]] for i in range(len(sents_tok_offset)-1)]
    sentid2srl = d['sentid2srl']
    eiid2vargtoks = {}
    for eiid in eiid2events:
        # get event info
        e_obj = eiid2events[eiid]
        event = e_obj['event']
        sent_id = e_obj['sent_id']
        sent_toks = sents_toks[sent_id]
        tok_span = [p - sents_tok_offset[sent_id] for p in e_obj['tok_span']]
        tok_start = e_obj['tok_start'] - sents_tok_offset[sent_id]
        tok_end = e_obj['tok_end'] - sents_tok_offset[sent_id]
        event_toks = [sent_toks[p] for p in tok_span]
        assert " ".join(event_toks) == event or event[-3:] == '...' # in UDS-T event too long will get cut
        sent_srl = sentid2srl[str(sent_id)]
        assert sent_srl['words'] == sent_toks
        matched_v_args = sent_srl['v_args'][eiid2srlvid[eiid]] if eiid2srlvid.get(eiid, None) is not None else None # some event like ``say`` may be removed
        if matched_v_args is not None:
            vargtoks = {'V_toks': get_toks_from_span(sent_toks, matched_v_args['V']),
                        'ARGS_toks': [get_toks_from_span(sent_toks, sp) for sp in matched_v_args['ARG_span']],
                        'V_span': matched_v_args['V'],
                        'ARGS_span': matched_v_args['ARG_span'],
                        'ARGS_type': matched_v_args['ARG_type'],
                        'tok_start_in_doc': matched_v_args['V'][0] + sents_tok_offset[sent_id],
                        'sent_toks': sent_toks,
                        'sent_tok_start_in_doc': sents_tok_offset[sent_id],
                        'sent_id': sent_id}
        else:
            vargtoks = None
        eiid2vargtoks[eiid] = vargtoks
    d['eiid2vargtoks'] = eiid2vargtoks


def read_chains(events_path, chains_path, args):
    # read events
    all_events = []
    for fn in sorted(glob.glob(events_path)):
        events, _ = read_data(fn, args)
        if args is not None and args.src is not None:
            events = [d for d in events if d['src'] in args.src]
        all_events += events
    all_events = {d['doc_id']: d for d in all_events}
    # read chains
    all_chains = []
    for fn in sorted(glob.glob(chains_path)):
        chains, _ = read_data(fn, args)
        if args is not None and args.src is not None:
            chains = [d for d in chains if d['doc_id'] in all_events]
            assert all(all_events[d['doc_id']]['src'] in args.src for d in chains)
        all_chains += chains
    # get events' varg toks
    for chains in all_chains:
        doc_id = chains['doc_id']
        eiid2srlvid = chains['eiid2srlvid']
        get_events_vargtoks(all_events[doc_id], all_events[doc_id]['eiid2events'], eiid2srlvid)
    return all_events, all_chains


def perm_generator(seq, num, seen=set()):
    length = len(seq)
    max_seen = math.factorial(length)
    res = []
    while len(seen) < max_seen and len(res) < num:
        perm = tuple(random.sample(seq, length))
        if perm not in seen:
            seen.add(perm)
            res.append(perm)
    return res


def get_neg_seq(seq, num_neg):
    indices = tuple(range(len(seq)))
    if len(seq) < 6:
        perm = [p for p in permutations(indices, len(seq)) if not p == indices]
        random.shuffle(perm)
    else:
        perm = perm_generator(indices, num_neg, seen=set([indices]))
        assert len(set(perm)) == len(perm) and len(perm) <= num_neg
    perm_ids = perm[:num_neg]
    perm = [[seq[i] for i in p] for p in perm][:num_neg]
    if math.factorial(len(seq)) >= num_neg + 1:
        assert len(perm) == num_neg, repr(ind_perm) + '\n' + repr(indices)
    elif math.factorial(len(seq)) < num_neg + 1:
        assert len(perm) == math.factorial(len(seq)) - 1, repr(ind_perm) + '\n' + repr(indices)
    return perm, perm_ids


def print_stat(all_seqs, print_func=print):
    num = len(all_seqs)
    tot_len = sum(len(s[0]) for s in all_seqs)
    print_func("Num Seq:", num)
    print_func("Avg eq Length:", tot_len / num)


def print_stat_with_posneg(data, print_func=print):
    print_func("Num Seq: {:d}".format(len(data)))
    print_func("Avg Seq Length: {:5.3f}".format(sum(len(d['varg_seq']) for d in data) / len(data)))
    print_func("Num POS Seq: {:d}".format(len([d for d in data if d['label'] == 'POS'])))
    if len([d for d in data if d['label'] == 'POS']) > 0:
        print_func("Avg POS Seq Length: {:5.3f}".format(sum(len(d['varg_seq']) for d in data if d['label'] == 'POS') / len([d for d in data if d['label'] == 'POS'])))
    else:
        print_func("Avg POS Seq Length: 0.")
    print_func("Num NEG Seq: {:d}".format(len([d for d in data if d['label'] == 'NEG'])))
    if len([d for d in data if d['label'] == 'NEG']) > 0:
        print_func("Avg NEG Seq Length: {:5.3f}".format(sum(len(d['varg_seq']) for d in data if d['label'] == 'NEG') / len([d for d in data if d['label'] == 'NEG'])))
    else:
        print_func("Avg NEG Seq Length: 0.")


def print_stat_chainlen(data, print_func=print):
    ls = [len(d['varg_seq']) for d in data]
    for i in range(max(ls)+1):
        print_func("length {:d}: {:5.3f}%".format(i, (sum([l == i for l in ls]) / len(ls)) * 100))


def get_topk_matches(matches, num_match, all_KB_chains, len2weight, seen):
    topk = []
    for m in matches[:num_match]:
        if m in seen:
            continue
        topk.append(m)
    if len2weight is None:
        return topk
    # compute len2cnts
    len2cnts = defaultdict(int)
    for m in topk:
        l = len(all_KB_chains[m[0]]['chains'][m[1]]['eiids'])
        len2cnts[l] += 1
    # compute len2clipcnts
    len2clipcnts = {l: int(len2weight[l] * len2cnts[l] + 0.5) for l in len2weight if len2weight[l] < 1}
    # decrease len ratio
    clip_topk = []
    for m in topk:
        l = len(all_KB_chains[m[0]]['chains'][m[1]]['eiids'])
        if not l in len2clipcnts:
            clip_topk.append(m)
        elif len2clipcnts[l] > 0:
            len2clipcnts[l] -= 1
            clip_topk.append(m)
    topk = clip_topk
    # compute len2addcnts
    len2addcnts = {l: int((len2weight[l] - 1) * len2cnts[l] + 0.5) for l in len2weight if len2weight[l] > 1}
    # increase len ratio
    for m in matches[num_match:]:
        if all(addcnt == 0 for addcnt in list(len2addcnts.values())):
            break
        if m in seen:
            continue
        l = len(all_KB_chains[m[0]]['chains'][m[1]]['eiids'])
        if l in len2addcnts and len2addcnts[l] > 0:
            topk.append(m)
            len2addcnts[l] -= 1
    return topk


def create_seqs_data_from_matched_chains(all_matched_chains, all_KB_events, all_KB_chains, args):
    args.len2weight = None if args.len2weight is None else {int(l): float(w) for l, w in json.loads(args.len2weight).items()}
    # get matched chains and their v-arg toks, group them by KB doc_id
    seen = set()
    # if seen_matched_chains in args, load seen matched chains
    if args.seen_matched_chains is not None:
        seen_matched_chains, _ = read_data(args.seen_matched_chains, args)
        for doc_target_chains_matches in tqdm(seen_matched_chains):
            for target_matches in doc_target_chains_matches['matches']:
                '''
                for m in target_matches[:args.num_match]:
                    if m in seen:
                        continue
                    seen.add(m)
                '''
                topk = get_topk_matches(target_matches, args.num_match, all_KB_chains, args.len2weight, seen)
                for m in topk:
                    assert not m in seen
                    seen.add(m)
    docid2chains = defaultdict(list)
    for doc_target_chains_matches in tqdm(all_matched_chains):
        for target_matches in doc_target_chains_matches['matches']:
            '''
            for m in target_matches[:args.num_match]:
                if m in seen:
                    continue
                seen.add(m)
                chain_id = str(m[0])+'_#'+str(m[1])
                eiid2vargtoks = all_KB_events[m[0]]['eiid2vargtoks']
                chain = all_KB_chains[m[0]]['chains'][m[1]]['eiids']
                vargtoks_seq = [eiid2vargtoks[eiid] for eiid in chain]
                docid2chains[m[0]].append((vargtoks_seq, chain_id))
            '''
            topk = get_topk_matches(target_matches, args.num_match, all_KB_chains, args.len2weight, seen)
            for m in topk:
                assert not m in seen
                seen.add(m)
                chain_id = str(m[0])+'_#'+str(m[1])
                eiid2vargtoks = all_KB_events[m[0]]['eiid2vargtoks']
                chain = all_KB_chains[m[0]]['chains'][m[1]]['eiids']
                vargtoks_seq = [eiid2vargtoks[eiid] for eiid in chain]
                docid2chains[m[0]].append((vargtoks_seq, chain_id))
    return docid2chains


def create_seqs_data_from_KB_chains(all_KB_events, all_KB_chains):
    docid2chains = defaultdict(list)
    for doc_KB_chains in tqdm(all_KB_chains):
        doc_id = doc_KB_chains['doc_id']
        if len(doc_KB_chains['chains']) > 0:
            eiid2vargtoks = all_KB_events[doc_id]['eiid2vargtoks']
            for chain_idx, chain in enumerate(doc_KB_chains['chains']):
                chain_id = str(doc_id)+'_#'+str(chain_idx)
                vargtoks_seq = [eiid2vargtoks[eiid] for eiid in chain['eiids']]
                docid2chains[doc_id].append((vargtoks_seq, chain_id))
    return docid2chains


def get_textorder_label(varg_seq):
    tok_starts = [varg['tok_start_in_doc'] for varg in varg_seq]
    diffs = [tok_starts[i+1]-tok_starts[i] for i in range(len(tok_starts)) if i < len(tok_starts)-1]
    if all([d >= 0 for d in diffs]):
        return 'POS'
    else:
        return 'NEG'


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
        if args.label_type == 'text':
            label = get_textorder_label(seq[0])
        elif args.label_type == 'event':
            label = 'POS'
        data.append({'varg_seq': seq[0],
                     'label': label,
                     'permutation': list(range(len(seq[0]))),
                     '_id': seq[1]+'_'+label.lower()})
        perm, perm_ids = get_neg_seq(seq[0], args.num_neg)
        for p, p_ids in zip(perm, perm_ids):
            if args.label_type == 'text':
                label = get_textorder_label(p)
            elif args.label_type == 'event':
                label = 'NEG'
            data.append({'varg_seq': p,
                         'label': label,
                         'permutation': p_ids,
                         '_id': seq[1]+'_'+label.lower()})
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
    each true chain -> add true chain, ``args.num_neg`` shuffled one into training set
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
    parser.add_argument('--num_neg', type=int, help='number of negative samples per seq', default=5)
    parser.add_argument('--num_match', type=int, help='number of matched chains to use for each target chain', default=5)
    parser.add_argument('--num_chain', type=str, help='number of true chain to sample', default=None)
    parser.add_argument('--num_pairwise', type=str, default=None,
                        help='number of pairwise chains to sample from all chains')
    parser.add_argument('--label_type', help='use event order or textual order to set the label', 
                        choices=['text', 'event'] ,default='event')
    parser.add_argument('--text_order', help='get in- or out-of- text order chains', 
                        choices=['POS', 'NEG'] ,default=None)
    parser.add_argument('--len2weight', type=str, help='upweight ratio for each length', default=None)
    parser.add_argument('--seed', type=int, help='random seed', default=2020)
    args = parser.parse_args()

    data = main(args.matched_chains, args.KB_events, args.KB_chains, args.output, args)
