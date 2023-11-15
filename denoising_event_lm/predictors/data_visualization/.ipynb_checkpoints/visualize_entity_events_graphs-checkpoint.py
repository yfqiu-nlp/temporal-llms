import sys
import os
import json
import pickle
import argparse
import glob
import math
import numpy as np
import time
import traceback
from tqdm import tqdm
from collections import defaultdict
from graphviz import Digraph
import bisect

from denoising_event_lm.utils.utils import read_data



def get_all_doc_spans(doc_len, eiid2events, events_edges, unmatchedsrleiids, unmatchedsrl_eiid2events, mentions, tokens):
    e_in_graph = set([eiid for eiid in events_edges.keys()]) | set([eiid for ends in events_edges.values() for eiid in ends])
    # temporal events
    obj_spans = [[e['tok_start'], e['tok_end'], ["in_graph" if eiid in e_in_graph else "not_in_graph", eiid]] 
                 for eiid, e in eiid2events.items()]
    # unmatched srl events
    obj_spans += [[unmatchedsrl_eiid2events[eiid]['tok_start'], unmatchedsrl_eiid2events[eiid]['tok_end'], ["srl", eiid]] for eiid in unmatchedsrleiids]
    # mentions, some mentions may be a predicate so we check here (eg: UDS-T dev #113)
    span2idx = {(s, e): i for i, (s, e, tags) in enumerate(obj_spans)}
    for m in mentions:
        if (m['span'][0], m['span'][1]) in span2idx:
            obj_spans[span2idx[(m['span'][0], m['span'][1])]][2][1] = ", entity"
        else:
            obj_spans.append([m['span'][0], m['span'][1], ["mention", "entity"]])
    obj_spans = sorted(obj_spans)
    #print(json.dumps(obj_spans, indent=4))
    # check non-overlap
    i = 0
    while i < len(obj_spans)-1:
        prev_s, prev_e, prev_tags = obj_spans[i]
        s, e, tags = obj_spans[i+1]
        if not s > prev_e:
            if not (tags[0] == "mention" or prev_tags[0] == "mention"):
                if e >= prev_e + 1: # s1 s2 e1 e2 -> (s1 e1)(e1+1, e2)
                    if i+2 == len(obj_spans) or not [prev_e+1, e] == [obj_spans[i+2][0], obj_spans[i+2][1]]: # prevent [e1+1, e2] already exists
                        obj_spans[i+1][0] = prev_e + 1
                        obj_spans = sorted(obj_spans) # when modify i+1, need to re-sort
                    else:
                        if tags[0] == "in_graph" or (tags[0] == "not_in_graph" and not obj_spans[i+2][2][0] == 'in_graph') or (tags[0] == "srl" and not obj_spans[i+2][2][0] == 'in_graph'):
                            obj_spans[i+2][2] = tags
                        obj_spans = obj_spans[:i+1] + obj_spans[i+2:]
                else:
                    # s1 s2 e2 e1 -> (s1, s2-1)(s2, e2)(e2, e1)
                    obj_spans[i][1] = s - 1
                    if s == prev_s:
                        print(tokens[prev_s:prev_e+1], tokens[s:e+1])
                        print((prev_s, prev_e), (s, e))
                        print(prev_tags, tags)
                    assert not s == prev_s
                    if prev_e > e+1: # prevent s1 s2 e2==e1
                        insert_sp = [e+1, prev_e, prev_tags]
                        insert_pos = bisect.bisect_left([(ele[0], ele[1]) for ele in obj_spans], (e+1, prev_e), lo=i+2) # get insert pos only by (s, e) or the already existed (e2+1, e1) may be at insert_pos-1 instead of insert_pos
                        if insert_pos == len(obj_spans) or not [e+1, prev_e] == [obj_spans[insert_pos][0], obj_spans[insert_pos][1]]: # prevent [e2+1, e1] already exists
                            obj_spans = obj_spans[:insert_pos] + [insert_sp] + obj_spans[insert_pos:]
            else:
                if prev_tags[0] == "mention":
                    if e >= prev_e + 1: # s1 s2 e1 e2 -> (s1 e1)(e1+1, e2)
                        if i+2 == len(obj_spans) or not [prev_e+1, e] == [obj_spans[i+2][0], obj_spans[i+2][1]]: # prevent [e1+1, e2] already exists
                            obj_spans[i+1][0] = prev_e + 1
                            obj_spans = sorted(obj_spans) # when modify i+1, need to re-sort
                        else:
                            if tags[0] == "in_graph" or (tags[0] == "not_in_graph" and not obj_spans[i+2][2][0] == 'in_graph') or (tags[0] == "srl" and not obj_spans[i+2][2][0] == 'in_graph'):
                                obj_spans[i+2][2] = tags
                            obj_spans = obj_spans[:i+1] + obj_spans[i+2:]
                    else:
                        # s1 s2 e2 e1 -> (s1, s2-1)(s2, e2)(e2, e1)
                        obj_spans[i][1] = s - 1
                        if s == prev_s:
                            print(tokens[prev_s:prev_e+1], tokens[s:e+1])
                            print((prev_s, prev_e), (s, e))
                            print(prev_tags, tags)
                        assert not s == prev_s
                        if prev_e >= e+1: # prevent s1 s2 e2==e1
                            insert_sp = [e+1, prev_e, ["mention", "entity"]]
                            insert_pos = bisect.bisect_left([(ele[0], ele[1]) for ele in obj_spans], (e+1, prev_e), lo=i+2) # get insert pos only by (s, e) or the already existed (e2+1, e1) may be at insert_pos-1 instead of insert_pos
                            if insert_pos == len(obj_spans) or not [e+1, prev_e] == [obj_spans[insert_pos][0], obj_spans[insert_pos][1]]: # prevent [e2+1, e1] already exists
                                obj_spans = obj_spans[:insert_pos] + [insert_sp] + obj_spans[insert_pos:]
                elif tags[0] == "mention":
                    if s - 1 >= prev_s: # s1 s2 e1 e2 or s1 s2 e2 e1 -> (s1, s2-1)(s2, e2)
                        obj_spans[i][1] = s - 1
                    else:
                        # s1==s2 e1 e2 -> (s1, e1)(e1+1, e2)
                        if i+2 == len(obj_spans) or not [prev_e+1, e] == [obj_spans[i+2][0], obj_spans[i+2][1]]: # prevent [e1+1, e2] already exists
                            obj_spans[i+1][0] = prev_e + 1
                            obj_spans = sorted(obj_spans) # when modify i+1, need to re-sort
                        else:
                            if tags[0] == "in_graph" or (tags[0] == "not_in_graph" and not obj_spans[i+2][2][0] == 'in_graph') or (tags[0] == "srl" and not obj_spans[i+2][2][0] == 'in_graph'):
                                obj_spans[i+2][2] = tags
                            obj_spans = obj_spans[:i+1] + obj_spans[i+2:]
                        if not e >= prev_e + 1:
                            print(span2idx)
                            print((prev_s, prev_e), (s, e))
                            print(prev_tags, tags)
                            exit()
        i += 1
    # check results
    assert all(obj_spans[i][0] > obj_spans[i-1][1] for i in range(1, len(obj_spans)))
    assert all(e >= s for s, e, tags in obj_spans)
    all_spans = []
    sp2tags = []
    last_end = -1
    for s, e, tags in obj_spans:
        if s > last_end+1:
            all_spans.append((last_end+1, s-1))
            sp2tags.append(["", ""])
        all_spans.append((s, e))
        sp2tags.append(tags)
        last_end = e
    if doc_len > last_end+1:
        all_spans.append((last_end+1, doc_len-1))
        sp2tags.append(["", ""])
    return all_spans, sp2tags


def get_digraph_template(eiid2events, events_edges):
    g = Digraph()
    for start, ends in events_edges.items():
        for end in ends:
            g.edge(("[%s]\n" % start)+eiid2events[start]['event'],
                   ("[%s]\n" % end)+eiid2events[end]['event'])
    return g.source


def get_instance_for_render(d_nlp, d_graphs):
    assert d_nlp['doc_id'] == d_graphs['doc_id']
    doc_text = d_nlp['text']
    doc_toks = d_nlp['tokens']
    sents_tok_offset = d_nlp['sents_tok_offset'] + [len(doc_toks)]
    eiid2srlvid = d_graphs['eiid2srlvid']
    unmatchedsrl_eiid2events = d_graphs['unmatchedsrl_eiid2events']
    clusterid2graph = d_graphs['clusterid2graph']
    clusterid2unmatchedsrleiids = d_graphs['clusterid2unmatchedsrleiids']
    # get coref
    coref_clusters = d_nlp['pred_coref']
    clusterid2mentions = defaultdict(list)
    for cluster in coref_clusters:
        for m in cluster:
            offset = sents_tok_offset[m['sent_id']]
            start, end = m['span']
            m['span'] = [start+offset, end+offset]
            clusterid2mentions[m['cluster_id']].append(m)
    # get render instance for each entity
    entity_objs = []
    for c_id in clusterid2graph:
        eiid2events = clusterid2graph[c_id]['eiid2events']
        events_edges = clusterid2graph[c_id]['events_edges']
        unmatchedsrleiids = clusterid2unmatchedsrleiids.get(c_id, [])
        mentions = clusterid2mentions[int(c_id)]
        all_doc_spans, doc_sp2tags = get_all_doc_spans(len(doc_toks), eiid2events, events_edges, unmatchedsrleiids, unmatchedsrl_eiid2events, mentions, doc_toks)
        graph_template = get_digraph_template(eiid2events, events_edges)
        obj = {"doc_tokens":        doc_toks,
               "all_doc_spans":     all_doc_spans,
               "doc_sp2tags":       doc_sp2tags,
               "graph_template":    graph_template,
               "doc_id":            d_nlp['doc_id'],
               }
        entity_objs.append(obj)
    return entity_objs


def render_token(tok, tags):
    style = ""
    if tags[0] == "in_graph":
        style = "background-color: rgba(0, 0, 255, 0.5); border-radius: 7px; padding-left: 3px; padding-right: 3px; border-style: solid; border-color: rgba(0, 0, 255, 0.6); border-width: 1.5px"
    elif tags[0] == "not_in_graph":
        style = "background-color: rgba(0, 0, 255, 0.2); border-radius: 7px; padding-left: 3px; padding-right: 3px; border-style: dashed; border-color: rgba(0, 0, 255, 0.3); border-width: 1.5px"
    elif tags[0] == "srl":
        style = "background-color: rgba(0, 0, 255, 0.2); border-radius: 7px; padding-left: 3px; padding-right: 3px; border-style: dashed; border-color: rgba(0, 0, 255, 0.3); border-width: 1.5px"
    elif tags[0] == "mention":
        style = "background-color: rgba(0, 179, 179, 0.4); border-radius: 7px; padding-left: 3px; padding-right: 3px;"
    style = repr(style)
    tip = repr(tags[1])
    br_splits = tok.split('<br />\n')
    block = "".join("<span>{:s}</span>".format(br_split if i == len(br_splits)-1 else f"<span>{br_split}</span><br/><br/>") 
                    for i, br_split in enumerate(br_splits))
    return \
    f"""<span><span data-toggle="tooltip" data-placement="auto top" title={tip} style={style}>{block}</span><span> </span></span>"""


def render_doc(entity_obj, c_id, last=False):
    """render documents with each special spans being highlighted, also add divs for graphviz rendering"""
    doc_id = entity_obj['doc_id'].replace('.', '_')
    tokens = entity_obj['doc_tokens']
    spans = entity_obj['all_doc_spans']
    sp2tags = entity_obj['doc_sp2tags']
    doc_block = "".join(render_token(" ".join(tokens[s:e+1]), sp2tags[i]) for i, (s, e) in enumerate(spans))
    hr = """<hr style="height: 1px" />""" if not last else ""
    return f"""
                    <div class="form__field">
                        <div class="doc">
                            <h4>Doc #{doc_id} - Entity #{c_id}</h4>
                            {doc_block}
                        </div>
                        <div id="graph_{doc_id}_{c_id}" style="text-align: center;" class="doc">
                        </div>
                        {hr}
                    </div>
    """

def render_entity_events_graphs(ins):
    """render documents with each special spans being highlighted, also add divs for graphviz rendering"""
    block = "".join(render_doc(entity_obj, c_id, c_id == len(ins)-1) for c_id, entity_obj in enumerate(ins))
    return f"""
                <div>
                    {block}
                    <br/>
                    <br/>
                    <br/>
                    <hr style="height: 2px; border: none; background-color: #b3b3b3;" />
                </div>
    """


def render_graphviz_objects(ins):
    """render graphviz object for each instance, put into the script part"""
    block = "\n".join('d3.select("#graph_{:s}_{:s}").graphviz().zoom(false).renderDot({:s});'.format(obj['doc_id'].replace('.', '_'), str(c_id), repr(obj['graph_template'])) for c_id, obj in enumerate(ins))
    return block 


def render_index_html(html_body, script_body):
    """get index.html"""
    return f"""
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="utf-8">
            <link href="https://fonts.googleapis.com/css?family=Roboto+Mono&display=swap" rel="stylesheet">
            <link href='https://fonts.googleapis.com/css?family=Source+Sans+Pro' rel='stylesheet' type='text/css'>
            <script src="https://d3js.org/d3.v5.min.js"></script>
            <script src="https://unpkg.com/@hpcc-js/wasm@0.3.6/dist/index.min.js"></script>
            <script src="https://unpkg.com/d3-graphviz@3.0.0/build/d3-graphviz.js"></script>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
            <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>
            <style>
                body,
                html {{
                  min-width: 48em;
                  font-size: 16px;
                  width: 100%;
                  height: 100%;
                  margin: 0;
                  padding: 0;
                }}
                * {{
                  font-family: 'Source Sans Pro', sans-serif;
                  color: #232323;
                }}
                .model__content {{
                    padding: 0.6em 2em 0.875em 2em;
                    margin: auto;
                    -webkit-transition: padding .2s ease, margin .2s ease;
                    transition: padding .2s ease, margin .2s ease;
                }}
                .form__field {{
                  -webkit-transition: margin .2s ease;
                  transition: margin .2s ease;
                }}
                div.doc {{
                    color:black;
                    font-size: 16px;
                    padding-left: 5px;
                    padding-top: 5px;
                    padding-bottom: 5px;
                    padding-right: 5px;
                    margin-bottom: 10px;
                    line-height: 40px;
                }}
            </style>
        </head>
        <body>
            <div class="model__content">
                {html_body}
            </div>
            <script>
                {script_body}
                $(document).ready(function(){{
                    $('[data-toggle="tooltip"]').tooltip();
                }});
            </script>
        </body>
    </html>
    """


def main(args):
    graphs_data, _ = read_data(args.graphs_input, args)
    nlp_data, _ = read_data(args.nlp_input, args)
    if args.num_splits is None or args.num_splits <= 1:
        all_instances = []
        num_graphs = 0
        num_graphs_with_distractor = 0
        num_distractors = 0
        for d_nlp, d_graphs in zip(tqdm(nlp_data), graphs_data):
            instance = get_instance_for_render(d_nlp, d_graphs)
            for obj in instance:
                num_graphs += 1
                num_graphs_with_distractor += int(any(tag[0] in ['not_in_graph', 'srl'] for tag in obj['doc_sp2tags']))
                num_distractors += sum(tag[0] in ['not_in_graph', 'srl'] for tag in obj['doc_sp2tags'])
            all_instances.append(instance)
        html_body = "".join(render_entity_events_graphs(ins) for ins in all_instances)
        script_body = "\n".join(render_graphviz_objects(ins) for ins in all_instances)
        index_html_string = render_index_html(html_body, script_body)
        with open(args.output, 'w') as f:
            f.write(index_html_string)
        print(num_graphs_with_distractor / num_graphs, num_graphs_with_distractor, num_graphs)
        print(num_distractors / num_graphs, num_distractors, num_graphs)
    else:
        batch = len(nlp_data) // args.num_splits
        for start in range(0, len(nlp_data), batch):
            end = start + batch
            all_instances = []
            for d_nlp, d_graphs in zip(tqdm(nlp_data[start:end]), graphs_data[start:end]):
                instance = get_instance_for_render(d_nlp, d_graphs)
                all_instances.append(instance)
            html_body = "".join(render_entity_events_graphs(ins) for ins in all_instances)
            script_body = "\n".join(render_graphviz_objects(ins) for ins in all_instances)
            index_html_string = render_index_html(html_body, script_body)
            with open(args.output+'_{:d}-{:d}_index.html'.format(start, end), 'w') as f:
                f.write(index_html_string)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="create index html for rendering entity event graphs")
    parser.add_argument("--graphs_input", help="input path to load entity event graphs")
    parser.add_argument("--nlp_input", help="input path to nlp annotated data")
    parser.add_argument("--output", default="./index.html", help="output path for index.html")
    parser.add_argument('--start', type=int, help='start idx of data to be processed', default=-1)
    parser.add_argument('--end', type=int, help='end idx of data to be processed', default=-1)
    parser.add_argument('--num_splits', type=int, help='split outputs to different files', default=None)
    args = parser.parse_args()

    main(args)
