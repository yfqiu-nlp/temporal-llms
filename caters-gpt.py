# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import pandas as pd
from tqdm import tqdm

import time
import torch
import numpy as np
# device = torch.device('cuda')

from accelerate import Accelerator
from accelerate import init_empty_weights
from accelerate import load_checkpoint_and_dispatch

from collections import defaultdict
import openai
import argparse

parser = argparse.ArgumentParser(description='GPT-CATERS')

# Saving paramerters
parser.add_argument('--model_name', type=str, help='Name of GPT model used to test.')
parser.add_argument('--output_path', type=str, help='Path for GPT model ouptuts.')

# Decoding Parameters
parser.add_argument('--temperature', type=float, default=0, help='Decoding temperature.')
parser.add_argument('--top_p', type=float, default=1.0, help='Decoding top-p.')
parser.add_argument('--max_events_length', type=int, default=1900, help='Max input article length.')
parser.add_argument('--max_batch_size', type=int, default=1, help='Max batch size.')
parser.add_argument('--max_new_decoding_tokens', type=int, default=2, help='Max new tokens for decoding output.')
parser.add_argument('--num_example', type=int, default=3, help='Number of examples used for ICL.')

# Experiment Parameters
parser.add_argument("--caters_eval", action= "store_true", help = "Do caters original experiment")
parser.add_argument('--train_set', type=str, help='Path for CATERS training set.')

# LLM ability test
parser.add_argument("--do_in_context_learning", action= "store_true", help = "Do ICL prompting.")

args = parser.parse_args()


print("Loading the Model and Tokenizer...")
print("Model:", args.model_name)

openai_apikey=""
openai.api_key = openai_apikey

if args.caters_eval:
    print("Evaluating "+args.model_name+" on CATERS...")
if args.do_in_context_learning:
    print("Prompting Strategy: In-context Learning.")
else:
    print("Prompting Strategy: Zero-shot direct prompting.")
    
    
def make_prompts(dataset, do_icl=False):
    
    sources=[s for s in dataset[['source']].values.tolist()]
    
    prompts = []
    
    # Making prompts from training set
    df = pd.read_csv(args.train_set, sep='\t')
    
    df = df.sample(n=args.num_example, random_state=44).values.tolist()
    
    instruction="Following the given template to order the events according to temporality: \n"
    
    fs_examplrs=''
    for src, tgt in df:
        example_str = "Input: "+src+"\nOutput: "+tgt+"\n"
        fs_examplrs+=example_str
    # fs_examplrs="Input: <EVENT> She basically stated everything that I wrote <EVENT> Nancy who contributed nothing to the project\nOutput: <EVENT> contributed <ARGS> Nancy who contributed nothing to the project <EVENT> stated <ARGS> She basically stated everything that I wrote\nInput: <EVENT> he tell me Happy Birthday <EVENT> I thought he was going to tell me Happy Birthday\nOutput: <EVENT> thought <ARGS> I thought he was going to tell me Happy Birthday <EVENT> tell <ARGS> he tell me Happy Birthday\nInput: <EVENT> Finally as dawn broke they woke their parents <EVENT> They ran downstairs to eagerly open presents\nOutput: <EVENT> woke <ARGS> Finally as dawn broke they woke their parents <EVENT> ran <ARGS> They ran downstairs to eagerly open presents\n"

    for source in sources:
        if do_icl == False:
            print("Only do ICL for CATERS.")
            1/0
        elif do_icl == True:
            p = instruction+fs_examplrs+"Input: {source}\nOutput: ".format(source=source[0])
            prompts.append(p)

    return prompts

if args.caters_eval:

    dataset = pd.read_csv("./dataset/caters/caters.csv", sep='\t')
    
    if args.do_in_context_learning:
        prompts = make_prompts(dataset, do_icl=True)
    else:
        # prompts = make_prompts(dataset, tokenizer, do_icl=False, prompt_style=args.prompt_style)
        1/0
    
    answers = []
    for p in tqdm(prompts):
        if args.model_name=="gpt-4":
            try:
                output=openai.ChatCompletion.create(
                    model=args.model_name,
                    messages=[
                            {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible. Knowledge cutoff: {knowledge_cutoff} Current date: {current_date}"},
                            {"role": "user", "content": p}
                        ]
                    )
            except:
                time.sleep(5)
                output=openai.ChatCompletion.create(
                    model=args.model_name,
                    messages=[
                            {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible. Knowledge cutoff: {knowledge_cutoff} Current date: {current_date}"},
                            {"role": "user", "content": p}
                        ]
                    )
                time.sleep(5)
            answers.append(output['choices'][0]['message']['content']+'\n')
        else:
            try:
                output = openai.Completion.create(
                        model=args.model_name,
                        prompt=p,
                        temperature=args.temperature,
                        max_tokens=args.max_new_decoding_tokens,
                        top_p=args.top_p,
                        frequency_penalty=0.0,
                        presence_penalty=0.0
                    )
            except:
                print("Now retry the API call.")
                time.sleep(5)
                output = openai.Completion.create(
                        model=args.model_name,
                        prompt=p,
                        temperature=args.temperature,
                        max_tokens=args.max_new_decoding_tokens,
                        top_p=args.top_p,
                        frequency_penalty=0.0,
                        presence_penalty=0.0
                    )
                time.sleep(5)
            answers.append(output['choices'][0]['text'].replace('\n','')+'\n')

    with open(args.output_path+"/caters-test-"+args.model_name+".out", "w+") as f:
        f.writelines(answers)
