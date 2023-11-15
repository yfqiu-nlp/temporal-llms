# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import pandas as pd
from tqdm import tqdm

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
# device = torch.device('cuda')

from accelerate import Accelerator
from accelerate import init_empty_weights
from accelerate import load_checkpoint_and_dispatch

from collections import defaultdict

accelerator = Accelerator()
device = accelerator.device

import argparse, os

parser = argparse.ArgumentParser(description='LLaMA-CATERS')

# Saving paramerters
parser.add_argument('--model_name', type=str, help='Name of LLaMA model used to test.')
parser.add_argument('--model_path', type=str, help='Path for LLaMA model.')
parser.add_argument('--tokenizer_path', type=str, help='Path for LLaMA tokenizer.')
parser.add_argument('--output_path', type=str, help='Path for LLaMA model ouptuts.')

# Decoding Parameters
parser.add_argument('--temperature', type=float, default=0, help='Decoding temperature.')
parser.add_argument('--top_p', type=float, default=1.0, help='Decoding top-p.')
parser.add_argument("--do_sample", action= "store_true", help = "Required if using temperature")
parser.add_argument('--max_events_length', type=int, default=1900, help='Max input article length.')
parser.add_argument('--max_batch_size', type=int, default=1, help='Max batch size.')
parser.add_argument('--max_new_decoding_tokens', type=int, default=2, help='Max new tokens for decoding output.')
parser.add_argument('--num_example', type=int, default=3, help='Number of examples used for ICL.')

# Experiment Parameters
parser.add_argument("--caters_eval", action= "store_true", help = "Do caters original experiment")
parser.add_argument('--train_set', type=str, help='Path for CATERS training set.')
parser.add_argument('--prompt_template', type=int, default=1, help='Template to use for the prompt.')

# LLM ability test
parser.add_argument("--do_in_context_learning", action= "store_true", help = "Do ICL prompting.")

args = parser.parse_args()


print("Loading the Model and Tokenizer...")
print("Model:", args.model_name)
print("Loading from:", args.model_path)

# Loading model with accelerate method
if "Llama-2" in args.model_name:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True) # llama2
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto", torch_dtype=torch.float16)
else:
    with init_empty_weights():
        config = AutoConfig.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_config(config)
        # model = LlamaForCausalLM.from_config(config)
    model.tie_weights()
    model = load_checkpoint_and_dispatch(
        model, args.model_path, device_map="auto", no_split_module_classes=["LlamaDecoderLayer"], dtype=torch.float16
    )

    model.eval()
    print(model.hf_device_map)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side='left') # batch decoding
    print("Finish Loading!")


if not os.path.exists(args.output_path):
   os.makedirs(args.output_path)

if args.caters_eval:
    print("Evaluating "+args.model_name+" on CATERS...")
if args.do_in_context_learning:
    print("Prompting Strategy: In-context Learning.")
else:
    print("Prompting Strategy: Zero-shot direct prompting.")
    
    
def make_prompts(dataset, tokenizer, do_icl=False):
    
    sources=[s for s in dataset[['source']].values.tolist()]
    
    prompts = []
    
    # Making prompts from training set
    df = pd.read_csv(args.train_set, sep='\t')
    
    if args.num_example == 10:
        # doint a special case for 10 examples, since example 7+8+9 exceed max length
        df = df.sample(n=20, random_state=44).values.tolist()
        df = df[0:7]+df[10:11]+df[11:12]+df[15:16]
    else:
        df = df.sample(n=args.num_example, random_state=44).values.tolist()
    
    
    if args.prompt_template == 1:
        instruction="Following the given template to order the events according to temporality: \n"
        fs_examplrs=''
        for src, tgt in df:
            example_str = "Input: "+src+"\nOutput: "+tgt+"\n"
            fs_examplrs+=example_str
        # fs_examplrs="Input: <EVENT> She basically stated everything that I wrote <EVENT> Nancy who contributed nothing to the project\nOutput: <EVENT> contributed <ARGS> Nancy who contributed nothing to the project <EVENT> stated <ARGS> She basically stated everything that I wrote\nInput: <EVENT> he tell me Happy Birthday <EVENT> I thought he was going to tell me Happy Birthday\nOutput: <EVENT> thought <ARGS> I thought he was going to tell me Happy Birthday <EVENT> tell <ARGS> he tell me Happy Birthday\nInput: <EVENT> Finally as dawn broke they woke their parents <EVENT> They ran downstairs to eagerly open presents\nOutput: <EVENT> woke <ARGS> Finally as dawn broke they woke their parents <EVENT> ran <ARGS> They ran downstairs to eagerly open presents\n"
    elif args.prompt_template == 2:
        instruction=''
        fs_examplrs=''
        for src, tgt in df:
            example_str = "Based on temporality in the given events \""+src+"\", arrange the events in temporal order. The order is: "+tgt+"\n"
            fs_examplrs+=example_str
    elif args.prompt_template == 3:
        instruction="Finish the following texts: \n"
        fs_examplrs=''
        for src, tgt in df:
            example_str = "According to the temporality in the given events \""+src+"\", the temporal order of the events is: "+tgt+"\n"
            fs_examplrs+=example_str
    
    
    for source in sources:
        if do_icl == False:
            print("Only do ICL for CATERS.")
            1/0
        elif do_icl == True:
            if args.prompt_template == 1:
                p = instruction+fs_examplrs+"Input: {source}\nOutput: ".format(source=source[0])
            elif args.prompt_template == 2:
                p = fs_examplrs+"Based on temporality in the given events \"{source}\", arrange the events in temporal order. The order is: ".format(source=source[0])
            elif args.prompt_template == 3:
                p = instruction+fs_examplrs+"According to the temporality in the given events \"{source}\", the temporal order of the events is: ".format(source=source[0])
            prompts.append(p)

    return prompts

if args.caters_eval:

    dataset = pd.read_csv("./dataset/caters/caters.csv", sep='\t')
    
    if args.do_in_context_learning:
        prompts = make_prompts(dataset, tokenizer, do_icl=True)
    else:
        # prompts = make_prompts(dataset, tokenizer, do_icl=False, prompt_style=args.prompt_style)
        1/0
    
    answers = []
    for i in tqdm(range(0, len(prompts), args.max_batch_size)):
        with torch.no_grad():
            batch_p = prompts[i:i+args.max_batch_size]
            inputs = tokenizer(batch_p, return_tensors="pt", padding=False).to(device) # batch decoding
            generate_ids = model.generate(**inputs, max_new_tokens=args.max_new_decoding_tokens, temperature=args.temperature, top_p=args.top_p, do_sample=args.do_sample)
            generate_ids=[ids[len(inputs['input_ids'][i]):] for i, ids in enumerate(generate_ids)] # truncate to get the raw output.
            outputs=tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            answers+=[output.replace('\n', ' ')+'\n' for output in outputs]

    with open(args.output_path+"/caters-test-"+args.model_name+".out", "w+") as f:
        f.writelines(answers)
