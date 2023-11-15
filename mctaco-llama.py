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

import argparse
import os

parser = argparse.ArgumentParser(description='LLaMA-MC-TACO')

# Saving paramerters
parser.add_argument('--model_name', type=str, help='Name of LLaMA model used to test.')
parser.add_argument('--model_path', type=str, help='Path for LLaMA model.')
parser.add_argument('--output_path', type=str, help='Path for LLaMA model ouptuts.')

# Decoding Parameters
parser.add_argument('--temperature', type=float, default=0, help='Decoding temperature.')
parser.add_argument('--top_p', type=float, default=1.0, help='Decoding top-p.')
parser.add_argument('--max_events_length', type=int, default=1900, help='Max input article length.')
parser.add_argument('--max_batch_size', type=int, default=1, help='Max batch size.')
parser.add_argument('--max_new_decoding_tokens', type=int, default=2, help='Max new tokens for decoding output.')

# Experiment Parameters
parser.add_argument("--mctaco_eval", action= "store_true", help = "Do mctaco original experiment")
parser.add_argument('--prompt_style', type=str, default='mcq', help='Prompt style used for McTACO dataset. Either mcq or qa.')
parser.add_argument('--prompt_template', type=int, default=1, help='Template to use for the prompt.')
parser.add_argument('--num_example', type=int, default=3, help='Number of examples used for ICL.')

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

if args.mctaco_eval:
    print("Evaluating "+args.model_name+" on MC-TACO...")
if args.do_in_context_learning:
    print("Prompting Strategy: In-context Learning.")
else:
    print("Prompting Strategy: Zero-shot direct prompting.")
print("Prompt style used for evaluation:", args.prompt_style)
    
    
def make_prompts(dataset, tokenizer, do_icl=False, prompt_style="mcq", append_label="yes"):
    passages=[q[0] for q in dataset[['passage']].values]
    questions=[e[0] for e in dataset[['question']].values]
    answers=[e[0] for e in dataset[['answer']].values]
    
    assert len(passages) == len(questions)
    assert len(answers) == len(questions)
    prompts = []
    
    if prompt_style == 'qa':
        if args.prompt_template == 1:
            prompt_format="Is the following candidate answer to the question true or false according to the passage?\nPassage: {passage}\nQuestion: {question}\nCandidate answer: {answer}\nThe answer is: "+append_label
            fs_examplrs_list=[
                """Is the following candidate answer to the question true or false according to the passage?
Passage: the majority religion during the centuries of Ottoman rule, though a significant Christian minority remained.
Question: What happened before Islam was the majority religion?
Candidate answer: christianity was the majority religion
The answer is: true""",

"""Is the following candidate answer to the question true or false according to the passage?
Passage: It's hail crackled across the comm, and Tara spun to retake her seat at the helm.
Question: How long was the storm?
Candidate answer: 6 years
The answer is: false""",

"""Is the following candidate answer to the question true or false according to the passage?
Passage: His counter-attack with Dayak warriors drove the Chinese out of Bau and across the Sarawak border.
Question: What time did the battle end?
Candidate answer: 7:00 PM
The answer is: true""",

"""Is the following candidate answer to the question true or false according to the passage?
Passage: In 1930, the poet Muhammad Iqbal proposed a separate Muslim homeland in the northwest of India.
Question: How long did Muhammad Iqbal consider his proposal?
Candidate answer: 0.56 seconds
The answer is: false""",

"""Is the following candidate answer to the question true or false according to the passage?
Passage: He them imprisons the royal family in his prison .
Question: What happened after word spread of the royal family being imprisoned?
Candidate answer: he and his family doing odd jobs
The answer is: false""",

"""Is the following candidate answer to the question true or false according to the passage?
Passage: Among the responses the Swiss trader got was one from the Soviet national shipping company, which hadn't been invited to submit a bid.
Question: How long did the bidding last?
Candidate answer: two days
The answer is: true""",

"""Is the following candidate answer to the question true or false according to the passage?
Passage: By the beginning of the fourth century the Algarve had a bishop in place, based in Faro.
Question: What happened after the Algarve installed the bishop?
Candidate answer: the state began collecting corporate taxes
The answer is: true""",

"""Is the following candidate answer to the question true or false according to the passage?
Passage: The legal system marketplace just doesn't serve low-income people too well, except in fee-generat-ing type cases, Brewer said.
Question: When did Brewer talk?
Candidate answer: 1:00 PM
The answer is: true""",

"""Is the following candidate answer to the question true or false according to the passage?
Passage: Lennon accuses his father of leaving him again , and then leaves , after telling his father that he wo n't live with him anymore .
Question: When did Lennon's father return?
Candidate answer: after he left earlier
The answer is: false""",

"""Is the following candidate answer to the question true or false according to the passage?
Passage: A majority of 65 votes in the 128-member body was required to reject his reinstatement.
Question: How often are elections held?
Candidate answer: every 2 weeks
The answer is: false"""]

            # fs_examplrs="Answer the question according to the passage.\nPassage: Islam later emerged as the majority religion during the centuries of Ottoman rule, though a significant Christian minority remained.\nQuestion: What happened before Islam was the majority religion?\nThe answer is: christianity was the majority religion\n\nAnswer the question according to the passage.\nPassage: It's hail crackled across the comm, and Tara spun to retake her seat at the helm.\nQuestion: How long was the storm?\nThe answer is: 6 hours\n\nAnswer the question according to the passage.\nPassage: His counter-attack with Dayak warriors drove the Chinese out of Bau and across the Sarawak border.\nQuestion: What time did the battle end?\nThe answer is: 7:00 PM\n\n"
            fs_examplrs="\n\n".join(fs_examplrs_list[:args.num_example])
            fs_examplrs+='\n\n'
        elif args.prompt_template == 2:
            prompt_format="Based on the information presented in the passage \"{passage}\", can the candidate answer \"{answer}\" answer the question \"{question}\"? The answer is: "+append_label
            
            fs_examplrs="""Based on the information presented in the passage \"the majority religion during the centuries of Ottoman rule, though a significant Christian minority remained.\", can the candidate answer \"christianity was the majority religion\" answer the question \"What happened before Islam was the majority religion?\"? The answer is: yes

Based on the information presented in the passage \"It's hail crackled across the comm, and Tara spun to retake her seat at the helm.\", can the candidate answer \"6 years\" answer the question \"How long was the storm?\"? The answer is: no 

Based on the information presented in the passage \"His counter-attack with Dayak warriors drove the Chinese out of Bau and across the Sarawak border.\", can the candidate answer \"7:00 PM\" answer the question \"What time did the battle end?\"? The answer is: yes

"""
        elif args.prompt_template == 3:
            if do_icl: # special case for few-shot since instruction is no need to be repeated
                prompt_format="According to the passage \"{passage}\", is the candidate answer \"{answer}\" correct to the question \"{question}\"? The answer is "+append_label
            else:
                prompt_format="Finish the following text: According to the passage \"{passage}\", is the candidate answer \"{answer}\" correct to the question \"{question}\"? The answer is "+append_label

            fs_examplrs="""Finish the following texts:
According to the passage \"the majority religion during the centuries of Ottoman rule, though a significant Christian minority remained.\", is the candidate answer \"christianity was the majority religion\" correct to the question \"What happened before Islam was the majority religion?\"? The answer is correct

According to the passage \"It's hail crackled across the comm, and Tara spun to retake her seat at the helm.\", is the candidate answer \"6 years\" correct to the question \"How long was the storm?\"? The answer is incorrect

According to the passage \"His counter-attack with Dayak warriors drove the Chinese out of Bau and across the Sarawak border.\", is the candidate answer \"7:00 PM\" correct to the question \"What time did the battle end?\"? The answer is correct

"""
        for q,p,a in zip(questions, passages, answers):
            if do_icl: # Few-shot
                prompts.append(fs_examplrs+prompt_format.format(passage=p, question=q, answer=a))
            else: # Zero-shot
                prompts.append(prompt_format.format(passage=p, question=q, answer=a))
    else:
        print("Plese specify the prompt style used for mctaco dataset.")
        
    assert len(prompts) == len(questions)
    
    return prompts

def to_tokens_and_logprobs(model, tokenizer, input_texts):
    input_ids = tokenizer(input_texts, padding=False, return_tensors="pt").input_ids.to(device)
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

    batch = []
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        for token, p in zip(input_sentence, input_probs):
            if token not in tokenizer.all_special_ids:
                text_sequence.append((tokenizer.decode(token), p.item()))
        batch.append(text_sequence)
    return batch


if args.mctaco_eval:

    dataset = pd.read_csv("./dataset/mctaco/test_9442.tsv", sep='\t', names=["passage", "question", "answer", "justification", "reasoning_type"])
    
    if args.prompt_template == 1:
        pos_label = 'true'
        neg_label = 'false'
    elif args.prompt_template == 2:
        pos_label = 'yes'
        neg_label = 'no'
    elif args.prompt_template == 3:
        pos_label = 'correct'
        neg_label = 'incorrect'
            
    if args.do_in_context_learning:
        yes_prompts = make_prompts(dataset, tokenizer, do_icl=True, prompt_style=args.prompt_style, append_label=pos_label)
        no_prompts = make_prompts(dataset, tokenizer, do_icl=True, prompt_style=args.prompt_style, append_label=neg_label)
    else:
        yes_prompts = make_prompts(dataset, tokenizer, do_icl=False, prompt_style=args.prompt_style, append_label=pos_label)
        no_prompts = make_prompts(dataset, tokenizer, do_icl=False, prompt_style=args.prompt_style, append_label=neg_label)
    
    yes_log_p = []
    no_log_p = []
    
    assert len(yes_prompts) == len(no_prompts)
    for i in tqdm(range(0, len(yes_prompts), args.max_batch_size)):
        with torch.no_grad():
            # Prediction for yes label
            yes_batch_p = yes_prompts[i:i+args.max_batch_size]
            log_p_outputs = to_tokens_and_logprobs(model, tokenizer, yes_batch_p) # larger is better
            label_log_p=[d[-1] for d in log_p_outputs] # get the log_p for yes/no token
            yes_log_p+=label_log_p

            # Prediction for no label
            no_batch_p = no_prompts[i:i+args.max_batch_size]
            log_p_outputs = to_tokens_and_logprobs(model, tokenizer, no_batch_p) # larger is better
            label_log_p=[d[-1] for d in log_p_outputs] # get the log_p for yes/no token
            no_log_p+=label_log_p

    answers=[]
    # Comparing yes and no
    for yes_d, no_d in zip(yes_log_p, no_log_p):
        assert (yes_d[0].strip() == 'yes') or (yes_d[0].strip() == 'correct') or (yes_d[0].strip() =='true')
        assert (no_d[0].strip() == 'no') or (no_d[0].strip() == 'incorrect') or (no_d[0].strip() =='false')
        if yes_d[1] >= no_d[1]:
            answers.append('yes')
        else:
            answers.append('no')

    with open(args.output_path+"/mctaco-test-"+args.model_name+".out", "w+") as f:
        f.writelines([str(d)+'\n' for d in answers])