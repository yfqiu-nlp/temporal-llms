# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import pandas as pd
from tqdm import tqdm

from transformers import GPT2Tokenizer
import torch
import numpy as np
# device = torch.device('cuda')

from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import load_dataset
import openai
from tqdm.contrib import tzip
import time
import pdb

openai_apikey=""
openai.api_key = openai_apikey

import argparse

parser = argparse.ArgumentParser(description='GPT-MC-TACO')

# Saving paramerters
parser.add_argument('--model_name', type=str, help='Name of LLaMA model used to test.')
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

# LLM ability test
parser.add_argument("--do_in_context_learning", action= "store_true", help = "Do ICL prompting.")

args = parser.parse_args()

print("Model:", args.model_name)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print("Finish Loading!")

if args.mctaco_eval:
    print("Evaluating "+args.model_name+" on MC-TACO...")
if args.do_in_context_learning:
    print("Prompting Strategy: In-context Learning.")
else:
    print("Prompting Strategy: Zero-shot direct prompting.")
print("Prompt style used for evaluation:", args.prompt_style)
    
    
def make_prompts(dataset, tokenizer, do_icl=False, prompt_style="mcq"):
    passages=[q[0] for q in dataset[['passage']].values]
    questions=[e[0] for e in dataset[['question']].values]
    answers=[e[0] for e in dataset[['answer']].values]
    
    completion_answer_length = [len(tokenizer(ans).input_ids) for ans in answers]
    
    assert len(passages) == len(questions)
    assert len(answers) == len(questions)
    prompts = []
    
    if prompt_style == 'mcq':
        mcq_dataset=defaultdict(list)
        idx=0 # recall the global idx
        question2passage={}
        for q,p,a in zip(questions, passages, answers):
            mcq_dataset[q].append((idx,a))
            question2passage[q] = p
            idx+=1

        fs_examplrs="Answer the following multiple-choice question with candidate answers according to the given passage. There can be multiple correct answers.\nPassage: Islam later emerged as the majority religion during the centuries of Ottoman rule, though a significant Christian minority remained.\nQuestion: What happened before Islam was the majority religion?\nCandidate answers: (A) christianity was the majority religion (B) the end of white-minority rule (C) he emerged as the heir apparent\nThe correct answer(s) is/are: (A) christianity was the majority religion\n\nAnswer the following multiple-choice question with candidate answers according to the given passage. There can be multiple correct answers.\nPassage: It's hail crackled across the comm, and Tara spun to retake her seat at the helm.\nQuestion: How long was the storm?\nCandidate answers: (A) an year (B) 6 hours (C) an hour (D) an week (E) 2 hours (F) a hour (G) 2 seconds\nThe correct answer(s) is/are: (B) 6 hours (C) an hour (E) 2 hours (F) a hour\n\nAnswer the following multiple-choice question with candidate answers according to the given passage. There can be multiple correct answers.\nPassage: His counter-attack with Dayak warriors drove the Chinese out of Bau and across the Sarawak border.\nQuestion: What time did the battle end?\nCandidate answers: (A) 7:00 PM (B) a minute before it started (C) 7:00 AM (D) midnight (E) 5:00 AM\nThe correct answer(s) is/are: (A) 7:00 PM (C) 7:00 AM (E) 5:00 AM\n\n"

        index = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)', '(N)', '(M)', '(O)', '(P)', '(Q)', '(R)', '(S)', '(T)', '(U)', '(V)', '(W)', '(X)', '(Y)', '(Z)']

        for q in mcq_dataset:
            context="Answer the following multiple-choice question with candidate answers according to the given passage. There can be multiple correct answers.\n"
            question = "Question: "+q+"\n"
            passage = "Passage: "+question2passage[q]+"\n"

            candidates = []
            for i, items in zip(index, mcq_dataset[q]):
                idx, candidate = items
                candidates.append(str(i)+' '+candidate)
            candidate_str = "Candidate answers: "+' '.join(candidates)+'\n'

            for i, items in zip(index, mcq_dataset[q]):
                idx, candidate = items
                if args.model_name != "gpt-4":
                    if do_icl == False: # Zero-shot
                        prompts.append(context+passage+question+candidate_str+"The correct answer(s) is/are: "+str(i)+' '+candidate)
                    elif do_icl == True: # Few-shot
                        prompts.append(fs_examplrs+context+passage+question+candidate_str+"The correct answer(s) is/are: "+str(i)+' '+candidate)
                elif args.model_name == "gpt-4":
                    if do_icl == False: # Zero-shot
                        prompts.append(context+passage+question+candidate_str+"The correct answer(s) is/are: ")
                    elif do_icl == True: # Few-shot
                        prompts.append(fs_examplrs+context+passage+question+candidate_str+"The correct answer(s) is/are: ")
    elif prompt_style == 'qa':
        prompt_format="Answer the question according to the passage.\nPassage: {passage}\nQuestion: {question}\nThe answer is: {answer}"
        fs_examplrs="Answer the question according to the passage.\nPassage: the majority religion during the centuries of Ottoman rule, though a significant Christian minority remained.\nQuestion: What happened before Islam was the majority religion?\nThe answer is: christianity was the majority religion\n\nAnswer the question according to the passage.\nPassage: It's hail crackled across the comm, and Tara spun to retake her seat at the helm.\nQuestion: How long was the storm?\nThe answer is: 6 hours\n\nAnswer the question according to the passage.\nPassage: His counter-attack with Dayak warriors drove the Chinese out of Bau and across the Sarawak border.\nQuestion: What time did the battle end?\nThe answer is: 7:00 PM\n\n"
        for q,p,a in zip(questions, passages, answers):
            if args.model_name != "gpt-4":
                if do_icl == False:
                    prompts.append(prompt_format.format(passage=p, question=q, answer=a))
                elif do_icl == True:
                    prompts.append(fs_examplrs+prompt_format.format(passage=p, question=q, answer=a))
            elif args.model_name == "gpt-4":
                    if do_icl == False: # Zero-shot
                        prompts.append(context+passage+question+candidate_str+"The correct answer(s) is/are: ")
                    elif do_icl == True: # Few-shot
                        prompts.append(fs_examplrs+context+passage+question+candidate_str+"The correct answer(s) is/are: ")
    else:
        print("Plese specify the prompt style used for mctaco dataset.")
        
    assert len(prompts) == len(questions)
    
    return prompts, completion_answer_length

if args.mctaco_eval:

    dataset = pd.read_csv("./dataset/mctaco/test_9442.tsv", sep='\t', names=["passage", "question", "answer", "justification", "reasoning_type"])
    
    if args.do_in_context_learning:
        prompts, completion_answer_length = make_prompts(dataset, tokenizer, do_icl=True, prompt_style=args.prompt_style)
    else:
        prompts, completion_answer_length = make_prompts(dataset, tokenizer, do_icl=False, prompt_style=args.prompt_style)
       
    if args.model_name != "gpt-4": # completion, using likelihood
        log_p = []
        for p,ans_len in tzip(prompts, completion_answer_length):
            output = openai.Completion.create(
                model=args.model_name,
                prompt=p,
                temperature=args.temperature,
                max_tokens=args.max_new_decoding_tokens,
                top_p=args.top_p,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                logprobs=1,
                echo=True,
            )
            completion = output.choices[0].logprobs.tokens[-1*ans_len:]
            likelihood = output.choices[0].logprobs.token_logprobs[-1*ans_len:]
            normalized_likelihood = np.sum(likelihood) / len(' '.join(completion))
            log_p+=normalized_likelihood
            
        with open(args.output_path+"/mctaco-test-"+args.model_name+".out", "w+") as f:
            f.writelines([str(d)+'\n' for d in log_p])
    
    
    elif args.model_name == 'gpt-4': # chat mode, using decoding+string match
        answers = [] 
        for p in tqdm(prompts):
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
            time.sleep(3)
            answers.append(output['choices'][0]['message']['content']+'\n')
        with open(args.output_path+"/mctaco-test-"+args.model_name+".out", "w+") as f:
            f.writelines([str(d)+'\n' for d in answers])
    
