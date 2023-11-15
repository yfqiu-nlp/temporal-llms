# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import pandas as pd
from tqdm import tqdm

from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from transformers import LlamaModel, LlamaConfig

import torch
# device = torch.device('cuda')

from accelerate import Accelerator
from accelerate import init_empty_weights
from accelerate import load_checkpoint_and_dispatch

accelerator = Accelerator()
device = accelerator.device

import argparse
import os

parser = argparse.ArgumentParser(description='LLaMA-TempEval')

# Saving paramerters
parser.add_argument('--model_name', type=str, help='Name of LLaMA model used to test.')
parser.add_argument('--model_path', type=str, help='Path for LLaMA model.')
parser.add_argument('--output_path', type=str, help='Path for LLaMA model ouptuts.')
parser.add_argument('--tokenizer_path', type=str, help='Path for LLaMA tokenizer.')
parser.add_argument('--data_path', type=str, help='Path for dataset.')

# Decoding Parameters
parser.add_argument('--temperature', type=float, default=0, help='Decoding temperature.')
parser.add_argument("--do_sample", action= "store_true", help = "Required if using temperature")
parser.add_argument('--top_p', type=float, default=1.0, help='Decoding top-p.')
parser.add_argument('--max_events_length', type=int, default=1900, help='Max input article length.')
parser.add_argument('--max_batch_size', type=int, default=1, help='Max batch size.')
parser.add_argument('--max_new_decoding_tokens', type=int, default=2, help='Max new tokens for decoding output.')

# Experiment Parameters
parser.add_argument("--temp_eval", action= "store_true", help = "Do temp_eval original experiment")
parser.add_argument("--bidirectional_temp_eval_likelihood", action= "store_true", help = "Do temp_eval bidirectional experiment using likelihood")
parser.add_argument("--bidirectional_temp_eval_decoding", action= "store_true", help = "Do temp_eval bidirectional experiment using decoding")
parser.add_argument('--prompt_template', type=int, default=1, help='Template to use for the prompt.')
parser.add_argument('--num_example', type=int, default=3, help='Number of examples used for ICL.')

# LLM ability test
parser.add_argument("--do_in_context_learning", action= "store_true", help = "Do ICL prompting.")# LLM ability test
parser.add_argument("--do_chain_of_thought", action= "store_true", help = "Do CoT prompting.")

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


def make_zs_prompt(dataset, tokenizer, label):
    questions=[q[0] for q in dataset[['question']].values]
    events=[e[0] for e in dataset[['events']].values]
    if args.prompt_template == 1:
        prompt_format="Answer the question according to the article. Only answer yes or no. \nArticle: {events}. \nQuestion: {question}. \nThe answer is: "+label
    elif args.prompt_template == 2:
        prompt_format="Based on the information presented in the article \"{events}\", answer the question \"{question}\" with yes or no. The answer is: "+label
    elif args.prompt_template == 3:
        prompt_format="Finish the following text: \nArticle: \"{events}\"\nThe answer to the yes or no question \"{question}\" according to the the article is: "+label
    elif args.prompt_template == 4:
        prompt_format="Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nAnswer the question according to the article. Only answer yes or no.\n\n### Input:\nArticle: {events}.\nQuestion: {question}\n\n### Response:\n"+label
    assert len(questions) == len(events)
    prompts = []
    for q, e in zip(questions, events):
        e = tokenizer.encode(e)
        if len(e) >= args.max_events_length: # truncation to max_events_len
            e = e[1:args.max_events_length-1] # Skip </s> at the begining and end of seq
        else:
            e = e[1:-1]
        e = tokenizer.decode(e)
        prompts.append(prompt_format.format(events=e, question=q))
    return prompts

def make_icl_prompt(dataset, tokenizer, label):
    questions=[q[0] for q in dataset[['question']].values]
    events=[e[0] for e in dataset[['events']].values]

    if args.prompt_template == 1:
        if args.do_chain_of_thought:
            prompt_format="Answer the question according to the article. Give step-by-step explanations and then answer yes or no. \nArticle: {events}. \nQuestion: {question}. \nThe step-by-step explanation and answer is: "
            fs_examplrs="Answer the question according to the article. Give step-by-step explanations and then answer yes or no. \nArticle: Tired of being sidelined, Hungarian astronaut Bertalan Farkas is leaving for the United States to start a new career, he said Saturday. \"Being 48 is too early to be retired,\" a fit-looking Farkas said on state TV's morning talk show. With American astronaut Jon McBride, Farkas set up an American-Hungarian joint venture called Orion 1980, manufacturing space-travel related technology. Farkas will move to the company's U.S. headquarters. Farkas, an air force captain, was sent into space on board the Soyuz 36 on May 26, 1980. He spent six days aboard the Salyut 6 spacecraft with three Soviet astronauts, Valery Kubasov, Leonid Popov and Valery Riumin. McBride, 54, of Lewisburg, West Virginia, was part of a seven-member crew aboard the Orbiter Challenger in October 1984 and later served as assistant administrator for congressional relations for NASA. Farkas expressed the hope he one day follow in the footsteps of fellow astronaut John Glenn, who at 77 is about to go into space again. On May 22, 1995, Farkas was made a brigadier general, and the following year he was appointed military attache at the Hungarian embassy in Washington. However, cited by District of Columbia traffic police in December for driving under the influence of alcohol, Farkas was ordered home and retired."
            fs_examplrs_question_list=[
                "\nQuestion: Is Farkas sent into space on board the Soyuz before McBride on board the Orbiter Challenger?\nThe step-by-step explanation and answer is: Farkas was sent into space on May 26, 1980 and McBride was on board the Orbiter Challenger in October 1984. May 26, 1980 is before October 1984. So the answer is: yes.",
                "\nQuestion: Is McBride on board the Orbiter Challenger after Farkas was made a brigadier general?\nThe step-by-step explanation and answer is: McBride was on board the Orbiter Challenger in October 1984. Farkas was made a brigadier general on May 22, 1995. October 1984 is before May 22, 1995. So the answer is: no.",
                "\nQuestion: Is Farkas was appointed military attache at the Hungarian embassy before he was made a brigadier?\nThe step-by-step explanation and answer is: Farkas was made a brigadier general on May 22, 1995. He was appointed military attache at the Hungarian embassy in the following year. So the answer is: no."
                ]
            assert args.num_example == 3
            fs_examplrs_questions = ''.join(fs_examplrs_question_list[:args.num_example])
            fs_examplrs+=fs_examplrs_questions
            fs_examplrs+='\n'
        else:
            prompt_format="Answer the question according to the article. Only answer yes or no. \nArticle: {events}. \nQuestion: {question}. \nThe answer is: "+label
            # # 330 tokens in fs_examplrs
            # fs_examplrs="Answer the question according to the article. Only answer yes or no. \nArticle: Lawrence Insurance Group Inc. said it acquired United Republic Reinsurance Co., a Houston property and casualty reinsurance company, from United Savings Association of Texas for $28 million. Lawrence Insurance also sold 3.2 million of its shares for $7.125 each to its parent, Lawrence Group Inc. Lawrence Insurance, based in Albany, N.Y., plans to use the $22.5 million in proceeds to help finance the acquisition of United Republic. By acquiring the shares, Lawrence Group increased its stake in Lawrence Insurance to 93.2% from 91.2%. Lawrence Insurance underwrites mostly primary insurance, a company spokesman said. A reinsurance company effectively insures insurance companies that wish to spread the risk of a particular policy. Lawrence Group also owns Lawrence Agency Corp., Schenectady, N.Y., an insurance agency and brokerage. \nQuestion: Is acquiring the shares before Lawrence Group increased its stake? \nThe answer is: yes.\nQuestion: Is Lawrence Insurance Group Inc. said anything before acquired United Republic Reinsurance Co.? \nThe answer is: no.\nQuestion: Is the company spokesman said Lawrence Insurance underwrites mostly primary insurance before 26th October 1986? \nThe answer is: yes.\n"
            fs_examplrs="Answer the question according to the article. Only answer yes or no. \nArticle: Tired of being sidelined, Hungarian astronaut Bertalan Farkas is leaving for the United States to start a new career, he said Saturday. \"Being 48 is too early to be retired,\" a fit-looking Farkas said on state TV's morning talk show. With American astronaut Jon McBride, Farkas set up an American-Hungarian joint venture called Orion 1980, manufacturing space-travel related technology. Farkas will move to the company's U.S. headquarters. Farkas, an air force captain, was sent into space on board the Soyuz 36 on May 26, 1980. He spent six days aboard the Salyut 6 spacecraft with three Soviet astronauts, Valery Kubasov, Leonid Popov and Valery Riumin. McBride, 54, of Lewisburg, West Virginia, was part of a seven-member crew aboard the Orbiter Challenger in October 1984 and later served as assistant administrator for congressional relations for NASA. Farkas expressed the hope he one day follow in the footsteps of fellow astronaut John Glenn, who at 77 is about to go into space again. On May 22, 1995, Farkas was made a brigadier general, and the following year he was appointed military attache at the Hungarian embassy in Washington. However, cited by District of Columbia traffic police in December for driving under the influence of alcohol, Farkas was ordered home and retired."
            fs_examplrs_question_list=[
                "\nQuestion: Is Farkas sent into space on board the Soyuz before McBride on board the Orbiter Challenger?\nThe answer is: yes.",
                "\nQuestion: Is McBride on board the Orbiter Challenger after Farkas was made a brigadier general?\nThe answer is: no.",
                "\nQuestion: Is Farkas was appointed military attache at the Hungarian embassy before he was made a brigadier?\nThe answer is: no.",
                "\nQuestion: Is Farkas was appointed military attache at the Hungarian embassy before he was cited by District of Columbia traffic police? \nThe answer is: yes.",
                "\nQuestion: Is Farkas was cited by District of Columbia traffic police before he said he is leaving for the United States?\nThe answer is: yes.",
                "\nQuestion: Is Farkas is leaving for the United States after he said Saturday? \nThe answer is: yes.",
                "\nQuestion: Is McBride part of a seven-member crew aboard the Orbiter Challenger before he served as assistant administrator for NASA?\nThe answer is: yes.",
                "\nQuestion: Is Farkas cited by traffic police in December before he driving under the influence of alcohol?\nThe answer is: no.",
                "\nQuestion: Is Farkas cited by traffic police in December before he was ordered  home?\nThe answer is: yes.",
                "\nQuestion: Is Farkas said on state TV's morning talk show before 18th April 1998?\nThe answer is: yes."
                ]
            fs_examplrs_questions = ''.join(fs_examplrs_question_list[:args.num_example])
            fs_examplrs+=fs_examplrs_questions
            fs_examplrs+='\n'
    elif args.prompt_template == 2:
        prompt_format="Based on the information presented in the article \"{events}\", answer the question \"{question}\" with yes or no. The answer is: "+label
        
        fs_examplrs="Based on the information presented in the article \"Lawrence Insurance Group Inc. said it acquired United Republic Reinsurance Co., a Houston property and casualty reinsurance company, from United Savings Association of Texas for $28 million. Lawrence Insurance also sold 3.2 million of its shares for $7.125 each to its parent, Lawrence Group Inc. Lawrence Insurance, based in Albany, N.Y., plans to use the $22.5 million in proceeds to help finance the acquisition of United Republic. By acquiring the shares, Lawrence Group increased its stake in Lawrence Insurance to 93.2% from 91.2%. Lawrence Insurance underwrites mostly primary insurance, a company spokesman said. A reinsurance company effectively insures insurance companies that wish to spread the risk of a particular policy. Lawrence Group also owns Lawrence Agency Corp., Schenectady, N.Y., an insurance agency and brokerage.\", answer the question \"Is acquiring the shares before Lawrence Group increased its stake?\" with yes or no. The answer is: yes.\nBased on the information presented in the previous article, answer the question \"Is Lawrence Insurance Group Inc. said anything before acquired United Republic Reinsurance Co.?\" with yes or no. The answer is: no.\nBased on the information presented in the previous article, answer the question \"Is the company spokesman said Lawrence Insurance underwrites mostly primary insurance before 26th October 1986?\" with yes or no. The answer is: yes.\n"
    elif args.prompt_template == 3:
        prompt_format="Article: \"{events}\"\nThe answer to the yes or no question \"{question}\" according to the the article is: "+label
        
        fs_examplrs="""Finish the following text:
Article: \"Lawrence Insurance Group Inc. said it acquired United Republic Reinsurance Co., a Houston property and casualty reinsurance company, from United Savings Association of Texas for $28 million. Lawrence Insurance also sold 3.2 million of its shares for $7.125 each to its parent, Lawrence Group Inc. Lawrence Insurance, based in Albany, N.Y., plans to use the $22.5 million in proceeds to help finance the acquisition of United Republic. By acquiring the shares, Lawrence Group increased its stake in Lawrence Insurance to 93.2% from 91.2%. Lawrence Insurance underwrites mostly primary insurance, a company spokesman said. A reinsurance company effectively insures insurance companies that wish to spread the risk of a particular policy. Lawrence Group also owns Lawrence Agency Corp., Schenectady, N.Y., an insurance agency and brokerage.\"
The answer to the yes or no question \"Is acquiring the shares before Lawrence Group increased its stake?\" according to the article is yes.
###
The answer to the yes or no question \"Is Lawrence Insurance Group Inc. said anything before acquired United Republic Reinsurance Co.?\" according to the article is: no.
###
The answer to the yes or no question \"Is the company spokesman said Lawrence Insurance underwrites mostly primary insurance before 26th October 1986?\" according to the article is yes.
###\n"""
    

    assert len(questions) == len(events)
    prompts = []
    for q, e in zip(questions, events):
        e = tokenizer.encode(e)
        if len(e) >= args.max_events_length:
            e = e[1:args.max_events_length-1] # Skip </s> at the begining and end of seq
        else:
            e = e[1:-1]
        e = tokenizer.decode(e)
        p=fs_examplrs+prompt_format.format(events=e, question=q)
        prompts.append(p)
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

domains=['blogs', 'news', 'wikipedia']

if args.temp_eval:
    print("Evaluating "+args.model_name+" on TempEvalQA...")
if args.bidirectional_temp_eval_likelihood or args.bidirectional_temp_eval_decoding:
    print("Evaluating "+args.model_name+" on Bidirectional-TempEvalQA...")
if args.do_in_context_learning:
    print("Prompting Strategy: In-context Learning.")
else:
    print("Prompting Strategy: Zero-shot direct prompting.")

if args.temp_eval:
    for domain in domains:
        dataset = pd.read_csv("./dataset/tempeval-bi/qa-tempeval-test-"+domain+".csv")
        
        if args.do_in_context_learning:
            yes_prompts = make_icl_prompt(dataset, tokenizer, 'yes')
            no_prompts = make_icl_prompt(dataset, tokenizer, 'no')
        else:
            yes_prompts = make_zs_prompt(dataset, tokenizer, 'yes')
            no_prompts = make_zs_prompt(dataset, tokenizer, 'no')

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
            assert yes_d[0] =='yes'
            assert no_d[0] == 'no'
            if yes_d[1] >= no_d[1]:
                answers.append('yes\n')
            else:
                answers.append('no\n')
        
        with open(args.output_path+"/tempeval-"+domain+"-"+args.model_name+".out", "w+") as f:
            f.writelines(answers)

if args.bidirectional_temp_eval_likelihood:
    for domain in domains:
        for bi_direction in ["original", "negative"]:
            dataset = pd.read_csv("./dataset/tempeval-bi/bidirectional-"+bi_direction+"-qa-tempeval-test-"+domain+".csv")
            
            if args.do_in_context_learning:
                yes_prompts = make_icl_prompt(dataset, tokenizer, 'yes')
                no_prompts = make_icl_prompt(dataset, tokenizer, 'no')
            else:
                yes_prompts = make_zs_prompt(dataset, tokenizer, 'yes')
                no_prompts = make_zs_prompt(dataset, tokenizer, 'no')

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
                assert yes_d[0].strip() =='yes'
                assert no_d[0].strip() == 'no'
                if yes_d[1] >= no_d[1]:
                    answers.append('yes\n')
                else:
                    answers.append('no\n')

            with open(args.output_path+"/tempeval-"+domain+"-"+args.model_name+"-"+bi_direction+".out", "w+") as f:
                f.writelines(answers)

if args.bidirectional_temp_eval_decoding:
    for domain in domains:
        for bi_direction in ["original", "negative"]:
            dataset = pd.read_csv(args.data_path+"/bidirectional-"+bi_direction+"-qa-tempeval-test-"+domain+".csv")
            
            if args.do_in_context_learning:
                prompts = make_icl_prompt(dataset, tokenizer, '') # note label is delivberately left blank since we are using decoding strategy
            else:
                prompts = make_zs_prompt(dataset, tokenizer, '') # note label is delivberately left blank since we are using decoding strategy

            answers = []
            for i in tqdm(range(0, len(prompts), args.max_batch_size)):
                with torch.no_grad():
                    batch_p = prompts[i:i+args.max_batch_size]
                    inputs = tokenizer(batch_p, return_tensors="pt", padding=False).to(device) # batch decoding
                    generate_ids = model.generate(**inputs, max_new_tokens=args.max_new_decoding_tokens, temperature=args.temperature, top_p=args.top_p, do_sample=args.do_sample)
                    generate_ids=[ids[len(inputs['input_ids'][i]):] for i, ids in enumerate(generate_ids)] # truncate to get the raw output.
                    outputs=tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    if args.do_chain_of_thought: 
                        answers+=[output.replace('\n', '\t')+'\n' for output in outputs]
                        # for output in outputs:
                        #     print(output)
                        #     print('------------------')
                            # try:
                            #     answers.append(output.lower().split('answer:')[1].split()[0].strip('.')+'\n')
                            # except IndexError:
                            #     answers.append(''+'\n')
                                
                                
                    else:
                        answers+=[output.replace('\n', '')+'\n' for output in outputs]

            with open(args.output_path+"/tempeval-"+domain+"-"+args.model_name+"-"+bi_direction+".out", "w+") as f:
                f.writelines(answers)

