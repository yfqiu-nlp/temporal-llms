import pandas as pd
import openai
from tqdm import tqdm
from transformers import AutoTokenizer
import time

import argparse

parser = argparse.ArgumentParser(description='GPT-TempEval')

parser.add_argument('--model_name', type=str, help='Name of GPT model used to test.')
parser.add_argument('--output_path', type=str, help='Path for GPT model ouptuts.')

parser.add_argument('--temperature', type=float, default=0, help='Decoding temperature.')
parser.add_argument('--top_p', type=float, default=1.0, help='Decoding top-p.')
parser.add_argument('--max_events_length', type=int, default=1900, help='Max input article length.')
parser.add_argument('--max_batch_size', type=int, default=32, help='Max batch size.')
parser.add_argument('--max_new_decoding_tokens', type=int, default=2, help='Max new tokens for decoding output.')
parser.add_argument('--num_example', type=int, default=3, help='number of demostrations.')

parser.add_argument("--temp_eval", action= "store_true", help = "Do temp_eval original experiment")
parser.add_argument("--bidirectional_temp_eval", action= "store_true", help = "Do temp_eval bidirectional experiment")

parser.add_argument("--do_in_context_learning", action= "store_true", help = "Do ICL prompting.")
parser.add_argument("--do_chain_of_thought", action= "store_true", help = "Do ICL prompting.")

args = parser.parse_args()

openai_apikey=""
openai.api_key = openai_apikey

tokenizer = AutoTokenizer.from_pretrained("gpt2")

domains=['blogs', 'news', 'wikipedia']

def make_zs_prompt(dataset, tokenizer):
    questions=[q[0] for q in dataset[['question']].values]
    events=[e[0] for e in dataset[['events']].values]
    prompt_format="Answer the question according to the article. Only answer yes or no. \nArticle: {events}. \nQuestion: {question}. \nThe answer is:"
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

def make_icl_prompt(dataset, tokenizer):
    questions=[q[0] for q in dataset[['question']].values]
    events=[e[0] for e in dataset[['events']].values]
    
    if args.do_chain_of_thought:
        # # 3 documents for few-shot ICL prompting
        # prompt_format="Answer the question according to the article. Give step-by-step explanations and then answer yes or no. \nArticle: {events}. \nQuestion: {question}. \nThe step-by-step explanation and answer is: "
        # fs_examplrs="Answer the question according to the article. Give step-by-step explanations and then answer yes or no. \nArticle: Lawrence Insurance Group Inc. said it acquired United Republic Reinsurance Co., a Houston property and casualty reinsurance company, from United Savings Association of Texas for $28 million. Lawrence Insurance also sold 3.2 million of its shares for $7.125 each to its parent, Lawrence Group Inc. Lawrence Insurance, based in Albany, N.Y., plans to use the $22.5 million in proceeds to help finance the acquisition of United Republic. By acquiring the shares, Lawrence Group increased its stake in Lawrence Insurance to 93.2% from 91.2%. Lawrence Insurance underwrites mostly primary insurance, a company spokesman said. A reinsurance company effectively insures insurance companies that wish to spread the risk of a particular policy. Lawrence Group also owns Lawrence Agency Corp., Schenectady, N.Y., an insurance agency and brokerage."
        
        # fs_examplrs_question_list=[
        #         "\nQuestion: Is acquiring the shares before Lawrence Group increased its stake?\nThe step-by-step explanation and answer is: Lawrence Group increased its stake because of acquiring the shares. So the answer is yes.",
        #         "\nQuestion: Is Lawrence Insurance Group Inc. said anything before acquired United Republic Reinsurance Co.?\nThe step-by-step explanation and answer is: Lawrence Insurance Group Inc. said they acquired United Republic Reinsurance Co. So the answer is no.",
        #         "\nQuestion: Is the company spokesman said Lawrence Insurance underwrites mostly primary insuance before 26th October 1986?\nThe step-by-step explanation and answer is: the news that company spokesman said Lawrence Insurance underwrites mostly primary insuance is written on 26th October 1986. So the answer is yes."
        #         ]

        # # 3 document for COT prompting
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
        prompt_format="Answer the question according to the article. Only answer yes or no. \nArticle: {events}. \nQuestion: {question}. \nThe answer is:"

        # 330 tokens in fs_examplrs
        fs_examplrs="Answer the question according to the article. Only answer yes or no. \nArticle: Lawrence Insurance Group Inc. said it acquired United Republic Reinsurance Co., a Houston property and casualty reinsurance company, from United Savings Association of Texas for $28 million. Lawrence Insurance also sold 3.2 million of its shares for $7.125 each to its parent, Lawrence Group Inc. Lawrence Insurance, based in Albany, N.Y., plans to use the $22.5 million in proceeds to help finance the acquisition of United Republic. By acquiring the shares, Lawrence Group increased its stake in Lawrence Insurance to 93.2% from 91.2%. Lawrence Insurance underwrites mostly primary insurance, a company spokesman said. A reinsurance company effectively insures insurance companies that wish to spread the risk of a particular policy. Lawrence Group also owns Lawrence Agency Corp., Schenectady, N.Y., an insurance agency and brokerage. \nQuestion: Is acquiring the shares before Lawrence Group increased its stake? \nThe answer is: Yes.\nQuestion: Is Lawrence Insurance Group Inc. said anything before acquired United Republic Reinsurance Co.? \nThe answer is: No.\nQuestion: Is the company spokesman said Lawrence Insurance underwrites mostly primary insuance before 26th October 1986? \nThe answer is: Yes.\n"
    
    assert len(questions) == len(events)
    prompts = []
    for q, e in zip(questions, events):
        e = tokenizer.encode(e)
        if len(e) >= args.max_events_length:
            print("Length of ",len(e)," is truncated to ",args.max_events_length)
            e = e[1:args.max_events_length-1] # Skip </s> at the begining and end of seq
        else:
            e = e[1:-1]
        e = tokenizer.decode(e)
        p=fs_examplrs+prompt_format.format(events=e, question=q)
        prompts.append(p)
    return prompts

if args.temp_eval:
    print("Evaluating "+args.model_name+" on TempEvalQA...")
if args.bidirectional_temp_eval:
    print("Evaluating "+args.model_name+" on Bidirectional-TempEvalQA...")
if args.do_in_context_learning:
    print("Prompting Strategy: In-context Learning.")
else:
    print("Prompting Strategy: Zero-shot direct prompting.")

if args.temp_eval:
    for domain in domains:
        dataset = pd.read_csv("./dataset/tempeval-bi/qa-tempeval-test-"+domain+".csv")
        
        if args.do_in_context_learning:
            prompts = make_icl_prompt(dataset, tokenizer)
        else:
            prompts = make_zs_prompt(dataset, tokenizer)
        
        answers = []
        for p in tqdm(prompts):
            output = openai.Completion.create(
                    model=args.model_name,
                    prompt=p,
                    temperature=args.temperature,
                    max_tokens=args.max_new_decoding_tokens,
                    top_p=args.top_p,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
            answers.append(output['choices'][0]['text'].replace('\n','')+'\n')

        with open("tempeval-"+domain+"-"+args.model_name+".out", "w+") as f:
            f.writelines(answers)

if args.bidirectional_temp_eval:
    for domain in domains:
        for bi_direction in ["original", "negative"]:
            dataset = pd.read_csv("./dataset/tempeval-bi/bidirectional-"+bi_direction+"-qa-tempeval-test-"+domain+".csv")
            
            if args.do_in_context_learning:
                prompts = make_icl_prompt(dataset, tokenizer)
            else:
                prompts = make_zs_prompt(dataset, tokenizer)

            # tok_num = 0
            # for p in prompts:
            #     tok_num+=len(tokenizer.encode(p))
            # print(tok_num)
            # 1/0

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
                    time.sleep(3)
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
                    time.sleep(3)
                    answers.append(output['choices'][0]['text'].replace('\n','')+'\n')
            with open(args.output_path+"/tempeval-"+domain+"-"+args.model_name+"-"+bi_direction+".out", "w+") as f:
                f.writelines(answers)
