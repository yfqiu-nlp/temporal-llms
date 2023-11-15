#!/bin/bash

conda activate llama-hf


model_path="/PATH/TO/YOUR/LLAMA/MODEL"

# model_name="llama-7b"
# model_name="llama-13b"
# model_name="llama-33b"
# model_name="llama-65b"
# model_name="llama-7b-alpaca"
# model_name="Llama-2-7b-hf"
# model_name="Llama-2-13b-hf"
# model_name="Llama-2-70b-hf"
# model_name="Llama-2-7b-chat-hf"
# model_name="Llama-2-13b-chat-hf"
model_name="Llama-2-70b-chat-hf"

num_examples=3

for prompt in 1 2 3
do
    # # # ### run mctaco with ICL
    python3 caters-llama.py \
        --model_name $model_name \
        --model_path $model_path \
        --output_path "llama-output/caters-fs-pt${prompt}-output-icl${num_examples}-del" \
        --temperature 0.8 \
        --top_p 0.95 \
        --do_sample \
        --max_events_length 3696 \
        --max_new_decoding_tokens 128 \
        --caters_eval \
        --do_in_context_learning \
        --max_batch_size 1 \
        --num_example ${num_examples} \
        --train_set ./dataset/caters/caters_train.csv \
        --prompt_template ${prompt}
done
    