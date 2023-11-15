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

prompt_style="qa"
# prompt_style="mcq"

prompt=1

# # # ### run mctaco
# ## Zero-shot experiment
for prompt in 1 2 3
do
python3 mctaco-llama-no-leakage.py \
    --model_name $model_name \
    --model_path $model_path \
    --output_path "llama-output/mctaco-"$prompt_style"-zs-pt${prompt}-output" \
    --temperature 0 \
    --top_p 1.0 \
    --max_events_length 3600 \
    --max_new_decoding_tokens 0 \
    --mctaco_eval \
    --max_batch_size 1 \
    --prompt_style $prompt_style \
    --prompt_template ${prompt}
done

# ## Few-shot ICL experiment
for prompt in 1 2 3
do
python3 mctaco-llama-no-leakage.py \
    --model_name $model_name \
    --model_path $model_path \
    --output_path "llama-output/mctaco-"$prompt_style"-fs-pt${prompt}-output" \
    --temperature 0 \
    --top_p 1.0 \
    --max_events_length 3600 \
    --max_new_decoding_tokens 0 \
    --mctaco_eval \
    --do_in_context_learning \
    --max_batch_size 1 \
    --prompt_style $prompt_style \
    --prompt_template ${prompt}
done
