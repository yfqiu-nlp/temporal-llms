#!/bin/bash

conda activate llama-hf

# specify model name
# model_name="davinci"
# model_name="text-davinci-003"
# model_name="text-davinci-002"
model_name="gpt-4"

# prompt_style="qa"
prompt_style="mcq"

# # ### Run mctaco with Zero-shot Prompting
python3 mctaco-gpt.py \
    --model_name $model_name \
    --output_path "gpt-output/mctaco-"$prompt_style"-zs-output" \
    --temperature 0 \
    --top_p 1.0 \
    --max_events_length 1900 \
    --max_new_decoding_tokens 0 \
    --mctaco_eval \
    --max_batch_size 1 \
    --prompt_style $prompt_style \

### Run mctaco with ICL
python3 mctaco-gpt.py \
    --model_name $model_name \
    --output_path "gpt-output/mctaco-"$prompt_style"-fs-output" \
    --temperature 0 \
    --top_p 1.0 \
    --max_events_length 1600 \
    --max_new_decoding_tokens 0 \
    --mctaco_eval \
    --do_in_context_learning \
    --max_batch_size 1 \
    --prompt_style $prompt_style \
    