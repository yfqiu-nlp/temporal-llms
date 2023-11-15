#!/bin/bash

conda activate llama-hf

# specify model name
# model_name="davinci"
# model_name="text-davinci-003"
# model_name="text-davinci-002"
model_name="gpt-4"

# # # ### run bi-tempqa with ICL
python3 caters-gpt.py \
    --model_name $model_name \
    --output_path "gpt-output/caters/caters-fs-output" \
    --temperature 0 \
    --top_p 1.0 \
    --max_new_decoding_tokens 128 \
    --caters_eval \
    --do_in_context_learning \
    --num_example 3 \
    --train_set ./dataset/caters/caters_train.csv \
    
