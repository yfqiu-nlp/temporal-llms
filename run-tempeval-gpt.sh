#!/bin/bash

# specify model name
# model_name="davinci"
# model_name="text-davinci-003"
# model_name="text-davinci-002"
model_name="gpt-4"

# # # ### run bi-tempqa-qa with zero-shot prompting
python3 tempeval-gpt.py \
    --model_name $model_name \
    --output_path "gpt-output/zs-bi-output" \
    --temperature 0 \
    --top_p 1.0 \
    --max_events_length 1900 \
    --max_new_decoding_tokens 2 \
    --bidirectional_temp_eval \

# # # ### run bi-tempqa with zero-shot prompting
python3 tempeval-gpt.py \
    --model_name $model_name \
    --output_path "gpt-output/fs-bi-output" \
    --temperature 0 \
    --top_p 1.0 \
    --max_events_length 1600 \
    --max_new_decoding_tokens 2 \
    --bidirectional_temp_eval \
    --do_in_context_learning \

python3 eval-tempeval-bi.py "gpt-output/fs-bi-output" $model_name