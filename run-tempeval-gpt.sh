#!/bin/bash

# specify model name
# model_name="davinci"
# model_name="text-davinci-003"
# model_name="text-davinci-002"
model_name="gpt-4"

# # # ### run bi-tempqa-qa with zero-shot prompting
python3 tempeval-gpt.py \
    --model_name $model_name \
    --output_path "gpt-output/tempeval-qa-bi/zero-shot" \
    --temperature 0 \
    --top_p 1.0 \
    --max_events_length 1900 \
    --max_new_decoding_tokens 2 \
    --bidirectional_temp_eval \

# # # ### run bi-tempqa with few-shot prompting
python3 tempeval-gpt.py \
    --model_name $model_name \
    --output_path "gpt-output/tempeval-qa-bi/few-shot-icl" \
    --temperature 0 \
    --top_p 1.0 \
    --max_events_length 1600 \
    --max_new_decoding_tokens 2 \
    --bidirectional_temp_eval \
    --do_in_context_learning \

# # # ### run bi-tempqa with ICL + COT
python3 tempeval-gpt.py \
    --model_name $model_name \
    --output_path "gpt-output/tempeval-qa-bi/cot-output" \
    --temperature 0 \
    --top_p 1.0 \
    --max_events_length 7000 \
    --max_new_decoding_tokens 128 \
    --bidirectional_temp_eval \
    --do_in_context_learning \
    --do_chain_of_thought \
