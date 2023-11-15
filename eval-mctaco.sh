#!/bin/bash

##### Activate the conda env
conda activate llama-hf

##### For GPT, we need to normalise the raw output into the target form for evaluation
##### Usage: python3 normalise-mctaco-gpt.py $OUTPUT_PATH $MODEL_NAME
##### Example:
# python3 normalise-mctaco-gpt.py gpt-output/mctaco/few-shot-icl/ gpt-4

##### Evaluating Script following McTACO paper, check the details in here: https://github.com/CogComp/MCTACO
python3 evaluator.py eval \
    --test_file ./dataset/mctaco/test_9442.tsv \
    --prediction_file llama-output/mctaco/mctaco-qa-fs-pt1-output/mctaco-test-Llama-2-70b-chat-hf.out \
    --output llama-output/mctaco/mctaco-qa-fs-pt1-output/mctaco-official-Llama-2-70b-chat-hf.json

