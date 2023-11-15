import json
import pickle
import sys
import os
import glob
import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor
import random
import torch
from copy import deepcopy

import argparse

from allennlp.common.checks import check_for_gpu
from allennlp.common.util import import_module_and_submodules
from allennlp.models.archival import load_archive

from transformers import AutoTokenizer, AutoModelWithLMHead

sys.path.append(os.getcwd())
from denoising_event_lm.models.event_lm.seq2seq import get_flatten_varg_toks

if __name__ == '__main__':
    random.seed(765)
    np.random.seed(765)
    torch.manual_seed(765)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(765)


def text_transformer_normzlize(text, tokenizer):
    return tokenizer.decode(tokenizer.encode_plus(text)['input_ids'], skip_special_tokens=True)

class GPT2baseline:
    def __init__(self, model_name, cuda_device):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)   # Initialize tokenizer
        self._model = AutoModelWithLMHead.from_pretrained(model_name)    # Download model and configuration from S3 and cache.
        self._model.to(cuda_device)

    def predict(self, hotpot_dict_instance):
        example = hotpot_dict_instance['example'] # dict
        replace_idx = hotpot_dict_instance.get('replace_idx', None) # int
        insert_idx = hotpot_dict_instance.get('insert_idx', None) # int, if gold: A B C D, insert_idx: 2 => A B E C D
        force_suffix = hotpot_dict_instance['force_suffix'] # bool
        override_decode_kwargs = hotpot_dict_instance['override_decode_kwargs'] # dict

        varg_seq = example['varg_seq']
        permutation = example.get('permutation') # indices to the true varg_seq
        aug_metadata = example.get('aug_metadata', None)

        # get the gold chain
        if aug_metadata is not None and 'source_chain' in aug_metadata:
            gold_varg_seq = aug_metadata['source_chain']
        elif permutation is not None:
            order = sorted(list(range(len(varg_seq))), key=lambda x: permutation[x])
            gold_varg_seq = [varg_seq[i] for i in order]

        for varg in gold_varg_seq:
            if not 'Description' in varg:
                varg['Description'] = " ".join(get_flatten_varg_toks(varg))

        # get the input varg seq
        if replace_idx is not None:
            input_varg_seq = [gold_varg_seq[i] for i in range(len(gold_varg_seq)) if not i == replace_idx]
            insert_idx = replace_idx
            num_output_events = len(gold_varg_seq)
        elif insert_idx is not None:
            input_varg_seq = gold_varg_seq
            num_output_events = len(gold_varg_seq) + 1

        suffix_varg_seq = [input_varg_seq[i] for i in range(len(input_varg_seq)) if i >= insert_idx]
        input_varg_seq = [input_varg_seq[i] for i in range(len(input_varg_seq)) if i < insert_idx]

        source_str = " . ".join([varg['Description'] for varg in input_varg_seq])
        if len(source_str) > 0:
            source_str += ' . '

        decode_suffix_str = " . ".join([varg['Description'] for varg in suffix_varg_seq])
        if len(decode_suffix_str) > 0:
            decode_suffix_str += ' . '

        # encode input context
        if len(source_str) > 0:
            input_ids = self._tokenizer.encode(source_str, return_tensors='pt').to(self._model.device)
        else:
            input_ids = None

        # generate sequences
        decode_kwargs = {"max_length": 375,
                         "early_stopping": True,
                         "no_repeat_ngram_size": 3}
        decode_kwargs.update(override_decode_kwargs)
        decode_kwargs['max_length'] = input_ids.shape[-1] + 50 if input_ids is not None else 50
        # (num_return_sequences, length)
        decode_prediction_ids = self._model.generate(input_ids=input_ids, **decode_kwargs)

        output_dict = {
            "source_str": self._tokenizer.decode(input_ids[0]) if input_ids is not None else None,
            "target_str": None,
            "input_varg_seq": input_varg_seq,
            "gold_varg_seq": gold_varg_seq
        }

        #decode_prediction_ids = decode_prediction_ids.view(decode_kwargs.get("num_return_sequences", 1),
        #                                                   decode_prediction_ids.size(-1))
        output_dict["decode_prediction_ids"] = decode_prediction_ids.detach().cpu().numpy()

        prediction_ids = decode_prediction_ids

        output_dict['beam_prediction_str'] = []
        output_dict['beam_prediction_varg_seq'] = []

        if len(input_varg_seq) > 0:
            decode_prefix_str = ". ".join([varg['Description'] for varg in input_varg_seq]) + '.'
            decode_prefix_str = text_transformer_normzlize(decode_prefix_str, self._tokenizer)
        else:
            decode_prefix_str = ""

        beam_size = prediction_ids.size(0)
        for beam_idx in range(beam_size):
            predicted_token_ids = prediction_ids[beam_idx].detach().cpu().numpy()
            prediction_str = self._tokenizer.decode(predicted_token_ids, skip_special_tokens=True).strip()

            assert prediction_str.startswith(decode_prefix_str), '\n' + repr(decode_prefix_str) + '\n' + repr(prediction_str)
            next_event = prediction_str[len(decode_prefix_str):]
            next_event_end = next_event.find('.')
            if next_event_end >= 0:
                next_event = next_event[:next_event_end+1]
                next_event = next_event.replace("\n", " ")
            prediction_str = prediction_str[:len(decode_prefix_str)] + next_event
            #if not prediction_str[-1] in ['.', '?', '!', ')', ']', '}', ',', ';', ':', '\'', '"']:
            #    prediction_str += '. '
            prediction_str += ' ' + decode_suffix_str
            output_dict['beam_prediction_str'].append(prediction_str)
            if next_event[-1] == '.':
                next_event = next_event[:-1]
            output_dict['beam_prediction_varg_seq'].append(input_varg_seq+[{'Description': next_event}]+suffix_varg_seq)

        predicted_token_ids = prediction_ids[0].detach().cpu().numpy()
        prediction_str = self._tokenizer.decode(predicted_token_ids, skip_special_tokens=True).strip()
        assert prediction_str.startswith(decode_prefix_str), '\n' + repr(decode_prefix_str) + '\n' + repr(prediction_str)
        next_event = prediction_str[len(decode_prefix_str):]
        next_event_end = next_event.find('.')
        if next_event_end >= 0:
            next_event = next_event[:next_event_end+1]
            next_event = next_event.replace("\n", " ")
        prediction_str = prediction_str[:len(decode_prefix_str)] + next_event
        #if not prediction_str[-1] in ['.', '?', '!', ')', ']', '}', ',', ';', ':', '\'', '"']:
        #    prediction_str += '. '
        prediction_str += ' ' + decode_suffix_str
        output_dict['prediction_str'] = prediction_str
        if next_event[-1] == '.':
            next_event = next_event[:-1]
        output_dict['prediction_varg_seq'] = input_varg_seq+[{'Description': next_event}]+suffix_varg_seq

        '''
        if i == 0:
            print("prediction_ids")
            print(predicted_token_ids)
            print("prediction_str raw")
            print(self._tokenizer.decode(predicted_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True))
            print("prediction_str")
            print(prediction_str)
            print("prediction varg seq")
            print(json.dumps(prediction_varg_seq, indent=2))
            print("gold varg seq")
            print(json.dumps(gold_varg_seq, indent=2))
        input()
        '''

        return output_dict


class InfillingGPT2baseline(GPT2baseline):
    def predict(self, hotpot_dict_instance):
        example = hotpot_dict_instance['example'] # dict
        replace_idx = hotpot_dict_instance.get('replace_idx', None) # int
        insert_idx = hotpot_dict_instance.get('insert_idx', None) # int, if gold: A B C D, insert_idx: 2 => A B E C D
        force_suffix = hotpot_dict_instance['force_suffix'] # bool
        override_decode_kwargs = hotpot_dict_instance['override_decode_kwargs'] # dict

        varg_seq = example['varg_seq']
        permutation = example.get('permutation') # indices to the true varg_seq
        aug_metadata = example.get('aug_metadata', None)

        # get the gold chain
        if aug_metadata is not None and 'source_chain' in aug_metadata:
            gold_varg_seq = aug_metadata['source_chain']
        elif permutation is not None:
            order = sorted(list(range(len(varg_seq))), key=lambda x: permutation[x])
            gold_varg_seq = [varg_seq[i] for i in order]

        for varg in gold_varg_seq:
            if not 'Description' in varg:
                varg['Description'] = " ".join(get_flatten_varg_toks(varg))

        # get the input varg seq
        if replace_idx is not None:
            input_varg_seq = [gold_varg_seq[i] for i in range(len(gold_varg_seq)) if not i == replace_idx]
            insert_idx = replace_idx
            num_output_events = len(gold_varg_seq)
        elif insert_idx is not None:
            input_varg_seq = gold_varg_seq
            num_output_events = len(gold_varg_seq) + 1

        suffix_varg_seq = [input_varg_seq[i] for i in range(len(input_varg_seq)) if i >= insert_idx]
        input_varg_seq = [input_varg_seq[i] for i in range(len(input_varg_seq)) if i < insert_idx]

        # TODO
        suffix_str = " . ".join([varg['Description'] for varg in suffix_varg_seq])
        assert len(suffix_str) > 0
        suffix_str += ' . '

        input_str = " . ".join([varg['Description'] for varg in input_varg_seq])
        if len(input_str) > 0:
            input_str += ' . '

        source_str = suffix_str + self._tokenizer.eos_token + input_str

        # encode input context
        if len(source_str) > 0:
            input_ids = self._tokenizer.encode(source_str, return_tensors='pt').to(self._model.device)
        else:
            input_ids = None

        # generate sequences
        decode_kwargs = {"max_length": 375,
                         "early_stopping": True,
                         "no_repeat_ngram_size": 3}
        decode_kwargs.update(override_decode_kwargs)
        decode_kwargs['max_length'] = input_ids.shape[-1] + 50 if input_ids is not None else 50
        # (num_return_sequences, length)
        decode_prediction_ids = self._model.generate(input_ids=input_ids, **decode_kwargs)

        output_dict = {
            "source_str": self._tokenizer.decode(input_ids[0]) if input_ids is not None else None,
            "target_str": None,
            "input_varg_seq": suffix_varg_seq + input_varg_seq,
            "gold_varg_seq": gold_varg_seq
        }

        #decode_prediction_ids = decode_prediction_ids.view(decode_kwargs.get("num_return_sequences", 1),
        #                                                   decode_prediction_ids.size(-1))
        output_dict["decode_prediction_ids"] = decode_prediction_ids.detach().cpu().numpy()

        prediction_ids = decode_prediction_ids

        output_dict['beam_prediction_str'] = []
        output_dict['beam_prediction_varg_seq'] = []

        decode_prefix_str = ". ".join([varg['Description'] for varg in suffix_varg_seq])
        assert len(decode_prefix_str) > 0
        decode_prefix_str += '. ' + self._tokenizer.eos_token

        if len(input_varg_seq) > 0:
            decode_prefix_str += ". ".join([varg['Description'] for varg in input_varg_seq]) + '.'
        decode_prefix_str = text_transformer_normzlize(decode_prefix_str, self._tokenizer)

        beam_size = prediction_ids.size(0)
        for beam_idx in range(beam_size):
            predicted_token_ids = prediction_ids[beam_idx].detach().cpu().numpy()
            prediction_str = self._tokenizer.decode(predicted_token_ids, skip_special_tokens=True).strip()

            assert prediction_str.startswith(decode_prefix_str), '\n' + repr(decode_prefix_str) + '\n' + repr(prediction_str)
            next_event = prediction_str[len(decode_prefix_str):]
            next_event_end = next_event.find('.')
            if next_event_end >= 0:
                next_event = next_event[:next_event_end+1]
                next_event = next_event.replace("\n", " ")
            prediction_str = input_str + next_event
            #if not prediction_str[-1] in ['.', '?', '!', ')', ']', '}', ',', ';', ':', '\'', '"']:
            #    prediction_str += '. '
            prediction_str += ' ' + suffix_str
            output_dict['beam_prediction_str'].append(prediction_str)
            if next_event[-1] == '.':
                next_event = next_event[:-1]
            output_dict['beam_prediction_varg_seq'].append(input_varg_seq+[{'Description': next_event}]+suffix_varg_seq)

        predicted_token_ids = prediction_ids[0].detach().cpu().numpy()
        prediction_str = self._tokenizer.decode(predicted_token_ids, skip_special_tokens=True).strip()
        assert prediction_str.startswith(decode_prefix_str), '\n' + repr(decode_prefix_str) + '\n' + repr(prediction_str)
        next_event = prediction_str[len(decode_prefix_str):]
        next_event_end = next_event.find('.')
        if next_event_end >= 0:
            next_event = next_event[:next_event_end+1]
            next_event = next_event.replace("\n", " ")
        prediction_str = input_str + next_event
        #if not prediction_str[-1] in ['.', '?', '!', ')', ']', '}', ',', ';', ':', '\'', '"']:
        #    prediction_str += '. '
        prediction_str += ' ' + suffix_str
        output_dict['prediction_str'] = prediction_str
        if next_event[-1] == '.':
            next_event = next_event[:-1]
        output_dict['prediction_varg_seq'] = input_varg_seq+[{'Description': next_event}]+suffix_varg_seq

        '''
        if i == 0:
            print("prediction_ids")
            print(predicted_token_ids)
            print("prediction_str raw")
            print(self._tokenizer.decode(predicted_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True))
            print("prediction_str")
            print(prediction_str)
            print("prediction varg seq")
            print(json.dumps(prediction_varg_seq, indent=2))
            print("gold varg seq")
            print(json.dumps(gold_varg_seq, indent=2))
        input()
        '''

        return output_dict


def chain_str(chain, insert_idx=None):
    texts = []
    for i, varg in enumerate(chain):
        #texts.append("<EVENT> " + " ".join(varg['V_toks']) + " <ARGS> " + varg['Description'])
        if insert_idx is not None and i == insert_idx:
            texts.append('[['+varg['Description']+' .]] ')
        else:
            texts.append(varg['Description']+' . ')
    return texts


def sample_unseen_events(data, insert_idx, predictor, print_result=True, file=sys.stdout):
    output = predictor.predict(data)
    if print_result:
        print('---'*20, file=file)
        print("Gold Chain:", file=file)
        print("\n".join(chain_str(output['gold_varg_seq'])), file=file)
        print('---'*20, file=file)
        print("Input Events:", file=file)
        print("\n".join(chain_str(output['input_varg_seq'])), file=file)
        print(file=file)
        print('---'*20, file=file)
        print(file=file)
        print("Predictions:", file=file)
        for b_idx, pred_varg_seq in enumerate(output['beam_prediction_varg_seq']):
            print("Beam {:d}".format(b_idx), file=file)
            #print("\n".join(chain_str(pred_varg_seq)), file=file)
            print("".join(chain_str(pred_varg_seq, insert_idx=insert_idx)), file=file)
            print(file=file)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test the predictor above')

    parser.add_argument('--archive-path', type=str, required=True, help='path to trained archive file')
    parser.add_argument('--predictor', type=str, required=True, help='name of predictor')
    parser.add_argument('--weights-file', type=str,
                        help='a path that overrides which weights file to use')
    parser.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')
    parser.add_argument('-o', '--overrides', type=str, default="",
                        help='a JSON structure used to override the experiment configuration')
    parser.add_argument('--include-package',
                        type=str,
                        action='append',
                        default=[],
                        help='additional packages to include')
    parser.add_argument('--input-path', type=str, nargs='+', help='input data')
    parser.add_argument('--insert-idx-path', type=str, help='list of insert idxs')
    parser.add_argument('--decode_kwargs', type=str, default='{}',
                        help='decode kwargs')
    parser.add_argument('--insert_mode', type=str, default='insert', choices=['insert', 'replace'],
                        help='whether to insert an event or replace an event')
    parser.add_argument('--force_suffix', action='store_true', default=False,
                        help='whether to make input events after the insertion point to be the decoding suffix')
    parser.add_argument('--num_instances', type=int, default=-1,
                        help='number of instances to process')
    parser.add_argument('--baseline', type=str, default=None, choices=["gpt2", "gpt2-medium", "gpt2-large"],
                        help='use baseline model instead')
    parser.add_argument('--output-path', type=str, default=None,
                        help='output path')

    args = parser.parse_args()

    args.decode_kwargs = json.loads(args.decode_kwargs)

    # Load modules
    for package_name in args.include_package:
        import_module_and_submodules(package_name)

    check_for_gpu(args.cuda_device)
    if args.baseline is None:
        archive = load_archive(args.archive_path,
                               weights_file=args.weights_file,
                               cuda_device=args.cuda_device,
                               overrides=args.overrides)

        predictor = Predictor.from_archive(archive, args.predictor)
    else:
        baselines = {
            'gpt2': GPT2baseline,
            'gpt2-medium': GPT2baseline,
            'gpt2-large': GPT2baseline,
            'infilling-gpt2': InfillingGPT2baseline,
            'infilling-gpt2-medium': InfillingGPT2baseline,
            'infilling-gpt2-large': InfillingGPT2baseline
        }
        predictor = baselines[args.baseline](args.baseline, args.cuda_device)

    if args.input_path is None:
        test_json = {
            'example': {
                'varg_seq': [
                    {
                        'V_toks': ['slipped'],
                        'Description': 'She slipped on a crack that was on one of the concrete tiles'
                    },
                    {
                        'V_toks': ['fell'],
                        'Description': 'She fell on the ground'
                    },
                    {
                        'V_toks': ['scattered'],
                        'Description': 'her belongings scattered'
                    }
                ],
                'permutation': (0, 1, 2),
                'aug_metadata': None
            },
            'replace_idx': 1 if args.insert_mode == 'replace' else None,
            'insert_idx': 1 if args.insert_mode == 'insert' else None,
            'force_suffix': args.force_suffix,
            'override_decode_kwargs': args.decode_kwargs
        }
        '''
        test_json = {
            'example': {
                'varg_seq': [
                    {
                        'V_toks': ['found'],
                        'Description': 'I found an old camera that I liked'
                    },
                    {
                        'V_toks': ['asked'],
                        'Description': 'I asked for the price'
                    },
                    {
                        'V_toks': ['asked'],
                        'Description': 'The seller asked for more rubles than what I had'
                    },
                    {
                        'V_toks': ['told'],
                        'Description': 'I told him that did n\'t have enough rubles , only Swedish krona'
                    }
                ],
                'permutation': (0, 1, 2, 3),
                'aug_metadata': None
            },
            'replace_idx': 2 if args.insert_mode == 'replace' else None,
            'insert_idx': 2 if args.insert_mode == 'insert' else None,
            'force_suffix': args.force_suffix,
            'override_decode_kwargs': args.decode_kwargs
        }
        '''
        sample_unseen_events(test_json, 1, predictor)
    else:
        data = []
        if args.insert_idx_path is not None:
            with open(args.insert_idx_path) as f:
                insert_idx_list = json.load(f)
        else:
            insert_idx_list = []
        for path_regex in args.input_path:
            for path in sorted(glob.glob(path_regex)):
                with open(path, 'rb') as f:
                    data += pickle.load(f)

        seen_ch_id = set()
        unique_data = []
        for d in data:
            ch_id = d['chain_id'] if not 'aug_metadata' in d or d['aug_metadata'] is None else d['aug_metadata']['source_chain_id']
            if not ch_id in seen_ch_id:
                unique_data.append(d)
                seen_ch_id.add(ch_id)
        data = unique_data

        random.shuffle(data)
        if args.num_instances > 0:
            data = data[:args.num_instances]
        results = []
        for d_idx, d in enumerate(tqdm(data)):
            d = deepcopy(d)
            if args.insert_idx_path is not None:
                insert_idx = insert_idx_list[d_idx]
            else:
                insert_idx = random.randint(0, len(d['varg_seq']) - 2)
                insert_idx_list.append(insert_idx)
            test_json = {
                "example": d,
                'replace_idx': insert_idx if args.insert_mode == 'replace' else None,
                'insert_idx': insert_idx if args.insert_mode == 'insert' else None,
                'force_suffix': args.force_suffix,
                'override_decode_kwargs': args.decode_kwargs
            }
            output = sample_unseen_events(test_json, insert_idx, predictor)
            print("\n\n")
            results.append(
                {
                    "gold_chain": output['gold_varg_seq'],
                    "input_events": output['input_varg_seq'],
                    "predictions": output['beam_prediction_varg_seq'],
                    "insert_idx": insert_idx
                }
            )
        if args.output_path is not None:
            with open(args.output_path, 'w') as f:
                json.dump(results, f, indent=2)
        if args.insert_idx_path is None:
            with open("./inster_idx_list.json", 'w') as f:
                json.dump(insert_idx_list, f, indent=2)
