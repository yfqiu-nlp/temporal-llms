import json, pickle
import logging
from typing import Any, Dict, List, Tuple, Optional, Iterable
from copy import deepcopy
import numpy as np
import random

from allennlp.data.fields import MetadataField, ArrayField
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance

from transformers import PreTrainedTokenizer
from transformers.tokenization_auto import AutoTokenizer

from denoising_event_lm.data.dataset_readers.seq2seq.seq2seq_transformer_reader import Seq2SeqTransformerReader
from denoising_event_lm.data.data_utils.event_lm.utils import print_stat_with_posneg, print_stat_chainlen
from denoising_event_lm.utils.constants import EVENT_TAG, ARGS_TAG, POINTER_EVENT_TAGS
import pandas as pd

logger = logging.getLogger(__name__)


@DatasetReader.register("event_seq2seq_transformer_reader")
class EventSeq2SeqTransformerReader(Seq2SeqTransformerReader):
    """
    Reads a Pickle QA file and returns a ``Dataset`` where the ``Instances`` have four
    fields:
     * ``question_with_context``, a ``TextField`` that contains the concatenation of question and context,
     * ``answer_span``, a ``SpanField`` into the ``question`` ``TextField`` denoting the answer.
     * ``context_span`` a ``SpanField`` into the ``question`` ``TextField`` denoting the context, i.e., the part of
       the text that potential answers can come from.
     * A ``MetadataField`` that stores the instance's ID, the original question, the original passage text, both of
       these in tokenized form, and the gold answer strings, accessible as ``metadata['id']``,
       ``metadata['question']``, ``metadata['context']``, ``metadata['question_tokens']``,
       ``metadata['context_tokens']``, and ``metadata['answers']. This is so that we can more easily use the
       official SQuAD evaluation script to get metrics.
    Parameters
    ----------
    transformer_model_name : ``str``, optional (default=``bert-base-cased``)
        This reader chooses tokenizer and token indexer according to this setting.
    length_limit : ``int``, optional (default=self._tokenizer.model_max_length)
        We will make sure that the length of all input text never exceeds this many word pieces.
    truncation_strategy : `str`, optional (default=`'longest_first'`)
        String selected in the following options:
        - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
        starting from the longest one at each token (when there is a pair of input sequences)
        - 'only_first': Only truncate the first sequence
        - 'only_second': Only truncate the second sequence
        - 'do_not_truncate': Do not truncate (raise an error if the input sequence is longer than max_length)
    test_mode : ``bool``, optional (default=True)
        whether we are in the test mode.
    source_prefix : ``str``, optional (default="")
        the string to prepend on context. Mainly for T5 models.
    target_suffix : ``str``, optional (default="")
        the string to append on target. Mainly for T5 models.
    """

    def __init__(
        self,
        tokenizer_model_name: str,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        lowercase: bool = False,
        length_limit: Optional[int] = None,
        truncation_strategy: str = "longest_first",
        test_mode: bool = False,
        source_prefix: str = "",
        target_prefix: str = "",
        target_suffix: str = "",
        task_specific_args: Optional[Dict[str, Any]] = None,
        event_sep: str = EVENT_TAG,
        args_sep: str = ARGS_TAG,
        event_del_prob: float = 0.0,
        event_del_all_prob: float = 0.0,
        allow_empty_events: bool = False,
        event_keep_num: int = -1,
        source_format: str = "ARGS",
        target_format: str = "V_ARGS",
        seed: int = 2020,
        chain_len_min: int = None,
        chain_len_max: int = None,
        pos_chain_prefix: str = "",
        neg_chain_prefix: str = "<NEGATIVE>",
        neg_chain_target_type: str = "unrelated_events",
        neg_chain_len_min: int = None, # if less then `neg_chain_len_min`, only predict <s><s><NEGATIVE>
        neg_chain_identifiers: Optional[List[str]] = None,
        do_pointer_tags: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            tokenizer_model_name=tokenizer_model_name,
            tokenizer_kwargs=tokenizer_kwargs,
            lowercase=lowercase,
            length_limit=length_limit,
            truncation_strategy=truncation_strategy,
            test_mode=test_mode,
            source_prefix=source_prefix,
            target_prefix=target_prefix,
            target_suffix=target_suffix,
            task_specific_args=task_specific_args,
            **kwargs)

        if not event_sep in self._tokenizer.additional_special_tokens:
            raise ValueError("event_sep should be added into the additional_special_tokens of the tokenizer")
        self._event_sep = event_sep

        if not args_sep in self._tokenizer.additional_special_tokens:
            raise ValueError("args_sep should be added into the additional_special_tokens of the tokenizer")
        self._args_sep = args_sep

        self._event_del_prob = event_del_prob
        self._event_del_all_prob = event_del_all_prob
        self._allow_empty_events = allow_empty_events
        self._event_keep_num = event_keep_num
        random.seed(seed)

        varg_format_map = {
            "ARGS": self.get_varg_strings_ARGS,
            "V_ARGS": self.get_varg_strings_V_ARGS,
        }
        self._get_source_strings = varg_format_map[source_format]
        self._get_target_strings = varg_format_map[target_format]

        self._chain_len_min = chain_len_min or 0
        self._chain_len_max = chain_len_max or 1000000000000 # just a large number

        self._neg_chain_target_type = neg_chain_target_type
        if not self._neg_chain_target_type in {"unrelated_events", "empty", "gold"}:
            raise ValueError("Unknown neg_chain_target_type: {}".format(neg_chain_target_type))

        self._neg_chain_identifiers = neg_chain_identifiers or []

        if len(self._neg_chain_identifiers) > 0:
            if len(pos_chain_prefix) > 0 and not pos_chain_prefix in self._tokenizer.additional_special_tokens:
                raise ValueError("pos_chain_prefix should be added into the additional_special_tokens of the tokenizer")
            if len(neg_chain_prefix) > 0 and not neg_chain_prefix in self._tokenizer.additional_special_tokens:
                raise ValueError("neg_chain_prefix should be added into the additional_special_tokens of the tokenizer")
            self._pos_chain_prefix = pos_chain_prefix
            self._neg_chain_prefix = neg_chain_prefix
        else:
            self._pos_chain_prefix = None
            self._neg_chain_prefix = None

        self._neg_chain_len_min = neg_chain_len_min

        self._do_pointer_tags = do_pointer_tags
        if do_pointer_tags:
            for sep in POINTER_EVENT_TAGS:
                if not sep in self._tokenizer.additional_special_tokens:
                    raise ValueError(f"{sep} should be added into the additional_special_tokens of the tokenizer")

    def preprocess_example(self, example, data_src):
        example['data_src'] = data_src
        # metadata
        if not "doc_id" in example:
            example["doc_id"] = None
        if not "aug_metadata" in example:
            example["aug_metadata"] = None

    @overrides
    def _read(self, file_path: str):
        if file_path[0] == '{' and file_path[-1] == '}':
            file_path_dict = json.loads(file_path)
            dataset = []
            for data_src, file_path in file_path_dict.items():
                file_params = self._task_specific_args[data_src]
                data_src_weight = file_params.pop('weight') if 'weight' in file_params else 1
                assert type(data_src_weight) == int
                # if `file_path` is a URL, redirect to the cache
                file_path = cached_path(file_path)

                self.set_task_specific_args(file_params)
                logger.info("Reading file at %s", file_path)
                with open(file_path, 'rb') as dataset_file:
                    cur_dataset = pickle.load(dataset_file)
                    for example in cur_dataset:
                        self.preprocess_example(example, data_src)
                    logger.info(f"Up-sample {data_src} dataset by {data_src_weight} ({len(cur_dataset)} -> {len(cur_dataset)*data_src_weight})")
                    dataset += cur_dataset * data_src_weight
                self.set_task_specific_args(self._default_task_specific_args)
        else:
            # if `file_path` is a URL, redirect to the cache
            file_path = cached_path(file_path)

            logger.info("Reading file at %s", file_path)
            with open(file_path, 'rb') as dataset_file:
                dataset = pickle.load(dataset_file)
                for example in dataset:
                    self.preprocess_example(example, '')
        # now allennlp's lazy dataset only works with unshuffle, so manually shuffle here.
        np.random.shuffle(dataset)
        logger.info("Reading the dataset")

        # get chains whose length is in [chain_len_min, chain_len_max]
        logger.info("Exract chains with length from %d to %d" % (self._chain_len_min, self._chain_len_max))
        dataset = [d for d in dataset if self._chain_len_min <= len(d['varg_seq']) <= self._chain_len_max]

        print_stat_with_posneg(dataset, print_func=logger.info)
        print_stat_chainlen(dataset, print_func=logger.info)

        # yield instances
        print("Loading instance to .csv file")
        num_instances = 0
        num_valid_examples = 0
        examples_with_more_than_one_instance = 0
        self._instances_exceed_length_limit = 0
        df = []
        for example in dataset:
            self.set_task_specific_args(self._task_specific_args[example['data_src']])
            instances = self.make_instances(example)
            instances_yielded = 0
            for instance in instances:
                # print("src:",instance['source_str'],";tgt:",instance['target_str'])
                metadata = instance.__getitem__("metadata").metadata
                df.append([metadata['source_str'].replace('<s><s> ', '').replace('</s>', '').replace('\t', ''), metadata['target_str'].replace('<s><s> ', '').replace('</s>', '').replace('\t', '')])
                yield instance
                num_instances += 1
                instances_yielded += 1
            num_valid_examples += 1
            if instances_yielded > 1:
                examples_with_more_than_one_instance += 1
            self.set_task_specific_args(self._default_task_specific_args)
        
        pd.DataFrame(df, columns=['source', 'target']).to_csv('caters.csv', index=None, sep='\t')
        

        logger.info("Total instances yielded: %d", num_instances)
        logger.info("%d (%.2f%%) examples have more than one instance",
                    examples_with_more_than_one_instance,
                    100 * examples_with_more_than_one_instance / num_valid_examples)
        logger.info("%d (%.2f%%) instances exceed the length limit",
                    self._instances_exceed_length_limit,
                    100 * self._instances_exceed_length_limit / num_instances)
        
        1/0
            

    def normalize_arg_type(self, arg_type):
        if arg_type[0] in ['R', 'C']:
            return arg_type[2:]
        else:
            return arg_type

    def get_flatten_varg_toks(self, varg):
        varg_toks = [varg['V_toks']] + varg['ARGS_toks']
        varg_span = [varg['V_span']] + varg['ARGS_span']
        varg_type = ['V'] + [self.normalize_arg_type(arg_type) for arg_type in varg['ARGS_type']]
        assert len(varg_toks) == len(varg_span) and len(varg_toks) == len(varg_type)
        indices = list(range(len(varg_toks)))
        # sort pred/args by their textual order
        indices = sorted(indices, key=lambda x: varg_span[x])
        varg_toks = [varg_toks[i] for i in indices]
        varg_type = [varg_type[i] for i in indices]
        flatten_toks = []
        for i, toks in enumerate(varg_toks):
            flatten_toks.extend(toks)
        return flatten_toks

    def del_vargs(self, varg_seq):
        if self._event_keep_num >= 0:
            idxs = list(range(len(varg_seq)))
            random.shuffle(idxs)
            keep_varg_ids = sorted(idxs[:self._event_keep_num])
            keep_varg_seq = [varg_seq[i] for i in keep_varg_ids]
            return keep_varg_seq, keep_varg_ids
        if self._event_del_all_prob > 0:
            if random.random() < self._event_del_all_prob:
                return [], []
        if self._event_del_prob > 0:
            keep_varg_seq = []
            keep_varg_ids = []
            for varg_id, varg in enumerate(varg_seq):
                if random.random() < self._event_del_prob:
                    if self._allow_empty_events:
                        continue
                    elif not (varg_id == len(varg_seq) - 1 and len(keep_varg_seq) == 0):
                        continue
                keep_varg_ids.append(varg_id)
                keep_varg_seq.append(varg)
            return keep_varg_seq, keep_varg_ids
        return varg_seq, list(range(len(varg_seq)))

    def get_varg_strings_ARGS(self, varg_seq, varg2eventtag):
        # <EVENT_SEP>ARG0 V ARG1 ... <EVENT_SEP>ARG0 V ARG1 ...
        all_str = ""
        for varg_id, varg in enumerate(varg_seq):
            if 'Description' in varg:
                desc = varg['Description']
                varg_key = desc + '\t' + " ".join(varg.get('sent_toks', [])) + '\t' + str(varg.get('tok_start_in_doc', None))
            else:
                varg_toks = self.get_flatten_varg_toks(varg)
                desc = " ".join(varg_toks)
                varg_key = desc + '\t' + " ".join(varg.get('sent_toks', [])) + '\t' + str(varg.get('tok_start_in_doc', None))
            if varg2eventtag is None:
                assert not self._do_pointer_tags
                cur_event_sep = self._event_sep
            else:
                assert self._do_pointer_tags
                cur_event_sep = varg2eventtag[varg_key] if varg_key in varg2eventtag else self._event_sep
            if not "EVENT_SEP" in varg:
                varg["EVENT_SEP"] = cur_event_sep
            else:
                assert varg["EVENT_SEP"] == cur_event_sep
            all_str += cur_event_sep
            if self._lowercase:
                all_str += desc.lower()
            else:
                all_str += desc
        return all_str

    def get_varg_strings_V_ARGS(self, varg_seq, varg2eventtag):
        # <EVENT_SEP>V<ARGS>ARG0 V ARG1 ... <EVENT_SEP>V<ARGS>ARG0 V ARG1 ...
        all_str = ""
        for varg_id, varg in enumerate(varg_seq):
            if 'Description' in varg:
                desc = varg['Description']
                varg_key = desc + '\t' + " ".join(varg.get('sent_toks', [])) + '\t' + str(varg.get('tok_start_in_doc', None))
            else:
                varg_toks = self.get_flatten_varg_toks(varg)
                desc = " ".join(varg_toks)
                varg_key = desc + '\t' + " ".join(varg.get('sent_toks', [])) + '\t' + str(varg.get('tok_start_in_doc', None))
            if varg2eventtag is None:
                assert not self._do_pointer_tags
                cur_event_sep = self._event_sep
            else:
                assert self._do_pointer_tags
                cur_event_sep = varg2eventtag[varg_key] if varg_key in varg2eventtag else self._event_sep
            if not "EVENT_SEP" in varg:
                varg["EVENT_SEP"] = cur_event_sep
            else:
                assert varg["EVENT_SEP"] == cur_event_sep
            all_str += cur_event_sep
            if self._lowercase:
                all_str += " ".join(varg['V_toks']).lower()
            else:
                all_str += " ".join(varg['V_toks'])
            all_str += self._args_sep
            if self._lowercase:
                all_str += desc.lower()
            else:
                all_str += desc
        return all_str

    def get_varg2eventtag(self, varg_seq):
        if self._do_pointer_tags:
            varg2eventtag = {}
            for varg_id, varg in enumerate(varg_seq):
                if 'Description' in varg:
                    desc = varg['Description']
                    varg_key = desc + '\t' + " ".join(varg.get('sent_toks', [])) + '\t' + str(varg.get('tok_start_in_doc', None))
                else:
                    varg_toks = self.get_flatten_varg_toks(varg)
                    desc = " ".join(varg_toks)
                    varg_key = desc + '\t' + " ".join(varg.get('sent_toks', [])) + '\t' + str(varg.get('tok_start_in_doc', None))
                varg2eventtag[varg_key] = POINTER_EVENT_TAGS[varg_id]
        else:
            varg2eventtag = None
        return varg2eventtag

    def make_single_instance(
        self,
        example: Dict[str, Any]
    ) -> Iterable[Instance]:
        example = deepcopy(example)
        varg_seq = example['varg_seq']
        permutation = example.get('permutation') # indices to the true varg_seq
        is_scrambled = example.get('label') # ``POS`` or ``NEG``
        _id = example["_id"]
        doc_id = example["doc_id"]
        data_src =  example['data_src']
        aug_metadata = example['aug_metadata']

        if len(self._neg_chain_identifiers) == 0:
            is_neg_chain = None
        elif aug_metadata is None or not aug_metadata['aug_type'] in self._neg_chain_identifiers:
            is_neg_chain = False
        else:
            is_neg_chain = True

        if not is_neg_chain:
            keep_varg_seq, keep_varg_ids = self.del_vargs(varg_seq)
        else:
            keep_varg_seq, keep_varg_ids = varg_seq, list(range(len(varg_seq)))

        varg2eventtag = self.get_varg2eventtag(keep_varg_seq)

        source_str = self._get_source_strings(keep_varg_seq, varg2eventtag=varg2eventtag)

        source_str = self._source_prefix + source_str
        source_encodes = self.tokenize_text(source_str)

        # get the gold chain, and target_str
        if not self._test_mode:
            if is_neg_chain is not None and is_neg_chain:
                if self._neg_chain_target_type == "unrelated_events":
                    gold_varg_seq = aug_metadata['replaced_events']
                    target_str = self._neg_chain_prefix + self._get_target_strings(gold_varg_seq, varg2eventtag=varg2eventtag)
                elif self._neg_chain_target_type == "empty":
                    gold_varg_seq = []
                    target_str = self._neg_chain_prefix
                elif self._neg_chain_target_type == "gold":
                    gold_varg_seq = aug_metadata['source_chain']
                    target_str = self._neg_chain_prefix + self._get_target_strings(gold_varg_seq, varg2eventtag=varg2eventtag)
            elif aug_metadata is not None and 'source_chain' in aug_metadata:
                gold_varg_seq = aug_metadata['source_chain']
                target_str = self._pos_chain_prefix + self._get_target_strings(gold_varg_seq, varg2eventtag=varg2eventtag)
            elif permutation is not None:
                order = sorted(list(range(len(varg_seq))), key=lambda x: permutation[x])
                gold_varg_seq = [varg_seq[i] for i in order]
                target_str = self._get_target_strings(gold_varg_seq, varg2eventtag=varg2eventtag)
                if is_neg_chain is not None: # should be pos here
                    target_str = self._pos_chain_prefix + target_str
            else:
                target_str = None
                gold_varg_seq = None
        else:
            target_str = None
            gold_varg_seq = None

        # preprocess target strings. Mainly for T5 models.
        target_str = None if target_str is None else self._target_prefix + target_str + self._target_suffix
        # tokenize the target.
        target_encodes = None if target_str is None else self.tokenize_text(target_str)

        additional_metadata = {
            "source_str": self._tokenizer.decode(source_encodes["input_ids"]),
            "target_str": None if target_str is None else self._tokenizer.decode(target_encodes["input_ids"]),
            "gold_varg_seq": gold_varg_seq,
            "varg_seq": varg_seq,
            "is_scrambled": is_scrambled,
            "permutation": permutation,
            "keep_varg_ids": keep_varg_ids,
            "_id": _id,
            "doc_id": doc_id,
            "data_src": data_src,
            "aug_metadata": aug_metadata,
            "is_neg_chain": is_neg_chain
        }

        instance = self.text_to_instance(
            source_encodes,
            target_encodes,
            deepcopy(additional_metadata)
        )
        return instance

    def make_instances(
        self,
        example: Dict[str, Any]
    ) -> Iterable[Instance]:

        yield self.make_single_instance(example)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        source_encodes: Dict[str, List[int]],
        target_encodes: Dict[str, List[int]] = None,
        additional_metadata: Dict[str, Any] = None,
    ) -> Instance:

        fields = {}

        if self._neg_chain_len_min is not None and additional_metadata is not None and additional_metadata['is_neg_chain'] and len(additional_metadata['gold_varg_seq']) < self._neg_chain_len_min and target_encodes is not None: # TODO: now assume the first three is <s> <s> <NEGATIVE>
            # remove events from target
            target_encodes["input_ids"] = target_encodes["input_ids"][:3]
            if self._return_token_type_ids:
                target_encodes["token_type_ids"] = target_encodes["token_type_ids"][:3]
            if self._return_attention_mask:
                target_encodes["attention_mask"] = target_encodes["attention_mask"][:3]

        # make the token_ids fields (array fields)
        pad_id = self._tokenizer.pad_token_id
        fields["source_tok_ids"] = ArrayField(np.array(source_encodes["input_ids"]), padding_value=pad_id, dtype=np.int64)
        if target_encodes is not None:
            fields["target_tok_ids"] = ArrayField(np.array(target_encodes["input_ids"]), padding_value=pad_id, dtype=np.int64)
        # make the token_type_ids fields (array fields)
        if self._return_token_type_ids:
            pad_id = self._tokenizer.pad_token_type_id
            fields["source_tok_type_ids"] = ArrayField(np.array(source_encodes["token_type_ids"]), padding_value=pad_id, dtype=np.int64)
            if target_encodes is not None:
                fields["target_tok_type_ids"] = ArrayField(np.array(target_encodes["token_type_ids"]), padding_value=pad_id, dtype=np.int64)
        if self._return_attention_mask:
            pad_id = 0
            fields["source_attention_mask"] = ArrayField(np.array(source_encodes["attention_mask"]), padding_value=pad_id, dtype=np.int64)
            if target_encodes is not None:
                fields["target_attention_mask"] = ArrayField(np.array(target_encodes["attention_mask"]), padding_value=pad_id, dtype=np.int64)

        '''
        print("source:")
        print(source_encodes)
        print(self._tokenizer.decode(source_encodes["input_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=True))
        print("target:")
        print(target_encodes)
        print(self._tokenizer.decode(target_encodes["input_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=True))
        print("meta")
        print(json.dumps(additional_metadata, indent=2))
        print("---"*20, '\n')
        input()
        '''
        if len(source_encodes["input_ids"]) >= self._length_limit:
            self._instances_exceed_length_limit += 1

        # make the metadata
        metadata = {}
        if additional_metadata is not None:
            metadata.update(additional_metadata)
        fields["metadata"] = MetadataField(metadata)

        return Instance(fields) 
