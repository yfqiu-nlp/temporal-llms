import json, pickle
import logging
from typing import Any, Dict, List, Tuple, Optional, Iterable
from copy import deepcopy
import numpy as np

from allennlp.data.fields import MetadataField, ArrayField
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance

from denoising_event_lm.modules.transformers import get_huggingface_tokenizer


logger = logging.getLogger(__name__)


@DatasetReader.register("seq2seq_transformer_reader")
class Seq2SeqTransformerReader(DatasetReader):
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
    stride : ``int``, optional (default=-1)
        If this is -1, we truncate the context as specified when calling ``self.tokenizer.encode``.
        Otherwise, when context are too long for the length limit, we emit multiple instances for one question,
        where the context is shifted. This parameter specifies the overlap between the shifted context window.
    truncation_strategy : `str`, optional (default=`'longest_first'`)
        String selected in the following options:
        - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
        starting from the longest one at each token (when there is a pair of input sequences)
        - 'only_first': Only truncate the first sequence
        - 'only_second': Only truncate the second sequence
        - 'do_not_truncate': Do not truncate (raise an error if the input sequence is longer than max_length)
    test_mode : ``bool``, optional (default=True)
        whether we are in the test mode.
    context_prefix : ``str``, optional (default="")
        the string to prepend on context. Mainly for T5 models.
    question_prefix : ``str``, optional (default="")
        the string to prepend on question. Mainly for T5 models.
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
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = get_huggingface_tokenizer(tokenizer_model_name, **(tokenizer_kwargs or {}))
        self._lowercase = lowercase
        self._length_limit = length_limit or self._tokenizer.model_max_length # since truncation in tokenizer will consider added special tokens
        self._truncation_strategy = truncation_strategy
        self._test_mode = test_mode

        self._source_prefix = source_prefix if len(source_prefix) == 0 else (source_prefix + ('' if source_prefix[-1] == ' ' else ' '))
        self._target_prefix = target_prefix if len(target_prefix) == 0 else (target_prefix + ('' if target_prefix[-1] == ' ' else ' '))
        self._target_suffix = target_suffix if len(target_suffix) == 0 else (('' if target_suffix[0] == ' ' else ' ') + target_suffix)
        #self._source_prefix = source_prefix
        #self._target_prefix = target_prefix
        #self._target_suffix = target_suffix

        self._return_token_type_ids = "token_type_ids" in self._tokenizer.model_input_names
        self._return_attention_mask = "attention_mask" in self._tokenizer.model_input_names

        if len(self._source_prefix) > 0:
            self._source_prefix_tok_ids = self.tokenize_text(self._source_prefix, add_special_tokens=False, add_prefix_space=False)["input_ids"]
        else:
            self._source_prefix_tok_ids = []

        # get default task-specific arguments for multi-task training
        self._default_task_specific_args = self.get_default_task_specific_args()
        self._task_specific_args = task_specific_args or {}
        self._task_specific_args[''] = self._default_task_specific_args

    @staticmethod
    def get_answer_strings(tokens, answer_tok_idxs):
        if answer_tok_idxs is not None:
            answers_str = [" ".join(tokens[p] for p in idxs) for idxs in answer_tok_idxs]
        else:
            answers_str = None
        return answers_str

    def get_default_task_specific_args(self):
        # remember to deepcopy if needed
        default_args = {
            "_length_limit": self._length_limit
        }
        return default_args

    def set_task_specific_args(self, kwargs):
        for k, v in kwargs.items():
            assert hasattr(self, k)
            setattr(self, k, v)

    def preprocess_example(self, example, data_src):
        example['data_src'] = data_src
        # metadata
        if not "doc_id" in example:
            example["doc_id"] = None
        # process source if needed
        if "source" in example:
            pass
        elif "context" in example:
            example["source"] = example["context"]
        # process answers if needed
        if "answers_str" in example:
            pass
        elif "answers" in example:
            example["answers_str"] = example["answers"]
        elif "answer_tok_idxs" in  example:
            # get answer strings
            context_toks = example["context_tokens"]
            answer_tok_idxs = example["answer_tok_idxs"]
            answers_str = self.get_answer_strings(context_toks, answer_tok_idxs)
            example["answers_str"] = answers_str


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

        # yield instances
        num_instances = 0
        num_valid_examples = 0
        examples_with_more_than_one_instance = 0
        self._instances_exceed_length_limit = 0
        for example in dataset:
            self.set_task_specific_args(self._task_specific_args[example['data_src']])
            instances = self.make_instances(example)
            instances_yielded = 0
            for instance in instances:
                yield instance
                num_instances += 1
                instances_yielded += 1
            num_valid_examples += 1
            if instances_yielded > 1:
                examples_with_more_than_one_instance += 1
            self.set_task_specific_args(self._default_task_specific_args)


        logger.info("Total instances yielded: %d", num_instances)
        logger.info("%d (%.2f%%) examples have more than one instance",
                    examples_with_more_than_one_instance,
                    100 * examples_with_more_than_one_instance / num_valid_examples)
        logger.info("%d (%.2f%%) instances exceed the length limit",
                    self._instances_exceed_length_limit,
                    100 * self._instances_exceed_length_limit / num_instances)

    def tokenize_text(self, text, text_pair=None, add_special_tokens=True, add_prefix_space=False):
        # note: default set ``add_prefix_space`` to True.
        # This makes roberta-style encoders produce correct results when special tokens are added.
        encodes = self._tokenizer.encode_plus(
            text=text,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            max_length=self._length_limit,
            truncation=self._truncation_strategy,
            return_tensors=None,
            return_token_type_ids=self._return_token_type_ids,
            return_attention_mask=self._return_attention_mask,
            return_overflowing_tokens=False,
            return_special_tokens_mask=False,
            return_offsets_mapping=False,
            add_prefix_space=add_prefix_space
        )
        return encodes

    def make_single_instance(
        self,
        example: Dict[str, Any]
    ) -> Iterable[Instance]:

        source_str = example["source"]
        answers_str = None if self._test_mode else example["answers_str"]
        if self._lowercase:
            source_str = source_str.lower()
            answers_str = None if self._test_mode else [ans.lower() for ans in answers_str]
        _id = example["_id"]
        doc_id = example["doc_id"]
        data_src =  example['data_src']

        # preprocess target strings. Mainly for T5 models.
        target_idx = 0
        target_str = None if answers_str is None else self._target_prefix + answers_str[target_idx] + self._target_suffix # use first one as the target
        # tokenize the target.
        target_encodes = None if answers_str is None else self.tokenize_text(target_str)

        additional_metadata = {
            "raw_source_str": source_str,
            "source_str": None,
            "target_str": self._tokenizer.decode(target_encodes["input_ids"]),
            "answers_str": answers_str,
            "_id": _id,
            "doc_id": doc_id,
            "data_src": data_src,
        }

        source_str = self._source_prefix + source_str
        source_encodes = self.tokenize_text(source_str)
        additional_metadata["source_str"] = self._tokenizer.decode(source_encodes["input_ids"])
        instance = self.text_to_instance(
            source_encodes,
            target_encodes,
            deepcopy(additional_metadata),
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
