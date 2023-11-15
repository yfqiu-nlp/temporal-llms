from typing import Callable, Dict, Iterable, List, Optional, Tuple, Any
import torch
from overrides import overrides
import os
import sys

from allennlp.common import FromParams

from transformers import AutoTokenizer
#from transformers import BartForConditionalGeneration
from denoising_event_lm.modules.transformers.bart_for_eventlm import BartForConditionalGeneration_for_EventLM as BartForConditionalGeneration
#from denoising_event_lm.modules.transformers.gpt2_for_infillinglm import GPT2LMHeadModel_for_InfillingLM as GPT2LMHeadModel



def get_huggingface_tokenizer(pretrained_model_name_or_path, *inputs, **kwargs):
    """
    make transformers 3.0.2 add special tokens deterministically
    """
    if 'additional_special_tokens' in kwargs:
        additional_special_tokens = kwargs.pop('additional_special_tokens')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path,
                                              *inputs,
                                              **kwargs)
    tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})
    for extra_tok_id in range(len(additional_special_tokens)):
        if not tokenizer.added_tokens_decoder[tokenizer.vocab_size+extra_tok_id] == additional_special_tokens[extra_tok_id]:
            raise ValueError("special tokens not added deterministically: "+repr(additional_special_tokens)+repr(tokenizer.added_tokens_decoder))
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": '<PAD>'})
    return tokenizer


class TransformerForConditionalGeneration(torch.nn.Module, FromParams):
    """
    An allennlp wrapper for BartForConditionalGeneration so that it can be instantiated by ``from_params``
    """
    def __init__(
        self,
        transformer_name: str,
        attention_dropout: float = 0.,
        output_hidden_states: bool = True,
        output_attentions: bool = True,
        normalize_before: bool = False,
        seq2seq_type: str = "standard",
    ) -> None:
        super().__init__()
        if transformer_name.split('/')[-1].startswith("bart"):
            if seq2seq_type == "standard":
                seq2seq_cls = BartForConditionalGeneration
            else:
                raise ValueError("Unsupported seq2seq type for ConditionalGeneration: {}".format(seq2seq_type))
            self._base_model = seq2seq_cls.from_pretrained(
                transformer_name,
                config=os.path.join(os.getcwd(), "denoising_event_lm/modules/transformers/bart-large-config.json"),
                attention_dropout=attention_dropout,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                normalize_before=normalize_before,
            )
        else:
            raise ValueError("Unsupported transformer model: {}".format(transformer_name))

    def forward(self, *args, **kwargs):
        return self._base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self._base_model.generate(*args, **kwargs)


class TransformerForLMHeadModel(torch.nn.Module, FromParams):
    """
    An allennlp wrapper for BartForConditionalGeneration so that it can be instantiated by ``from_params``
    """
    def __init__(
        self,
        transformer_name: str,
        attn_pdrop: float = 0.,
        output_hidden_states: bool = True,
        output_attentions: bool = True,
        lm_type: str = "standard",
    ) -> None:
        super().__init__()
        if transformer_name.startswith("gpt2"):
            if lm_type == "standard":
                lm_cls =  GPT2LMHeadModel
            else:
                raise ValueError("Unsupported lm type for LMHeadModel: {}".format(lm_type))
            self._base_model = lm_cls.from_pretrained(
                transformer_name,
                config=os.path.join(os.getcwd(), f"denoising_event_lm/modules/transformers/{transformer_name}-config.json"),
                attn_pdrop=attn_pdrop,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
        else:
            raise ValueError("Unsupported transformer model: {}".format(transformer_name))

    def forward(self, *args, **kwargs):
        return self._base_model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self._base_model.generate(*args, **kwargs)
