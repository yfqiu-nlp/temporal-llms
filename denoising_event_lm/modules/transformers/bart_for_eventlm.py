import logging
import math
import random
import numpy
import warnings
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss

from transformers import BartForConditionalGeneration
from transformers.generation_utils import BeamHypotheses, top_k_top_p_filtering, calc_banned_ngram_tokens, calc_banned_bad_words_ids


logger = logging.getLogger(__name__)


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    valid_target = target.clone().detach()
    valid_target[target == ignore_index] = 0
    nll_loss = -lprobs.gather(dim=-1, index=valid_target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def calc_banned_bad_verbs_ids(prev_input_ids: Iterable[int], batch_bad_verbs_ids: Iterable[int]) -> Iterable[int]:
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_input_ids):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice, bad_verbs_ids_slice in zip(prev_input_ids, batch_bad_verbs_ids):
        if bad_verbs_ids_slice is not None:
            banned_tokens_slice = []

            for banned_token_seq in bad_verbs_ids_slice:
                assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                    bad_verbs_ids_slice
                )

                if _tokens_match(prev_input_ids_slice.tolist(), banned_token_seq[:-1]) is False:
                    # if tokens do not match continue
                    continue

                banned_tokens_slice.append(banned_token_seq[-1])

            banned_tokens.append(banned_tokens_slice)
        else:
            banned_tokens.append(None)

    return banned_tokens


class BartForConditionalGeneration_for_EventLM(BartForConditionalGeneration):
    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        loss_reduction='mean',
        label_smoothing=0.0,
        **unused,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
            with labels
            in ``[0, ..., config.vocab_size]``.
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Conditional generation example::
            # Mask filling only works for bart-large
            from transformers import BartTokenizer, BartForConditionalGeneration
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
            TXT = "My friends are <mask> but they eat too many carbs."
            model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
            input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
            logits = model(input_ids)[0]
            masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
            probs = logits[0, masked_index].softmax(dim=0)
            values, predictions = probs.topk(5)
            tokenizer.decode(predictions).split()
            # ['good', 'great', 'all', 'really', 'very']
        """
        if "lm_labels" in unused:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                DeprecationWarning,
            )
            labels = unused.pop("lm_labels")

        if labels is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        outputs = (lm_logits,) + outputs[1:]  # Add cache, hidden states and attention if they are here
        if labels is not None:
            if label_smoothing == 0.0:
                loss_fct = nn.CrossEntropyLoss(reduction=loss_reduction)
                # TODO(SS): do we need to ignore pad tokens in labels?
                masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            else:
                lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
                masked_lm_loss, _ = label_smoothed_nll_loss(
                    lprobs, labels, label_smoothing, ignore_index=-100
                )
            outputs = (masked_lm_loss,) + outputs

        return outputs

    def postprocess_next_token_scores(
        self,
        scores,
        input_ids,
        no_repeat_ngram_size,
        bad_words_ids,
        batch_bad_verbs_ids,
        cur_len,
        min_length,
        max_length,
        eos_token_id,
        repetition_penalty,
        batch_size,
        num_beams,
        cur_num_events,
        cur_num_args,
        max_num_events,
        min_num_events,
        event_token_ids,
        arg_token_id,
    ):
        # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:
            self.enforce_repetition_penalty_(
                scores, batch_size, num_beams, input_ids, repetition_penalty,
            )

        # set eos token prob to zero if min_length is not reached
        if eos_token_id is not None and cur_len < min_length:
            scores[:, eos_token_id] = -float("inf")

        # set eos token prob to zero if num_min_events is not reached
        if eos_token_id is not None and min_num_events is not None:
            scores[cur_num_events < min_num_events, eos_token_id] = -float("inf")

        # set event token prob to zero if num_max_events is reached
        if max_num_events is not None:
            for i, n in enumerate(cur_num_events.detach().cpu().numpy()):
                if n >= max_num_events:
                    scores[i, event_token_ids] = -float("inf")

        if no_repeat_ngram_size > 0:
            # calculate a list of banned tokens to prevent repetitively generating the same ngrams
            num_batch_hypotheses = batch_size * num_beams
            # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
            banned_batch_tokens = calc_banned_ngram_tokens(
                input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
            )
            for i, banned_tokens in enumerate(banned_batch_tokens):
                scores[i, banned_tokens] = -float("inf")

        if bad_words_ids is not None:
            # calculate a list of banned tokens according to bad words
            banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

            for i, banned_tokens in enumerate(banned_tokens):
                scores[i, banned_tokens] = -float("inf")

        if batch_bad_verbs_ids is not None:
            # calculate a list of banned tokens according to bad verbs
            banned_tokens = calc_banned_bad_verbs_ids(input_ids, batch_bad_verbs_ids)

            for i, banned_tokens in enumerate(banned_tokens):
                if banned_tokens is not None:
                    scores[i, banned_tokens] = -float("inf")

        return scores

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bad_verbs_ids: Optional[Iterable[int]] = None,
        ban_bad_verbs_event_idxs: Optional[List[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        encoder_outputs: Optional[torch.LongTensor] = None,
        max_num_events: Optional[int] = None,
        min_num_events: Optional[int] = None,
        event_token_ids: Optional[Iterable[int]] = None,
        arg_token_id: Optional[int] = None,
        input_suffix_ids: Optional[Iterable[int]] = None,
        input_suffix_start_event_idx: Optional[int] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        **model_specific_kwargs
    ) -> torch.LongTensor:
        r""" Generates sequences for models with a LM head. The method currently supports greedy decoding, beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.

        Adapted in part from `Facebook's XLM beam search code`_.

        .. _`Facebook's XLM beam search code`:
           https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529


        Parameters:

            input_ids: (`optional`) `torch.LongTensor` of shape `(batch_size, sequence_length)`
                The sequence used as a prompt for the generation. If `None` the method initializes
                it as an empty `torch.LongTensor` of shape `(1,)`.

            max_length: (`optional`) int
                The max length of the sequence to be generated.  Between `min_length` and infinity. Default to 20.

            min_length: (`optional`) int
                The min length of the sequence to be generated.  Between 0 and infinity. Default to 0.

            do_sample: (`optional`) bool
                If set to `False` greedy decoding is used. Otherwise sampling is used. Defaults to `False` as defined in `configuration_utils.PretrainedConfig`.

            early_stopping: (`optional`) bool
                if set to `True` beam search is stopped when at least `num_beams` sentences finished per batch. Defaults to `False` as defined in `configuration_utils.PretrainedConfig`.

            num_beams: (`optional`) int
                Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search. Default to 1.

            temperature: (`optional`) float
                The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.

            top_k: (`optional`) int
                The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.

            top_p: (`optional`) float
                The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

            repetition_penalty: (`optional`) float
                The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.

            pad_token_id: (`optional`) int
                Padding token. Default to specicic model pad_token_id or None if it does not exist.

            bos_token_id: (`optional`) int
                BOS token. Defaults to `bos_token_id` as defined in the models config.

            eos_token_id: (`optional`) int
                EOS token. Defaults to `eos_token_id` as defined in the models config.

            length_penalty: (`optional`) float
                Exponential penalty to the length. Default to 1.

            no_repeat_ngram_size: (`optional`) int
                If set to int > 0, all ngrams of size `no_repeat_ngram_size` can only occur once.
            bad_words_ids: (`optional`) list of lists of int
                `bad_words_ids` contains tokens that are not allowed to be generated. In order to get the tokens of the words that should not appear in the generated text, use `tokenizer.encode(bad_word, add_prefix_space=True)`.

            num_return_sequences: (`optional`) int
                The number of independently computed returned sequences for each element in the batch. Default to 1.

            attention_mask (`optional`) obj: `torch.LongTensor` of same shape as `input_ids`
                Mask to avoid performing attention on padding token indices.
                Mask values selected in ``[0, 1]``:
                ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
                Defaults to `None`.

                `What are attention masks? <../glossary.html#attention-mask>`__

            decoder_start_token_id=None: (`optional`) int
                If an encoder-decoder model starts decoding with a different token than BOS.
                Defaults to `None` and is changed to `BOS` later.

            use_cache: (`optional`) bool
                If `use_cache` is True, past key values are used to speed up decoding if applicable to model. Defaults to `True`.

            model_specific_kwargs: (`optional`) dict
                Additional model specific kwargs will be forwarded to the `forward` function of the model.

        Return:

            output: `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`
                sequence_length is either equal to max_length or shorter if all batches finished early due to the `eos_token_id`

        Examples::

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            outputs = model.generate(max_length=40)  # do greedy decoding
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, num_return_sequences=3)  # 3 generate sequences using by sampling
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('ctrl')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('ctrl')    # Download model and configuration from S3 and cache.
            input_context = 'Legal My neighbor is'  # "Legal" is one of the control codes for ctrl
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('gpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('gpt2')    # Download model and configuration from S3 and cache.
            input_context = 'My cute dog'  # "Legal" is one of the control codes for ctrl
            bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in ['idiot', 'stupid', 'shut up']]
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True, bad_words_ids=bad_words_ids)  # generate sequences without allowing bad_words to be generated
        """

        # We cannot generate if the model does not have a LM head
        if self.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head."
                "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
            )

        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
        assert temperature > 0, "`temperature` should be strictly positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."
        assert length_penalty > 0, "`length_penalty` should be strictly positive."
        assert (
            isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
        ), "`no_repeat_ngram_size` should be a positive integer."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictly positive integer."
        assert (
            bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
        ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"
        assert (
            bad_verbs_ids is None or isinstance(bad_verbs_ids, list) and isinstance(bad_verbs_ids[0], list)
        ), "`bad_verbs_ids` is either `None` or a list of lists of tokens that should not be generated"
        assert (
            ban_bad_verbs_event_idxs is None or isinstance(ban_bad_verbs_event_idxs, list) and isinstance(ban_bad_verbs_event_idxs[0], int)
        ), "`ban_bad_verbs_event_idxs` is either `None` or a list of integers"

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device,
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."
        assert (
            isinstance(event_token_ids, list) and all(isinstance(i, int) and i >= 0 for i in event_token_ids)
        ), "`event_token_ids` should be a list of positive integer."
        assert (
            isinstance(arg_token_id, int) and (arg_token_id >= 0)
        ), "`arg_token_id` should be a positive integer."

        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams == 1:
                # no_beam_search greedy generation conditions
                assert (
                    num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

            else:
                # beam_search greedy generation conditions
                assert (
                    num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

        # create attention mask if necessary
        # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
        if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
            attention_mask = input_ids.ne(pad_token_id).long()
        elif attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # set pad_token_id to eos_token_id if not set. Important that this is done after
        # attention_mask is created
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(
                "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
            )
            pad_token_id = eos_token_id

        # current position and vocab size
        if hasattr(self.config, "vocab_size"):
            vocab_size = self.config.vocab_size
        elif (
            self.config.is_encoder_decoder
            and hasattr(self.config, "decoder")
            and hasattr(self.config.decoder, "vocab_size")
        ):
            vocab_size = self.config.decoder.vocab_size

        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        if self.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                decoder_start_token_id = bos_token_id

            assert (
                decoder_start_token_id is not None
            ), "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
            assert hasattr(self, "get_encoder"), "{} should have a 'get_encoder' function defined".format(self)
            assert callable(self.get_encoder), "{} should be a method".format(self.get_encoder)

            if encoder_outputs is None:
                # get encoder and store encoder outputs
                encoder = self.get_encoder()

                encoder_outputs: tuple = encoder(input_ids, attention_mask=attention_mask)

        # Expand input ids if num_beams > 1 or num_return_sequences > 1
        if num_return_sequences > 1 or num_beams > 1:
            input_ids_len = input_ids.shape[-1]
            input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
            attention_mask = attention_mask.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len
            )

            input_ids = input_ids.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            attention_mask = attention_mask.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

        if self.config.is_encoder_decoder:
            # create empty decoder_input_ids
            if decoder_input_ids is None:
                input_ids = torch.full(
                    (effective_batch_size * num_beams, 1),
                    decoder_start_token_id,
                    dtype=torch.long,
                    device=next(self.parameters()).device,
                )
                cur_len = 1
                cur_num_events = input_ids.new_zeros((effective_batch_size * num_beams,))
                cur_num_args = input_ids.new_zeros((effective_batch_size * num_beams,))
            else:
                assert (
                    2 == len(decoder_input_ids.shape)
                ), f"expected decoder_input_ids to have 2 dimensions, got {len(decoder_input_ids.shape)} "
                assert (
                    batch_size == decoder_input_ids.shape[0]
                ), f"expected decoder_input_ids to have 1st dimension bs={batch_size}, got {decoder_input_ids.shape[0]} "
                assert (
                    torch.all(decoder_input_ids[:, 0] == decoder_start_token_id)
                ), f"expected decoder_input_ids to have 1st dimension bs={batch_size}, got\n {decoder_input_ids} "
                input_ids = decoder_input_ids
                cur_len = input_ids.shape[-1]

                # expand input_ids
                input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, cur_len)
                input_ids = input_ids.contiguous().view(
                    effective_batch_size * num_beams, cur_len
                )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

                # compute cur_num_events
                cur_num_events = torch.sum(torch.any(input_ids.unsqueeze(-1) == input_ids.new_tensor(event_token_ids), dim=-1), dim=-1)
                cur_num_args = torch.sum(input_ids == arg_token_id, dim=-1)

            assert (
                batch_size == encoder_outputs[0].shape[0]
            ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

            # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
            expanded_batch_idxs = (
                torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, num_beams * effective_batch_mult)
                .view(-1)
                .to(input_ids.device)
            )
            # expand encoder_outputs
            encoder_outputs = (encoder_outputs[0].index_select(0, expanded_batch_idxs), *encoder_outputs[1:])

        else:
            encoder_outputs = None
            cur_len = input_ids.shape[-1]
            # compute cur_num_events
            cur_num_events = torch.sum(torch.any(input_ids.unsqueeze(-1) == input_ids.new_tensor(event_token_ids), dim=-1), dim=-1)
            cur_num_args = torch.sum(input_ids == arg_token_id, dim=-1)

        assert (
            cur_len < max_length
        ), f"The context has {cur_len} number of tokens, but `max_length` is only {max_length}. Please make sure that `max_length` is bigger than the number of tokens, by setting either `generate(max_length=...,...)` or `config.max_length = ...`"
        assert (
            max_num_events is None or torch.all(cur_num_events <= max_num_events)
        ), f"The context has {cur_num_events} number of events, but `max_num_events` is only {max_num_events}. Please make sure that `max_num_events` is bigger than or equal to the number of events, by setting `generate(max_num_events=...,...)`"
        assert (
            ban_bad_verbs_event_idxs is None or all(torch.all(cur_num_args <= e_id) for e_id in ban_bad_verbs_event_idxs)
        ), f"The context has {cur_num_args} number of args, but `ban_bad_verbs_event_idxs` is {ban_bad_verbs_event_idxs}. Please make sure that `ban_bad_verbs_event_idxs` is bigger than or equal to the number of events, by setting `generate(ban_bad_verbs_event_idxs=...,...)`"
        assert (
            input_suffix_start_event_idx is None or torch.all(cur_num_events <= input_suffix_start_event_idx)
        ), f"The context has {cur_num_events} number of args, but `input_suffix_start_event_idx` is {input_suffix_start_event_idx}. Please make sure that `input_suffix_start_event_idx` is bigger than or equal to the number of events, by setting `generate(input_suffix_start_event_idx=...,...)`"

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len=cur_len,
                cur_num_events=cur_num_events,
                cur_num_args=cur_num_args,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                early_stopping=early_stopping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                bad_verbs_ids=bad_verbs_ids,
                ban_bad_verbs_event_idxs=ban_bad_verbs_event_idxs,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                num_beams=num_beams,
                vocab_size=vocab_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                use_cache=use_cache,
                max_num_events=max_num_events,
                min_num_events=min_num_events,
                event_token_ids=event_token_ids,
                arg_token_id=arg_token_id,
                input_suffix_ids=input_suffix_ids,
                input_suffix_start_event_idx=input_suffix_start_event_idx,
                model_specific_kwargs=model_specific_kwargs,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                cur_len=cur_len,
                cur_num_events=cur_num_events,
                cur_num_args=cur_num_args,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                bad_verbs_ids=bad_verbs_ids,
                ban_bad_verbs_event_idxs=ban_bad_verbs_event_idxs,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                use_cache=use_cache,
                max_num_events=max_num_events,
                min_num_events=min_num_events,
                event_token_ids=event_token_ids,
                arg_token_id=arg_token_id,
                input_suffix_ids=input_suffix_ids,
                input_suffix_start_event_idx=input_suffix_start_event_idx,
                model_specific_kwargs=model_specific_kwargs,
            )

        return output

    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        cur_num_events,
        cur_num_args,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        bad_verbs_ids,
        ban_bad_verbs_event_idxs,
        pad_token_id,
        eos_token_id,
        batch_size,
        encoder_outputs,
        attention_mask,
        use_cache,
        max_num_events,
        min_num_events,
        event_token_ids,
        arg_token_id,
        input_suffix_ids,
        input_suffix_start_event_idx,
        model_specific_kwargs,
    ):
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        # the indices to the input_suffix_ids
        if input_suffix_ids is not None:
            input_suffix_timesteps = [None] * batch_size
        else:
            input_suffix_timesteps = None

        past = (encoder_outputs, None) if encoder_outputs is not None else None

        decoded_probs = -1.0 * torch.ones((batch_size, 1), dtype=torch.float, device=input_ids.device)
        while cur_len < max_length:
            if bad_verbs_ids is not None and ban_bad_verbs_event_idxs is not None:
                batch_bad_verbs_ids = [bad_verbs_ids if n-1 in ban_bad_verbs_event_idxs else None for n in cur_num_events.detach().cpu().numpy()] # n event tokens means now in the span of the (n-1)th events
                if all(e is None for e in batch_bad_verbs_ids):
                    batch_bad_verbs_ids = None
            else:
                batch_bad_verbs_ids = None

            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
            )

            outputs = self(**model_inputs)
            next_token_logits = outputs[0][:, -1, :]

            scores = self.postprocess_next_token_scores(
                scores=next_token_logits,
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                batch_bad_verbs_ids=batch_bad_verbs_ids,
                cur_len=cur_len,
                min_length=min_length,
                max_length=max_length,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=1,
                cur_num_events=cur_num_events,
                cur_num_args=cur_num_args,
                max_num_events=max_num_events,
                min_num_events=min_num_events,
                event_token_ids=event_token_ids,
                arg_token_id=arg_token_id,
            )

            # if model has past, then set the past variable to speed up decoding
            if self._use_cache(outputs, use_cache):
                past = outputs[1]

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    scores = scores / temperature
                # Top-p/top-k filtering
                next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
                # Sample
                probs = F.softmax(next_token_logscores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                probs = F.softmax(scores, dim=-1)
                next_token = torch.argmax(scores, dim=-1)
            # replace next token with input suffix if needed
            if input_suffix_ids is not None:
                for b_idx, suffix_t in enumerate(input_suffix_timesteps):
                    if suffix_t is not None:
                        if suffix_t < len(input_suffix_ids):
                            next_token[b_idx] = input_suffix_ids[suffix_t]
                            input_suffix_timesteps[b_idx] += 1
                        else:
                            next_token[b_idx] = pad_token_id
            next_token_probs = torch.gather(probs, dim=1, index=next_token.unsqueeze(-1)).squeeze(-1)

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token
            scores_to_add = next_token_probs * unfinished_sents + (-1) * (1 - unfinished_sents)

            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            decoded_probs = torch.cat([decoded_probs, scores_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            # update cur_num_events
            cur_num_events = cur_num_events + (torch.any(tokens_to_add.unsqueeze(-1) == tokens_to_add.new_tensor(event_token_ids), dim=-1)).int()
            cur_num_args = cur_num_args + (tokens_to_add == arg_token_id).int()

            # update input suffix time steps
            if input_suffix_ids is not None:
                for b_idx, n in enumerate(cur_num_events.detach().cpu().numpy()):
                    if n-1 == input_suffix_start_event_idx and input_suffix_timesteps[b_idx] is None:
                        assert input_suffix_ids[0] in event_token_ids
                        assert tokens_to_add[b_idx] in event_token_ids, repr(tokens_to_add[b_idx])+'\n'+repr(event_token_ids) + '\n' + repr(torch.sum(torch.any(tokens_to_add.unsqueeze(-1) == tokens_to_add.new_tensor(event_token_ids), dim=-1), dim=-1)) + '\n' + repr(tokens_to_add)
                        input_suffix_timesteps[b_idx] = 1

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            # extend attention_mask for new generated input if only decoder
            if self.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        assert torch.all(torch.logical_and(min_num_events <= cur_num_events, cur_num_events <= max_num_events))

        # compute sequence scores
        best_scores = []
        for hypo_idx, hypo in enumerate(input_ids):
            scored_toks = [v.item() for v in decoded_probs[hypo_idx, :] if v != -1]
            scored_toks = [v+1e-9 if v == 0 else v for v in scored_toks]
            # compute avg of log probs
            score = numpy.sum(numpy.log(scored_toks)) / len(scored_toks)
            best_scores.append(score)

        return input_ids

    def _generate_beam_search(
        self,
        input_ids,
        cur_len,
        cur_num_events,
        cur_num_args,
        max_length,
        min_length,
        do_sample,
        early_stopping,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        bad_verbs_ids,
        ban_bad_verbs_event_idxs,
        pad_token_id,
        eos_token_id,
        batch_size,
        num_return_sequences,
        length_penalty,
        num_beams,
        vocab_size,
        encoder_outputs,
        attention_mask,
        use_cache,
        max_num_events,
        min_num_events,
        event_token_ids,
        arg_token_id,
        input_suffix_ids,
        input_suffix_start_event_idx,
        model_specific_kwargs,
    ):
        """ Generate sequences for each example with beam search.
        """

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
            for _ in range(batch_size)
        ]

        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

        # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
        if do_sample is False:
            beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # the indices to the input_suffix_ids
        if input_suffix_ids is not None:
            input_suffix_timesteps = [None] * (batch_size * num_beams)
        else:
            input_suffix_timesteps = None

        # cache compute states
        past = (encoder_outputs, None) if encoder_outputs is not None else None

        # done sentences
        done = [False for _ in range(batch_size)]

        while cur_len < max_length:
            if bad_verbs_ids is not None and ban_bad_verbs_event_idxs is not None:
                batch_bad_verbs_ids = [bad_verbs_ids if n-1 in ban_bad_verbs_event_idxs else None for n in cur_num_events.detach().cpu().numpy()] # n event tokens means now in the span of the (n-1)th events
                if all(e is None for e in batch_bad_verbs_ids):
                    batch_bad_verbs_ids = None
            else:
                batch_bad_verbs_ids = None

            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
            )
            outputs = self(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
            next_token_logits = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

            # if model has past, then set the past variable to speed up decoding
            if self._use_cache(outputs, use_cache):
                past = outputs[1]
            if self.config.is_encoder_decoder and do_sample is False:
                # TODO (PVP) still a bit hacky here - there might be a better solution
                next_token_logits = self.adjust_logits_during_generation(
                    next_token_logits, cur_len=cur_len, max_length=max_length
                )

            scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            scores = self.postprocess_next_token_scores(
                scores=scores,
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                batch_bad_verbs_ids=batch_bad_verbs_ids,
                cur_len=cur_len,
                min_length=min_length,
                max_length=max_length,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=num_beams,
                cur_num_events=cur_num_events,
                cur_num_args=cur_num_args,
                max_num_events=max_num_events,
                min_num_events=min_num_events,
                event_token_ids=event_token_ids,
                arg_token_id=arg_token_id,
            )

            # replace next token with input suffix if needed
            if input_suffix_ids is not None:
                for b_idx, suffix_t in enumerate(input_suffix_timesteps):
                    if suffix_t is not None:
                        if suffix_t < len(input_suffix_ids):
                            scores[b_idx, [not v == input_suffix_ids[suffix_t] for v in range(vocab_size)]] = -float("inf")
                            input_suffix_timesteps[b_idx] += 1
                        else:
                            scores[b_idx, [not v == pad_token_id for v in range(vocab_size)]] = -float("inf")

            assert scores.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
                scores.shape, (batch_size * num_beams, vocab_size)
            )

            if do_sample:
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                # Temperature
                if temperature != 1.0:
                    _scores = _scores / temperature
                # Top-p/top-k filtering
                _scores = top_k_top_p_filtering(
                    _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together to sample from all beam_idxs
                _scores = _scores.contiguous().view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
                probs = F.softmax(_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)  # (batch_size, num_beams * 2)
                # Compute next scores
                next_scores = torch.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)
                # sort the sampled vector to make sure that the first num_beams samples are the best
                next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)

            else:
                next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                next_scores = next_scores.view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)

            # next batch beam content
            next_batch_beam = []

            # for each sentence
            for batch_idx in range(batch_size):

                # if we are done with this sentence, add a pad token
                if done[batch_idx]:
                    assert (
                        len(generated_hyps[batch_idx]) >= num_beams
                    ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                    assert (
                        eos_token_id is not None and pad_token_id is not None
                    ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content, this will get added to next_batch_beam
                next_sent_beam = []

                # next tokens for this sentence
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    # get beam and token IDs
                    beam_id = beam_token_id // vocab_size
                    token_id = beam_token_id % vocab_size

                    effective_beam_id = batch_idx * num_beams + beam_id
                    # add to generated hypotheses if end of sentence
                    if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                        # if beam_token does not belong to top num_beams tokens, it should not be added
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                    else:
                        # add next predicted token since it is not eos_token
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    # once the beam for next step is full, don't add more tokens to it.
                    if len(next_sent_beam) == num_beams:
                        break

                # Check if we are done so that we can save a pad step if all(done)
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    next_scores[batch_idx].max().item(), cur_len
                )

                # update next beam content
                assert len(next_sent_beam) == num_beams, "Beam should always be full"
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_idx + 1), "We should have added num_beams each step"

            # stop when we are done with each sentence
            if all(done):
                break

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch and update current length
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1

            # re-order cur_num_events and update cur_num_events
            cur_num_events = cur_num_events[beam_idx]
            cur_num_args = cur_num_args[beam_idx]
            cur_num_events = cur_num_events + (torch.any(beam_tokens.unsqueeze(-1) == beam_tokens.new_tensor(event_token_ids), dim=-1)).int()
            cur_num_args = cur_num_args + (beam_tokens == arg_token_id).int()

            # update input suffix time steps
            if input_suffix_ids is not None:
                input_suffix_timesteps = [input_suffix_timesteps[b_idx] for b_idx in beam_idx]
                for b_idx, n in enumerate(cur_num_events.detach().cpu().numpy()):
                    if n-1 == input_suffix_start_event_idx and input_suffix_timesteps[b_idx] is None:
                        assert input_suffix_ids[0] in event_token_ids
                        assert beam_tokens[b_idx] in event_token_ids
                        input_suffix_timesteps[b_idx] = 1

            # re-order internal states
            if past is not None and use_cache:
                past = self._reorder_cache(past, beam_idx)
            else:
                # encoder outputs are same with for each beam in a sample
                pass

            # extend attention_mask for new generated input if only decoder
            if self.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue

            # test that beam scores match previously calculated scores if not eos and batch_idx not done
            if eos_token_id is not None and all(
                (token_id % vocab_size).item() != eos_token_id for token_id in next_tokens[batch_idx]
            ):
                assert torch.all(
                    next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
                ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                    next_scores[:, :num_beams][batch_idx], beam_scores.view(batch_size, num_beams)[batch_idx],
                )

            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(num_beams):
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)

        # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
        output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
        output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences

        # select the best hypotheses
        sent_lengths = input_ids.new(output_batch_size)
        best = []

        # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)

        # shorter batches are padded
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined"
            sent_max_len = min(sent_lengths.max().item() + 1, max_length)
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

            # fill with hypothesis and eos_token_id if necessary
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < max_length:
                    decoded[i, sent_lengths[i]] = eos_token_id
        else:
            # none of the hypotheses have an eos_token
            assert (len(hypo) == max_length for hypo in best)
            decoded = torch.stack(best).type(torch.long).to(next(self.parameters()).device)

        return decoded
