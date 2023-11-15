import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Any, Dict, List, Optional, Union
import re
import logging
import json
import torch
from torch import nn
from torch.nn import functional as F

from denoising_event_lm.modules.transformers import get_huggingface_tokenizer

from allennlp.common.params import Params
from allennlp.data import Instance, Vocabulary
from allennlp.data.batch import Batch
from allennlp.nn import util
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder
from allennlp.nn import util, RegularizerApplicator
from allennlp.training.metrics import Average

from rouge_score import rouge_scorer
rouge_scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)

from denoising_event_lm.models import ModelBase
from denoising_event_lm.training.metrics import metric_map
from denoising_event_lm.modules.transformers import TransformerForConditionalGeneration
from denoising_event_lm.utils.constants import EVENT_TAG, POINTER_EVENT_TAGS, ARGS_TAG


logger = logging.getLogger(__name__)


def normalize_arg_type(arg_type):
    if arg_type[0] in ['R', 'C']:
        return arg_type[2:]
    else:
        return arg_type


def get_flatten_varg_toks(varg):
    varg_toks = [varg['V_toks']] + varg['ARGS_toks']
    varg_span = [varg['V_span']] + varg['ARGS_span']
    varg_type = ['V'] + [normalize_arg_type(arg_type) for arg_type in varg['ARGS_type']]
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


def V_ARGS_string_to_varg_seq(varg_string, add_event_sep_entry=True):
    #vargs = varg_string.split(EVENT_TAG)[1:]
    regex = rf"({'|'.join([EVENT_TAG]+POINTER_EVENT_TAGS)})(.*?)(?={'|'.join(['$',EVENT_TAG]+POINTER_EVENT_TAGS)})"
    vargs = [(x.group(1), x.group(2)) for x in re.finditer(regex, varg_string)]
    varg_seq = []
    for event_sep, varg in vargs:
        v, *desc = varg.split(ARGS_TAG)
        desc = f" {ARGS_TAG} ".join(desc)
        if add_event_sep_entry:
            varg_seq.append(
                {
                    "V_toks": [v.strip()],
                    "Description": desc.strip(),
                    "EVENT_SEP": event_sep
                }
            )
        else:
            varg_seq.append(
                {
                    "V_toks": [v.strip()],
                    "Description": desc.strip()
                }
            )
    return varg_seq


def get_event_matching(varg_seq_a, varg_seq_b):
    # get description if needed: ARG0 Pred ARG1 ...
    if len(varg_seq_a) > 0 and not 'Description' in varg_seq_a[0]:
        for varg in varg_seq_a:
            varg['Description'] = " ".join(get_flatten_varg_toks(varg))
    if len(varg_seq_b) > 0 and not 'Description' in varg_seq_b[0]:
        for varg in varg_seq_b:
            varg['Description'] = " ".join(get_flatten_varg_toks(varg))

    # miximum weighted bipartite matching
    if len(varg_seq_a) > 0 and len(varg_seq_b) > 0:
        scores = [[0 for j in range(len(varg_seq_b))] for i in range(len(varg_seq_a))]
        for i in range(len(varg_seq_a)):
            for j in range(len(varg_seq_b)):
                e_sep_a = varg_seq_a[i]['EVENT_SEP']
                v_a = " ".join(varg_seq_a[i]['V_toks'])
                desc_a = varg_seq_a[i]['Description']
                e_sep_b = varg_seq_b[j]['EVENT_SEP']
                v_b = " ".join(varg_seq_b[j]['V_toks'])
                desc_b = varg_seq_b[j]['Description']
                scores[i][j] = float(e_sep_a == e_sep_b) * float(v_a == v_b) * rouge_scorer.score(desc_a, desc_b)['rougeLsum'].fmeasure
        rows, cols = linear_sum_assignment(scores, maximize=True)
        total_score = sum(scores[i][j] for i, j in zip(rows, cols)) / len(rows)
    else:
        rows, cols = [], []
        total_score = 0
    # build seq representations
    repr_a = list(range(len(varg_seq_a)))
    repr_b = list(range(len(varg_seq_a), len(varg_seq_a)+len(varg_seq_b)))
    for i, j in zip(rows, cols):
        if scores[i][j] > 0:
            repr_b[j] = repr_a[i]
    return repr_a, repr_b, total_score


@Model.register("event_lm_transformer_seq2seq")
class EventLMTransformerSeq2Seq(ModelBase):
    """
    transformer_name (str): the pretrained transformer to use.
    decode_kwargs (Dict[str, Any]): decoding arguments for ``generate`` function of ``seq2seq_generator``.
    tokenizer_kwargs (Dict[str, Any]): tokenizer arguments.
    """
    def __init__(self,
                 vocab: Vocabulary,
		 transformer_name: str,
                 seq2seq_generator: TransformerForConditionalGeneration,
                 label_smoothing: float = 0.0,
                 decode_kwargs: Optional[Dict[str, Any]] = None,
                 tokenizer_kwargs: Optional[Dict[str, Any]] = None,
                 dropout: float = 0.1,
                 extra_metrics: Dict[str, str]=None,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        super().__init__(vocab, regularizer)

        self._seq2seq_generator = seq2seq_generator

        self._label_smoothing = label_smoothing

        self._tokenizer = get_huggingface_tokenizer(transformer_name, **(tokenizer_kwargs or {}))

        if len(self._tokenizer) > self._seq2seq_generator._base_model.config.vocab_size:
            self._seq2seq_generator._base_model.resize_token_embeddings(len(self._tokenizer))
            logger.info("Resize embeddings from %d to %d",
                        self._seq2seq_generator._base_model.config.vocab_size,
                        len(self._tokenizer))

        self._decode_kwargs = decode_kwargs or {}

        self._dropout = dropout

        self._pad_token_id = self._seq2seq_generator._base_model.config.pad_token_id

        self._decoder_start_token_id = self._seq2seq_generator._base_model.config.decoder_start_token_id

        self._event_token_id = self._tokenizer.added_tokens_encoder[EVENT_TAG]

        self._pointer_event_token_ids = [self._tokenizer.added_tokens_encoder[tag] for tag in POINTER_EVENT_TAGS if tag in self._tokenizer.added_tokens_encoder]

        self._arg_token_id = self._tokenizer.added_tokens_encoder[ARGS_TAG]

        self._orig_special_ids = set(self._tokenizer.all_special_ids) - set(self._tokenizer.additional_special_tokens_ids)

        self._loss_trackers = {'loss': Average()}
        self._pairwise_metric = metric_map['chain_pairwise_accuracy']()
        self._desc_rouge = Average()
        if extra_metrics is not None:
            self._extra_metrics = {}
            for m_name, m_type in extra_metrics.items():
                self._extra_metrics[m_name] = metric_map[m_type]()
        else:
            self._extra_metrics = None

    def forward(self,  # type: ignore
                source_tok_ids:                     torch.LongTensor,
                target_tok_ids:                     torch.LongTensor = None,
                source_attention_mask:              torch.LongTensor = None,
                target_attention_mask:              torch.LongTensor = None,
                metadata:                           List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        '''
        print("question context tok ids")
        print(question_context_tok_ids[0])
        print(self._tokenizer.decode(question_context_tok_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=True))
        print("target tok ids")
        print(target_tok_ids[0])
        print(self._tokenizer.decode(target_tok_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=True))
        '''
        batch_size = source_tok_ids.size(0)

        output_dict = {}

        # compute loss and predictions under teacher forcing
        if target_tok_ids is not None:
            # create decoder inputs, need to add ``decoder_start_token_id``: in bart, it is </s>
            decoder_input_ids = target_tok_ids.new_zeros(target_tok_ids.shape)
            decoder_input_ids[..., 1:] = target_tok_ids[..., :-1].clone()
            decoder_input_ids[..., 0] = self._decoder_start_token_id
            decoder_attention_mask = target_attention_mask.new_zeros(target_attention_mask.shape)
            decoder_attention_mask[..., 1:] = target_attention_mask[..., :-1].clone()
            decoder_attention_mask[..., 0] = 1
            # create labels
            labels = target_tok_ids.clone().detach()
            labels[target_tok_ids == self._pad_token_id] = -100
            '''
            print("decoder_input_ids")
            print(decoder_input_ids[0])
            print("decoder attention mask")
            print(decoder_attention_mask[0])
            print("labels")
            print(labels[0])
            input()
            '''
            # loss, prediction_scores, cache, all_dec_hiddens, all_dec_attns, encoder_outputs; if exists
            seq2seq_outputs = self._seq2seq_generator(
                input_ids=source_tok_ids,
                attention_mask=source_attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                lm_labels=labels,
                label_smoothing=self._label_smoothing
            )
            loss = seq2seq_outputs[0]
            # shape: (batch_size, length, vocab_size)
            logits = seq2seq_outputs[1]

            # get teacher forcing prediction ids, (batch_size, length)
            tf_prediction_ids = torch.max(logits, dim=-1)[1]
            output_dict["tf_prediction_ids"] = tf_prediction_ids

            self._loss_trackers['loss'](loss)
            output_dict["loss"] = loss

            prediction_ids = tf_prediction_ids
            '''
            print("tf pred")
            print(self._tokenizer.decode(prediction_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=True))
            '''

        # decode the prediction
        if not self.training:
            # get decode prediction ids, (batch_size*num_return_sequences, length)
            decode_prediction_ids = self._seq2seq_generator.generate(
                input_ids=source_tok_ids,
                attention_mask=source_attention_mask,
                event_token_ids=[self._event_token_id]+self._pointer_event_token_ids,
                arg_token_id=self._arg_token_id,
                **self._decode_kwargs
            )
            # (batch_size, num_return_sequences, length)
            decode_prediction_ids = decode_prediction_ids.view(batch_size,
                                                               self._decode_kwargs.get("num_return_sequences", 1),
                                                               decode_prediction_ids.size(-1))
            output_dict["decode_prediction_ids"] = decode_prediction_ids

            prediction_ids = decode_prediction_ids

            if self._decode_kwargs.get('num_return_sequences', 1) > 1:
                prediction_ids = prediction_ids.view(batch_size, self._decode_kwargs['num_return_sequences'], prediction_ids.size(-1))

        # Compute the EM and F1 on SQuAD and add the textual prediction to the output.
        if metadata is not None:
            output_dict['prediction_str'] = []
            output_dict['prediction_varg_seq'] = []
            output_dict['gold_varg_seq'] = []
            if len(prediction_ids.size()) == 3:
                output_dict['beam_prediction_str'] = []
                output_dict['beam_prediction_varg_seq'] = []
            source_strs = []
            target_strs = []
            is_scrambleds = []
            varg_seqs = []
            keep_varg_ids = []
            ids = []
            doc_ids = []
            data_srcs = []
            for i in range(batch_size):
                source_strs.append(metadata[i]['source_str'])
                target_strs.append(metadata[i]['target_str'])
                is_scrambleds.append(metadata[i]['is_scrambled'])
                varg_seqs.append(metadata[i]['varg_seq'])
                keep_varg_ids.append(metadata[i]['keep_varg_ids'])
                ids.append(metadata[i]['_id'])
                doc_ids.append(metadata[i]['doc_id'])
                data_srcs.append(metadata[i]['data_src'])

                if len(prediction_ids.size()) == 2:
                    predicted_token_ids = prediction_ids[i].detach().cpu().numpy()
                elif len(prediction_ids.size()) == 3:
                    output_dict['beam_prediction_str'].append([])
                    output_dict['beam_prediction_varg_seq'].append([])
                    beam_size = prediction_ids.size(1)
                    for beam_idx in range(beam_size):
                        predicted_token_ids = prediction_ids[i, beam_idx].detach().cpu().numpy()
                        predicted_token_ids = [tok_id for tok_id in predicted_token_ids if not tok_id in self._orig_special_ids] # remove original special tokens
                        prediction_str = self._tokenizer.decode(predicted_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
                        prediction_varg_seq = V_ARGS_string_to_varg_seq(prediction_str)
                        output_dict['beam_prediction_str'][i].append(prediction_str)
                        output_dict['beam_prediction_varg_seq'][i].append(prediction_varg_seq)

                    predicted_token_ids = prediction_ids[i, 0].detach().cpu().numpy()
                predicted_token_ids = [tok_id for tok_id in predicted_token_ids if not tok_id in self._orig_special_ids] # remove original special tokens
                prediction_str = self._tokenizer.decode(predicted_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
                prediction_varg_seq = V_ARGS_string_to_varg_seq(prediction_str)
                output_dict['prediction_str'].append(prediction_str)
                output_dict['prediction_varg_seq'].append(prediction_varg_seq)

                gold_varg_seq = metadata[i].get('gold_varg_seq', None)
                output_dict['gold_varg_seq'].append(gold_varg_seq)
                if gold_varg_seq is not None:
                    pred_seq, gold_seq, matching_score = get_event_matching(prediction_varg_seq, gold_varg_seq)
                    self._pairwise_metric(pred_seq, gold_seq)
                    self._desc_rouge(matching_score)
                    if self._extra_metrics is not None:
                        for m_name, met in self._extra_metrics.items():
                            met(pred_seq, gold_seq)

                
                if i == 0:
                    print("source_str")
                    print(self._tokenizer.decode(source_tok_ids[0].detach().cpu().numpy(), skip_special_tokens=False, clean_up_tokenization_spaces=True))
                    print("target_str")
                    print(self._tokenizer.decode(target_tok_ids[0].detach().cpu().numpy(), skip_special_tokens=False, clean_up_tokenization_spaces=True))
                    
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
                    print("After get_event_matching:", pred_seq, "; GOLD:",gold_seq)
                1/0
                
            output_dict['source_str'] = source_strs
            output_dict['target_str'] = target_strs
            output_dict['is_scrambled'] = is_scrambleds
            output_dict['varg_seq'] = varg_seqs
            output_dict['keep_varg_ids'] = keep_varg_ids
            output_dict['_id'] = ids
            output_dict['doc_id'] = doc_ids
            output_dict['data_src'] = data_srcs

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        metrics['pairwise_acc'] = self._pairwise_metric.get_metric(reset)
        metrics['desc_rouge'] = self._desc_rouge.get_metric(reset)
        if self._extra_metrics is not None:
            for m_name, met in self._extra_metrics.items():
                score = met.get_metric(reset)
                if type(score) == dict:
                    for score_key in score:
                        metrics[m_name+'_'+score_key] = score[score_key]
                else:
                    metrics[m_name] = score
        for name, tracker in self._loss_trackers.items():
            metrics[name] = tracker.get_metric(reset).item()
        return metrics

    def compute_sequence_scores(self, instances: List[Instance]) -> List[Dict[str, np.ndarray]]:
        batch_size = len(instances)
        with torch.no_grad():
            cuda_device = self._get_prediction_device()
            dataset = Batch(instances)
            dataset.index_instances(self.vocab)
            model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)

            assert len(model_input) == 5
            source_tok_ids = model_input['source_tok_ids']
            target_tok_ids = model_input['target_tok_ids']
            source_attention_mask = model_input['source_attention_mask']
            target_attention_mask = model_input['target_attention_mask']
            metadata = model_input['metadata']

            output_dict = {}

            # compute loss under teacher forcing, which should be the sequence score
            # create decoder inputs, need to add ``decoder_start_token_id``: in bart, it is </s>
            decoder_input_ids = target_tok_ids.new_zeros(target_tok_ids.shape)
            decoder_input_ids[..., 1:] = target_tok_ids[..., :-1].clone()
            decoder_input_ids[..., 0] = self._decoder_start_token_id
            decoder_attention_mask = target_attention_mask.new_zeros(target_attention_mask.shape)
            decoder_attention_mask[..., 1:] = target_attention_mask[..., :-1].clone()
            decoder_attention_mask[..., 0] = 1
            # create labels
            labels = target_tok_ids.clone().detach()
            labels[target_tok_ids == self._pad_token_id] = -100
            # prediction_scores, cache, all_dec_hiddens, all_dec_attns, encoder_outputs; if exists
            seq2seq_outputs = self._seq2seq_generator(
                input_ids=source_tok_ids,
                attention_mask=source_attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                lm_labels=None, # don't compute loss,
                use_cache=False
            )
            # shape: (batch_size, length, vocab_size)
            logits = seq2seq_outputs[0]

            loss_fct = nn.CrossEntropyLoss(reduction='none')
            # shape: (batch_size*length,)
            label_len = labels.size(1)
            neg_logprob = loss_fct(logits.view(batch_size*label_len, self._seq2seq_generator._base_model.config.vocab_size),
                                   labels.view(batch_size*label_len))
            # shape: (batch_size,)
            seq_scores = -torch.sum(neg_logprob.view(batch_size, label_len), dim=-1)
            '''
            event_tags_mask = torch.any(labels.unsqueeze(-1) == labels.new_tensor([self._event_token_id]+self._pointer_event_token_ids), dim=-1)
            seq_scores = -torch.sum(neg_logprob.view(batch_size, label_len) * event_tags_mask.float(), dim=-1)
            '''

            # shape: (batch_size,)
            seq_len = torch.sum((labels != -100).float(), dim=-1)

            output_dict['seq_score'] = seq_scores / seq_len

            instance_separated_output: List[Dict[str, np.ndarray]] = [
                {} for _ in dataset.instances
            ]
            for name, output in list(output_dict.items()):
                if isinstance(output, torch.Tensor):
                    # NOTE(markn): This is a hack because 0-dim pytorch tensors are not iterable.
                    # This occurs with batch size 1, because we still want to include the loss in that case.
                    if output.dim() == 0:
                        output = output.unsqueeze(0)

                    if output.size(0) != batch_size:
                        self._maybe_warn_for_unseparable_batches(name)
                        continue
                    output = output.detach().cpu().numpy()
                elif len(output) != batch_size:
                    self._maybe_warn_for_unseparable_batches(name)
                    continue
                for instance_output, batch_element in zip(instance_separated_output, output):
                    instance_output[name] = batch_element
            return instance_separated_output

    def generate_extra_events(self, instances: List[Instance]) -> List[Dict[str, np.ndarray]]:
        batch_size = len(instances)
        # TODO: if batch_size > 1, need to modify the generate function to start decode from the shortest sequence
        assert batch_size == 1
        with torch.no_grad():
            cuda_device = self._get_prediction_device()
            dataset = Batch(instances)
            dataset.index_instances(self.vocab)
            model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)

            assert len(model_input) == 5
            source_tok_ids = model_input['source_tok_ids']
            target_tok_ids = model_input['target_tok_ids']
            source_attention_mask = model_input['source_attention_mask']
            target_attention_mask = model_input['target_attention_mask']
            metadata = model_input['metadata']
            override_decode_kwargs = metadata[0].get('override_decode_kwargs', {})

            decode_kwargs = {k: v for k, v in self._decode_kwargs.items()}
            decode_kwargs.update(override_decode_kwargs)

            if 'bad_verbs_ids' in metadata[0]:
                bad_verbs_ids = metadata[0]['bad_verbs_ids']
                ban_bad_verbs_event_idxs = metadata[0]['ban_bad_verbs_event_idxs']
            else:
                bad_verbs_ids = None
                ban_bad_verbs_event_idxs = None
            
            if 'target_suffix_encodes' in metadata[0]:
                target_suffix_ids = metadata[0]['target_suffix_encodes']['input_ids']
                target_suffix_start_event_idx = metadata[0]['target_suffix_start_event_idx']
            else:
                target_suffix_ids = None
                target_suffix_start_event_idx = None

            num_output_events = metadata[0]['num_output_events']

            output_dict = {}

            # create decoder inputs, need to add ``decoder_start_token_id``: in bart, it is </s>
            _, target_len = target_tok_ids.shape
            decoder_input_ids = target_tok_ids.new_zeros((batch_size, target_len+1))
            decoder_input_ids[..., 1:] = target_tok_ids.clone()
            decoder_input_ids[..., 0] = self._decoder_start_token_id
            # TODO: if batch_size > 1, need to pass decoder_attention_mask
            '''
            decoder_attention_mask = target_attention_mask.new_zeros((batch_size, target_len+1))
            decoder_attention_mask[..., 1:] = target_attention_mask.clone()
            decoder_attention_mask[..., 0] = 1
            '''
            # get decode prediction ids, (batch_size*num_return_sequences, length)
            decode_prediction_ids = self._seq2seq_generator.generate(
                input_ids=source_tok_ids,
                attention_mask=source_attention_mask,
                no_repeat_ngram_size=3,
                bad_verbs_ids=bad_verbs_ids,
                ban_bad_verbs_event_idxs=ban_bad_verbs_event_idxs,
                max_num_events=num_output_events,
                min_num_events=num_output_events,
                event_token_ids=[self._event_token_id]+self._pointer_event_token_ids,
                arg_token_id=self._arg_token_id,
                input_suffix_ids=target_suffix_ids,
                input_suffix_start_event_idx=target_suffix_start_event_idx,
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
                **decode_kwargs
            )
            # (batch_size, num_return_sequences, length)
            decode_prediction_ids = decode_prediction_ids.view(batch_size,
                                                               decode_kwargs.get("num_return_sequences", 1),
                                                               decode_prediction_ids.size(-1))
            output_dict["decode_prediction_ids"] = decode_prediction_ids

            prediction_ids = decode_prediction_ids

            if decode_kwargs.get('num_return_sequences', 1) > 1:
                prediction_ids = prediction_ids.view(batch_size, decode_kwargs['num_return_sequences'], prediction_ids.size(-1))

            if metadata is not None:
                output_dict['prediction_str'] = []
                output_dict['prediction_varg_seq'] = []
                output_dict['gold_varg_seq'] = []
                output_dict['input_varg_seq'] = []
                if len(prediction_ids.size()) == 3:
                    output_dict['beam_prediction_str'] = []
                    output_dict['beam_prediction_varg_seq'] = []
                for i in range(batch_size):
                    if len(prediction_ids.size()) == 2:
                        predicted_token_ids = prediction_ids[i].detach().cpu().numpy()
                    elif len(prediction_ids.size()) == 3:
                        output_dict['beam_prediction_str'].append([])
                        output_dict['beam_prediction_varg_seq'].append([])
                        beam_size = prediction_ids.size(1)
                        for beam_idx in range(beam_size):
                            predicted_token_ids = prediction_ids[i, beam_idx].detach().cpu().numpy()
                            predicted_token_ids = [tok_id for tok_id in predicted_token_ids if not tok_id in self._orig_special_ids] # remove original special tokens
                            prediction_str = self._tokenizer.decode(predicted_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
                            prediction_varg_seq = V_ARGS_string_to_varg_seq(prediction_str)
                            output_dict['beam_prediction_str'][i].append(prediction_str)
                            output_dict['beam_prediction_varg_seq'][i].append(prediction_varg_seq)

                        predicted_token_ids = prediction_ids[i, 0].detach().cpu().numpy()
                    predicted_token_ids = [tok_id for tok_id in predicted_token_ids if not tok_id in self._orig_special_ids] # remove original special tokens
                    prediction_str = self._tokenizer.decode(predicted_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
                    prediction_varg_seq = V_ARGS_string_to_varg_seq(prediction_str)
                    output_dict['prediction_str'].append(prediction_str)
                    output_dict['prediction_varg_seq'].append(prediction_varg_seq)

                    gold_varg_seq = metadata[i].get('gold_varg_seq', None)
                    output_dict['gold_varg_seq'].append(gold_varg_seq)

                    output_dict['input_varg_seq'].append(metadata[i]['input_varg_seq'])
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

            instance_separated_output: List[Dict[str, np.ndarray]] = [
                {} for _ in dataset.instances
            ]
            for name, output in list(output_dict.items()):
                if isinstance(output, torch.Tensor):
                    # NOTE(markn): This is a hack because 0-dim pytorch tensors are not iterable.
                    # This occurs with batch size 1, because we still want to include the loss in that case.
                    if output.dim() == 0:
                        output = output.unsqueeze(0)

                    if output.size(0) != batch_size:
                        self._maybe_warn_for_unseparable_batches(name)
                        continue
                    output = output.detach().cpu().numpy()
                elif len(output) != batch_size:
                    self._maybe_warn_for_unseparable_batches(name)
                    continue
                for instance_output, batch_element in zip(instance_separated_output, output):
                    instance_output[name] = batch_element
            return instance_separated_output
