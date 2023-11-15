from typing import Optional, List, Any

from itertools import combinations
from overrides import overrides
import torch
from torch.nn import functional as F

from allennlp.training.metrics.metric import Metric


def compute_f1(pred_set, gold_set):
    tp = len(pred_set & gold_set)
    prec = tp / (len(pred_set) + 1e-13)
    recl = tp / (len(gold_set) + 1e-13)
    f1 = 2 * prec * recl / (prec + recl + 1e-13)
    return f1, prec, recl


@Metric.register("event_coverage")
class EventCoverage(Metric):
    """
    Checks the event coverage in a given chain with repsect to the label.
    Each event should be representation by an integer of a string.
    It may happen that the length of the predicted chain is shorter than that of the labeled chain (like during validation).
    If the provided predictions contains multiple chains (such as in beamsearch), takes the chain with maximum F1.
    """
    def __init__(self) -> None:
        self._total_prec = 0.
        self._total_recl = 0.
        self._total_f1 = 0.
        self._total_count = 0.

    def __call__(self, pred_seqs: List[Any], gold_seq: List[Any]):
        """
        Parameters
        ----------
        pred_seqs : ``List[Any]``, required.
            prediction sequences. Could be List[List[Any]] for beamsearch.
        gold_seq : ``List[Any]``, required.
            gold sequences. Assume that two identical objects in prediction and gold are the same event.
        """

        if len(pred_seqs) == 0 or not type(pred_seqs[0]) == list:
            # add extra beam dimension
            pred_seqs = [pred_seqs]
        if len(gold_seq) > 0:
            e_type = type(gold_seq[0])
            assert e_type == int or e_type == str
            assert all(type(e) == e_type for seq in pred_seqs for e in seq)
            assert all(type(e) == e_type for e in gold_seq)

        gold_events = set(gold_seq)
        f1, prec, recl = max(compute_f1(set(seq), gold_events) for seq in pred_seqs)

        self._total_prec += prec
        self._total_recl += recl
        self._total_f1 += f1
        self._total_count += 1
        #print("pairwise")
        #print(pair_correct, pair_cnt)

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        prec = float(self._total_prec) / float(self._total_count + 1e-13)
        recl = float(self._total_recl) / float(self._total_count + 1e-13)
        f1 = float(self._total_f1) / float(self._total_count + 1e-13)
        if reset:
            self.reset()
        return {'prec': prec, 'recl': recl, 'f1': f1}

    @overrides
    def reset(self):
        self._total_prec = 0.0
        self._total_recl = 0.0
        self._total_f1 = 0.0
        self._total_count = 0.0
