from typing import Optional, List, Any

from itertools import combinations
from overrides import overrides
import torch
from torch.nn import functional as F

from allennlp.training.metrics.metric import Metric
from allennlp.nn.util import get_range_vector, get_device_of


def get_before_pairs(seq):
    pairs = set((i, j) for i, j in combinations(seq, 2))
    return pairs


def pairwise_accuracy(pred_seq, gold_seq):
    """ pairwise accuracy
    """
    gold_before_pairs = get_before_pairs(gold_seq) # in a chain, there are only `before` relations
    pred_before_pairs = get_before_pairs(pred_seq)
    corr = len(gold_before_pairs & pred_before_pairs)
    pair_acc = corr / len(gold_before_pairs)
    return pair_acc


@Metric.register("chain_pairwise_accuracy")
class ChainPairwiseAccuracy(Metric):
    """
    Checks the correctness of event prediction for each step in a given chain with repsect to the label.
    Each event should be representation by an integer of a string.
    It may happen that the length of the predicted chain is shorter than that of the labeled chain (like during validation).
    If the provided predictions contains multiple chains (such as in beamsearch), takes the chain with maximum true positive.
    """
    def __init__(self) -> None:
        self._total_acc = 0.
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
        e_type = type(gold_seq[0])
        assert e_type == int or e_type == str
        assert all(type(e) == e_type for seq in pred_seqs for e in seq)
        assert all(type(e) == e_type for e in gold_seq)

        pair_acc = max(pairwise_accuracy(seq, gold_seq) for seq in pred_seqs)

        self._total_acc += pair_acc
        self._total_count += 1
        #print("pairwise")
        #print(pair_correct, pair_cnt)

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        accuracy = float(self._total_acc) / float(self._total_count + 1e-13)
        if reset:
            self.reset()
        return accuracy

    @overrides
    def reset(self):
        self._total_acc = 0.0
        self._total_count = 0.0
