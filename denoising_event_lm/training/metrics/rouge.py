from typing import Tuple

from allennlp.training.metrics.metric import Metric
from overrides import overrides

#from pycocoevalcap.rouge.rouge import Rouge as Rouge_scorer
from rouge_score import rouge_scorer


@Metric.register("rouge")
class Rouge(Metric):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computed exact match and F1 score using the official SQuAD
    evaluation script.
    """

    def __init__(self) -> None:
        #self._rouge = Rouge_scorer()
        self._rouge = scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
        self._total_r_score = 0.0
        self._count = 0

    @overrides
    def __call__(self, best_span_string, answer_strings):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        #best_span_string = best_span_string.strip().lower()
        #answer_strings = [ans.strip().lower() for ans in answer_strings]
        #wtd = {"dummy": answer_strings}
        #wrd = {"dummy": [best_span_string]}
        #r_score, _ = self._rouge.compute_score(wtd, wrd)
        best_span_string = best_span_string.replace(". ", " .\n")
        answer_strings = [ans.replace(". ", " .\n") for ans in answer_strings]
        r_score = max(self._rouge.score(best_span_string, ans)['rougeLsum'].fmeasure for ans in answer_strings)
        self._total_r_score += r_score
        self._count += 1

    @overrides
    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Returns
        -------
        Average Rouge-L score
        over all inputs.
        """
        r_score = self._total_r_score / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return r_score

    @overrides
    def reset(self):
        self._total_r_score = 0.0
        self._count = 0

    def __str__(self):
        return f"Rouge(score={self._total_r_score})"
