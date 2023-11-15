from denoising_event_lm.training.metrics.squad_em_and_f1 import SquadEmAndF1
from denoising_event_lm.training.metrics.rouge import Rouge
from denoising_event_lm.training.metrics.chain_pairwise_accuracy import ChainPairwiseAccuracy
from denoising_event_lm.training.metrics.event_coverage import EventCoverage

metric_map = {
    "rouge": Rouge,
    "squad": SquadEmAndF1,
    "chain_pairwise_accuracy": ChainPairwiseAccuracy,
    "event_coverage": EventCoverage,
}
