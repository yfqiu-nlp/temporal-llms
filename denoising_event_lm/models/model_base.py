import os
from os import PathLike
import numpy as np
from typing import Any, Dict, List, Optional, Union
import re
import logging
import torch

from allennlp.common.params import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model, _DEFAULT_WEIGHTS, remove_pretrained_embedding_params
from allennlp.nn import util, RegularizerApplicator


logger = logging.getLogger(__name__)


@Model.register("model_base")
class ModelBase(Model):
    @classmethod
    def _load(
        cls,
        config: Params,
        serialization_dir: Union[str, PathLike],
        weights_file: Optional[Union[str, PathLike]] = None,
        cuda_device: int = -1,
    ) -> "Model":
        """
        Instantiates an already-trained model, based on the experiment
        configuration and some optional overrides.
        """
        weights_file = weights_file or os.path.join(serialization_dir, _DEFAULT_WEIGHTS)

        # Load vocabulary from file
        vocab_dir = os.path.join(serialization_dir, "vocabulary")
        # If the config specifies a vocabulary subclass, we need to use it.
        vocab_params = config.get("vocabulary", Params({}))
        vocab_choice = vocab_params.pop_choice("type", Vocabulary.list_available(), True)
        vocab_class, _ = Vocabulary.resolve_class_name(vocab_choice)
        vocab = vocab_class.from_files(
            vocab_dir, vocab_params.get("padding_token"), vocab_params.get("oov_token")
        )

        model_params = config.get("model")

        # The experiment config tells us how to _train_ a model, including where to get pre-trained
        # embeddings from.  We're now _loading_ the model, so those embeddings will already be
        # stored in our weights.  We don't need any pretrained weight file anymore, and we don't
        # want the code to look for it, so we remove it from the parameters here.
        remove_pretrained_embedding_params(model_params)
        # The '_pretrained' keys in the config is also not needed here, and will cause infinite loop on beaker.
        # So we also remove this key from the parameters here.
        # However, for this to work, note the parameters for a pretrained module should be well-specified in the config.
        remove_pretrained_modules_params(model_params)
        model = Model.from_params(vocab=vocab, params=model_params)

        # Force model to cpu or gpu, as appropriate, to make sure that the embeddings are
        # in sync with the weights
        if cuda_device >= 0:
            model.cuda(cuda_device)
        else:
            model.cpu()

        # If vocab+embedding extension was done, the model initialized from from_params
        # and one defined by state dict in weights_file might not have same embedding shapes.
        # Eg. when model embedder module was transferred along with vocab extension, the
        # initialized embedding weight shape would be smaller than one in the state_dict.
        # So calling model embedding extension is required before load_state_dict.
        # If vocab and model embeddings are in sync, following would be just a no-op.
        model.extend_embedder_vocab()

        model_state = torch.load(weights_file, map_location=util.device_mapping(cuda_device))
        model.load_state_dict(model_state)

        return model

def remove_pretrained_modules_params(params: Params):
    if isinstance(params, Params):  # The model could possibly be a string, for example.
        keys = params.keys()
        if "_pretrained" in keys:
            del params["_pretrained"]
        if "modules_from_pretrained" in keys:
            del params["modules_from_pretrained"]
        for value in params.values():
            if isinstance(value, Params):
                remove_pretrained_modules_params(value)
