# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional, Sequence, Type, Union

import torch

from composer.models.huggingface import HuggingFaceModel

try:
    from transformers import BertPreTrainedModel
    from transformers.models.bert.configuration_bert import BertConfig

    from composer.algorithms.pre_layernorm.pre_layernorm_layers import (BERTGatedFFOutput, BertGatedFFOutputPre,
                                                                        BertIntermediate, BertIntermediatePre,
                                                                        BertOutput, BertOutputPre, BertSelfAttention,
                                                                        BertSelfAttentionPre, BertSelfOutput,
                                                                        BertSelfOutputPre)
    IS_TRANSFORMERS_INSTALLED = True
except ImportError as e:
    IS_TRANSFORMERS_INSTALLED = False

# from composer.algorithms.gated_linear_units.gated_linear_unit_layers import BERTGatedFFOutput
from composer.algorithms.warnings import NoEffectWarning
from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import MissingConditionalImportError, module_surgery

log = logging.getLogger(__name__)


def apply_pre_layernorm(model: torch.nn.Module,
                        optimizers: Optional[Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]]] = None,
                        head_scale: bool = False,
                        attn_output_layernorm: bool = False,
                        ffn_layernorm: bool = False) -> None:
    """
    Modifies the location of the LayerNorm operations to convert a Post-LN model to a Pre-LN model.

    Note:
        This can be used to further convert to the `NormFormer <http://arxiv.org/abs/2110.09456>` version of the Pre-LN architecture.
        To do so, set ``head_scale``, ``post_attn_layernorm``, and ``ffn_layernorm`` to ``True``. Each of these activate
        components of the NormFormer, but, if desired, this implementation allows any combination of them to be used.

    Args:
        model (`torch.nn.Module`): The model to modify in-place.
        optimizers (`torch.optim.Optimizer` | Sequence[`torch.optim.Optimizer`], optional):
            Existing optimizers bound to ``model.parameters()``. All optimizers that have already been
            constructed with ``model.parameters()`` must be specified here so that
            they will optimize the correct parameters.

            If the optimizer(s) are constructed after calling this function,
            then it is safe to omit this parameter. These optimizers will see the correct
            model parameters.
        head_scale (bool, optional): Whether to apply head scaling, which is part of NormFormer, default ``False``.
        attn_output_layernorm (bool, optional): Whether to apply a LayerNorm after doing attention, which is part of NormFormer, default ``False``.
        ffn_layernorm (bool, optional): Whether to apply a LayerNorm after the FFN, which is part of NormFormer, default ``False``.
    """
    if not IS_TRANSFORMERS_INSTALLED:
        raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers')

    unwrapped_model = model.model if isinstance(model, HuggingFaceModel) else model

    # ensure that the model is an instance of a Hugging Face BertPreTrainedModel class, since our replacement policy is only defined for BERTs
    if not isinstance(unwrapped_model, BertPreTrainedModel):
        raise TypeError(
            'Pre-LayerNorm only has a surgery policy defined for subclasses of transformers.BertPreTrainedModel')

    # prepare the replacement policy and perform replacement
    if not hasattr(model, 'config'):
        raise TypeError('Bert config must be accessible through model.config')
    if not hasattr(model.config, 'layer_norm_eps'):
        raise TypeError('Bert config must include "layer_norm_eps" field')
    if not hasattr(model.config, 'num_hidden_layers'):
        raise TypeError('Bert config must include "num_hidden_layers" field')
    assert isinstance(model.config, BertConfig)
    layer_norm_eps = getattr(model.config, 'layer_norm_eps')
    num_hidden_layers = getattr(model.config, 'num_hidden_layers')
    assert isinstance(layer_norm_eps, float)
    assert isinstance(num_hidden_layers, int)
    
    def from_BertSelfAttention(module: torch.nn.Module, idx: int):
        if hasattr(module, 'pre_ln_applied'):
            if module.pre_ln_applied:
                return # Skip surgery because it has already been used to wrap `module`
        return BertSelfAttentionPre(module, layer_norm_eps=layer_norm_eps, head_scale=head_scale)

    def from_BertIntermediate(module: torch.nn.Module, idx: int):
        if hasattr(module, 'pre_ln_applied'):
            if module.pre_ln_applied:
                return # Skip surgery because it has already been used to wrap `module`
        return BertIntermediatePre(module, layer_norm_eps=layer_norm_eps, ffn_layernorm=ffn_layernorm)

    policy: Dict[Type[torch.nn.Module], module_surgery.ReplacementFunction] = {
        BertSelfAttention: from_BertSelfAttention,
        BertIntermediate: from_BertIntermediate,
        BertSelfOutput:
            lambda module, idx: BertSelfOutputPre(
                module, layer_norm_eps=layer_norm_eps, attn_output_layernorm=attn_output_layernorm),
        BertOutput:
            lambda module, idx: BertOutputPre(
                module, layer_norm_eps=layer_norm_eps, last_layer=bool(idx + 1 == num_hidden_layers)),
        BERTGatedFFOutput:
            lambda module, idx: BertGatedFFOutputPre(module,
                                                     layer_norm_eps=layer_norm_eps,
                                                     ffn_layernorm=ffn_layernorm,
                                                     last_layer=bool(idx + 1 == num_hidden_layers)),
    }
    replaced_instances = module_surgery.replace_module_classes(module=model, optimizers=optimizers, policies=policy)
    if len(replaced_instances) == 0:
        warnings.warn(
            NoEffectWarning(
                'No instances of the supported BERT modules were found, and therefore, there were no modules to replace.'
            ))
    log.info(f'Successfully replaced {len(replaced_instances)} modules with their Pre-LN version.')


class PreLayerNorm(Algorithm):
    """Converts the BERT transformer model from a "Post-LN" architecture to a "Pre-LN" (Pre-LayerNorm) architecture.
    The Pre-LN transformer improves training stability and enables more aggressive hyperparameters.

    Runs on :attr:`.Event.INIT`, so it can perform the necessary model surgery.

    Note:
        This can be used to further convert to the `NormFormer <http://arxiv.org/abs/2110.09456>` version of the Pre-LN architecture.
        To do so, set ``head_scale``, ``post_attn_layernorm``, and ``ffn_layernorm`` to ``True``. Each of these activate
        components of the NormFormer, but, if desired, this implementation allows any combination of them to be used.

    Args:
        head_scale (bool, optional): Whether to apply head scaling, which is part of NormFormer, default ``False``.
        attn_output_layernorm (bool, optional): Whether to apply a LayerNorm after doing attention, which is part of NormFormer, default ``False``.
        ffn_layernorm (bool, optional): Whether to apply a LayerNorm after the FFN, which is part of NormFormer, default ``False``.

    Example:
        .. testsetup::

           model, train_dataloader, optimizer = _make_synthetic_bert_state()

        .. testcode::

           from composer.algorithms import PreLayerNorm

           algorithm = PreLayerNorm()
           trainer = Trainer(
               model=model,
               train_dataloader=train_dataloader,
               max_duration="1ep",
               algorithms=[algorithm],
               optimizers=[optimizer]
           )
    """

    def __init__(self, head_scale: bool = False, attn_output_layernorm: bool = False, ffn_layernorm: bool = False):
        if not IS_TRANSFORMERS_INSTALLED:
            raise MissingConditionalImportError(extra_deps_group='nlp', conda_package='transformers')

        self.head_scale = head_scale
        self.attn_output_layernorm = attn_output_layernorm
        self.ffn_layernorm = ffn_layernorm

    def match(self, event: Event, state: State) -> bool:
        del state  # unused
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        del event, logger  # unused
        apply_pre_layernorm(model=state.model,
                            optimizers=state.optimizers,
                            head_scale=self.head_scale,
                            attn_output_layernorm=self.attn_output_layernorm,
                            ffn_layernorm=self.ffn_layernorm)
