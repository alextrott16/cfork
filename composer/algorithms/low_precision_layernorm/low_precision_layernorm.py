# Copyright 2022 MosaicML Agent authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML. All Rights Reserved.

from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional, Sequence, Type, Union

import torch
import torch.nn.functional as F

from composer.algorithms.warnings import NoEffectWarning
from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import module_surgery

log = logging.getLogger(__name__)


def _cast_if_autocast_enabled(hidden_states, force_fp16=False):
    if not torch.is_autocast_enabled():
        return hidden_states
    else:
        if force_fp16:
            return torch.cuda.amp.autocast_mode._cast(hidden_states, torch.float16) # <-- Use this instead to hardcode to fp16
        else:
            return torch.cuda.amp.autocast_mode._cast(hidden_states, torch.get_autocast_gpu_dtype())  


class CastLayerNorm(torch.nn.LayerNorm):

    def __init__(self, normalized_shape, eps, elementwise_affine, force_fp16=False):
        super().__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.force_fp16 = bool(force_fp16)

    def forward(self, x):
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x, self.force_fp16)
        downcast_weight = _cast_if_autocast_enabled(self.weight, self.force_fp16)
        downcast_bias = _cast_if_autocast_enabled(self.bias, self.force_fp16)
        with torch.autocast(enabled=False, device_type=module_device.type):
            return F.layer_norm(downcast_x, self.normalized_shape, downcast_weight, downcast_bias, self.eps)


def from_Layer(layer: torch.nn.LayerNorm, idx: int, force_fp16: bool) -> CastLayerNorm:
    del idx
    return CastLayerNorm(layer.normalized_shape, layer.eps, layer.elementwise_affine, force_fp16=force_fp16)


def apply_low_precision_layernorm(model, optimizers: Union[torch.optim.Optimizer, Sequence[torch.optim.Optimizer]], force_fp16=False):

    policy: Dict[Type[torch.nn.Module], module_surgery.ReplacementFunction] = {
        torch.nn.LayerNorm: lambda module, idx: from_Layer(module, idx, force_fp16),
    }

    replaced_instances = module_surgery.replace_module_classes(module=model, optimizers=optimizers, policies=policy)
    if len(replaced_instances) == 0:
        warnings.warn(NoEffectWarning('No instances of torch.nn.LayerNorm found.'))
    log.info(f'Successfully replaced {len(replaced_instances)} instances of LayerNorm')

    return model


class LowPrecisionLayerNorm(Algorithm):
    """Replaces torch.nn.LayerNorm instances with an autocast-enabled version.

    Runs on :attr:`.Event.INIT`, so it can perform the necessary model surgery.

    Args:
        force_fp16 (bool, optional): Whether to force this module to use fp16 precision, default ``False``.
    """

    def __init__(self, force_fp16: bool = False):
        self.force_fp16 = force_fp16
        # # LowPrecisionLayerNorm takes no arguments
        # pass

    @property
    def find_unused_parameters(self) -> bool:
        # return True  # <-- Nikhil's original code
        return False

    def match(self, event: Event, state: State) -> bool:
        del state  # unused
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Logger) -> Optional[int]:
        del event, logger  # unused
        apply_low_precision_layernorm(model=state.model, optimizers=state.optimizers, force_fp16=self.force_fp16)