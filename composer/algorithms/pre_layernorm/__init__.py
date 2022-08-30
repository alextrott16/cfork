# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Converts the BERT transformer model from a "Post-LN" architecture to a "Pre-LN" (Pre-LayerNorm) architecture.

The Pre-LN transformer improves training stability and enables more aggressive hyperparameters.

See the :doc:`Method Card </method_cards/pre_layernorm>` for more details.
"""

from composer.algorithms.pre_layernorm.pre_layernorm import PreLayerNorm, apply_pre_layernorm

__all__ = ['PreLayerNorm', 'apply_pre_layernorm']
