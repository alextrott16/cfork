# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import pytest

from composer.algorithms.warnings import NoEffectWarning
from composer.algorithms.pre_layernorm import PreLayerNorm, apply_pre_layernorm
from composer.core.event import Event
from composer.loggers import Logger
from tests.fixtures.synthetic_hf_state import make_dataset_configs, synthetic_hf_state_maker


@pytest.fixture()
def synthetic_bert_state():
    synthetic_config = make_dataset_configs(model_family=['bert'])[0]
    return synthetic_hf_state_maker(synthetic_config)


def assert_is_pre_layernorm_instance(model):
    pytest.importorskip('transformers')
    from composer.algorithms.pre_layernorm.pre_layernorm_layers import (BertIntermediatePre,
                                                                        BertOutput, BertOutputPre,
                                                                        BertSelfAttentionPre, BertSelfOutput,
                                                                        BertSelfOutputPre)

    # ensure that within the entire model, no BertOutput exists, and at least one BERTGatedFFOutput does.
    assert model.modules is not None, 'model has .modules method'
    replacement_counts = {BertSelfAttentionPre: 0, BertSelfOutputPre: 0, BertIntermediatePre: 0, BertOutputPre: 0}
    for module_class in model.modules():
        assert not isinstance(
            module_class, BertSelfOutput
        ), 'A transformers.models.bert.modeling_bert.BertSelfOutput should not be found in the model after surgery is applied.'
        assert not isinstance(
            module_class, BertOutput
        ), 'A transformers.models.bert.modeling_bert.BertOutput should not be found in the model after surgery is applied.'

        for cls, count in replacement_counts.items():
            if isinstance(module_class, cls):
                replacement_counts[cls] = count + 1

    n_bert_self_attention_pre = replacement_counts[BertSelfAttentionPre]
    assert n_bert_self_attention_pre > 0, 'After surgery, there should be at least one BertSelfAttentionPre module.'

    for cls, count in replacement_counts.items():
        assert count == n_bert_self_attention_pre, f'Number of {cls} modules should be the same as number of BertSelfAttentionPre modules.'


@pytest.mark.xfail(raises=NoEffectWarning)
@pytest.mark.filterwarnings('error::composer.algorithms.warnings.NoEffectWarning')
def test_effect_and_no_effect_triggered(synthetic_bert_state: Tuple):
    """Test that PreLayerNorm has an effect when first applied but not on a second application."""
    pytest.importorskip('transformers')
    state, _, _ = synthetic_bert_state
    # Apply it once, should NOT see a NoEffectWarning
    apply_pre_layernorm(state.model, state.optimizers)
    
    # Apply it again, should see a NoEffectWarning (triggering the expected test failure)
    apply_pre_layernorm(state.model, state.optimizers)
    

def test_pre_layernorm_functional(synthetic_bert_state: Tuple):
    state, _, _ = synthetic_bert_state
    apply_pre_layernorm(state.model, state.optimizers)
    assert_is_pre_layernorm_instance(state.model)


def test_pre_layernorm_algorithm(synthetic_bert_state: Tuple, empty_logger: Logger):
    state, _, _ = synthetic_bert_state
    pre_layernorm = PreLayerNorm()
    pre_layernorm.apply(Event.INIT, state, empty_logger)
    assert_is_pre_layernorm_instance(state.model)


def test_normformer_algorithm(synthetic_bert_state: Tuple, empty_logger: Logger):
    state, _, _ = synthetic_bert_state
    pre_layernorm = PreLayerNorm(head_scale=True, attn_output_layernorm=True, ffn_layernorm=True)
    pre_layernorm.apply(Event.INIT, state, empty_logger)
    assert_is_pre_layernorm_instance(state.model)
