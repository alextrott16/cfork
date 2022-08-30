# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput, BertSelfAttention, BertSelfOutput


# A replacement function should replace BertSelfAttention with BertSelfAttentionPre
class BertSelfAttentionPre(torch.nn.Module):
    """
    Places a layer norm before the input to BertSelfAttention, consistent with the Pre-LN architecture.

    Args:
        module (BertSelfAttention): The BertSelfAttention module this "replaces".
        layer_norm_eps (float, optional): The epsilon parameter for the layer norm, default ``1e-12``.
        normformer (bool, optional): Whether to implement this as a NormFormer, default ``False``.
    """

    def __init__(self, module: torch.nn.Module, layer_norm_eps: float = 1e-12, normformer: bool = False):
        super().__init__()
        assert isinstance(module, BertSelfAttention)
        self.normformer = normformer
        assert layer_norm_eps > 0

        self.LayerNorm = torch.nn.LayerNorm(module.query.in_features, eps=layer_norm_eps)
        self.attn = module

        # Some extra HeadScale parameters needed if we're doing full NormFormer
        if self.normformer:
            raise NotImplementedError

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # Layer norm on the inputs to self attention
        hidden_states = self.LayerNorm(hidden_states)
        attn_output = self.attn(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask,
                                past_key_value, output_attentions)
        # If full NormFormer, we need to do Head Scaling
        if self.normformer:
            raise NotImplementedError
        return attn_output


# Replacement function should replace BertIntermediate with BertIntermediatePre
class BertIntermediatePre(torch.nn.Module):
    """
    Places a layer norm before the input to BertIntermediate, consistent with the Pre-LN architecture.

    Args:
        module (BertIntermediate): The BertIntermediate module this "replaces".
        layer_norm_eps (float, optional): The epsilon parameter for the layer norm, default ``1e-12``.
        normformer (bool, optional): Whether to implement this as a NormFormer, default ``False``.
    """

    def __init__(self, module: torch.nn.Module, layer_norm_eps: float = 1e-12, normformer: bool = False):
        super().__init__()
        assert isinstance(module, BertIntermediate)
        self.normformer = normformer
        assert layer_norm_eps > 0
        hidden_size = module.dense.in_features
        intermediate_size = module.dense.out_features

        self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.intermediate = module

        # An extra layernorm needed if we're doing full NormFormer
        if self.normformer:
            self.extra_LayerNorm = torch.nn.LayerNorm(intermediate_size, eps=layer_norm_eps)
        else:
            self.extra_LayerNorm = None

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor]:
        # Layer norm on the inputs to intermediate
        hidden_states = self.LayerNorm(hidden_states)
        intermediate_output = self.intermediate(hidden_states)
        # If full NormFormer, we need to do another LN on intermediate_output
        if self.extra_LayerNorm is not None:
            intermediate_output = self.extra_LayerNorm(intermediate_output)
        return intermediate_output


# A replacement function should replace BertSelfOutput with BertSelfOutputPre
class BertSelfOutputPre(torch.nn.Module):
    """
    Removes the LayerNorm after the residual connection, consistent with the Pre-LN architecture.

    Args:
        module (BertSelfOutput): The BertSelfOutput module this replaces.
        layer_norm_eps (float, optional): The epsilon parameter for the layer norm, default ``1e-12``.
        normformer (bool, optional): Whether to implement this as a NormFormer, default ``False``.
    """

    def __init__(self, module: torch.nn.Module, layer_norm_eps: float = 1e-12, normformer: bool = False):
        super().__init__()
        assert isinstance(module, BertSelfOutput)
        self.normformer = normformer
        assert layer_norm_eps > 0
        hidden_size = module.dense.in_features
        dropout_prob = module.dropout.p

        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(dropout_prob)
        if self.normformer:
            self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        else:
            self.LayerNorm = None

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.LayerNorm is not None:
            hidden_states = self.LayerNorm(hidden_states)
        # No more LayerNorm after residual connection
        hidden_states = hidden_states + input_tensor
        return hidden_states


# A replacement function should replace BertOutput with BertOutputPre
class BertOutputPre(torch.nn.Module):
    """
    Removes the LayerNorm after the residual connection, consistent with the Pre-LN architecture.

    Args:
        module (BertOutput): The BertOutput module this replaces.
        layer_norm_eps (float, optional): The epsilon parameter for the layer norm, default ``1e-12``.
        normformer (bool, optional): Whether to implement this as a NormFormer, default ``False``.
        last_layer (bool, optional): Whether to treat this instance as the last transformer layer, default ``False``.
    """

    def __init__(self,
                 module: torch.nn.Module,
                 layer_norm_eps: float = 1e-12,
                 normformer: bool = False,
                 last_layer: bool = False):
        super().__init__()
        assert isinstance(module, BertOutput)
        self.normformer = normformer
        self.last_layer = last_layer
        assert layer_norm_eps > 0
        intermediate_size = module.dense.in_features
        hidden_size = module.dense.out_features
        dropout_prob = module.dropout.p

        self.dense = torch.nn.Linear(intermediate_size, hidden_size)
        self.dropout = torch.nn.Dropout(dropout_prob)

        if self.last_layer:
            self.LayerNorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        else:
            self.LayerNorm = None

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # No more LayerNorm after residual connection (unless this is the last layer)
        hidden_states = hidden_states + input_tensor
        if self.LayerNorm is not None:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
