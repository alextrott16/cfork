# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput, BertSelfAttention, BertSelfOutput

from composer.algorithms.gated_linear_units.gated_linear_unit_layers import BERTGatedFFOutput


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

        self.n_heads = module.num_attention_heads

        # # Some extra HeadScale parameters needed if we're doing full NormFormer
        # if self.normformer:
        #     self.head_scales = torch.nn.Parameter(torch.ones(1, 1, self.n_heads, 1), requires_grad=True)
        # else:
        #     self.head_scales = None

    def _apply_head_scaling(self, attn_output):
        """Normformer's head scaling op."""
        assert self.normformer and self.head_scales is not None
        hidden_states = attn_output[0]  # size = [batch_size, seq_length, emb_dim]
        orig_shape = hidden_states.size()

        # De-concatenate the head-specific outputs
        new_shape = orig_shape[:-1] + (self.n_heads, -1)
        hidden_states = hidden_states.view(new_shape)  # size = [batch_size, seq_length, n_heads, emb_dim/n_heads]

        # Scale each head's attention output
        hidden_states = hidden_states * self.head_scales  # size = [batch_size, seq_length, n_heads, emb_dim/n_heads]

        # Re-concatenate the attention outputs
        hidden_states = hidden_states.view(orig_shape)  # size = [batch_size, seq_length, emb_dim]

        attn_output = (hidden_states,) + attn_output[1:]
        return attn_output

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
        # # If full NormFormer, we need to do Head Scaling
        # if self.normformer:
        #     attn_output = self._apply_head_scaling(attn_output)
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

        # An extra LayerNorm needed if we're doing full NormFormer
        if self.normformer:
            self.ffn_layernorm = torch.nn.LayerNorm(intermediate_size, eps=layer_norm_eps)
        else:
            self.ffn_layernorm = None

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor]:
        # Layer norm on the inputs to intermediate
        hidden_states = self.LayerNorm(hidden_states)
        intermediate_output = self.intermediate(hidden_states)
        # If full NormFormer, we need to do another LayerNorm on intermediate_output
        if self.ffn_layernorm is not None:
            intermediate_output = self.ffn_layernorm(intermediate_output)
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
        if self.normformer:
            self.layernorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        else:
            self.layernorm = None
        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        if self.layernorm is not None:
            # NormFormer places a LayerNorm before the attention module's residual connection
            hidden_states = self.layernorm(hidden_states)
        hidden_states = self.dropout(hidden_states)
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


# A replacement function should replace BertGatedFFOutput with BertGatedFFOutputPre
class BertGatedFFOutputPre(torch.nn.Module):
    """
    Reconfigures the GLU layer norms to be consistent with the Pre-LN architecture.

    Note: If this layer is present, it has likely replaced the BertIntermediate and BertOutput pair.

    Args:
        module (BertOutput): The BertGatedFFOutput module this replaces.
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
        assert isinstance(module, BERTGatedFFOutput)
        self.normformer = normformer
        self.last_layer = last_layer
        assert layer_norm_eps > 0

        hidden_size = module.gated_layer.in_features
        intermediate_size = module.gated_layer.out_features
        gated_layer_bias = module.gated_layer.bias is not None
        non_gated_layer_bias = module.non_gated_layer.bias is not None
        dropout_rate = module.dropout.p

        self.layernorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.gated_layer = torch.nn.Linear(hidden_size, intermediate_size, bias=gated_layer_bias)
        self.non_gated_layer = torch.nn.Linear(hidden_size, intermediate_size, bias=non_gated_layer_bias)
        self.act = module.act
        if self.normformer:
            self.ffn_layernorm = torch.nn.LayerNorm(intermediate_size, eps=layer_norm_eps)
        else:
            self.ffn_layernorm = None
        self.wo = torch.nn.Linear(intermediate_size, hidden_size)
        self.dropout = torch.nn.Dropout(dropout_rate)
        if self.last_layer:
            self.final_layernorm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        else:
            self.final_layernorm = None

    def forward(self, hidden_states: torch.Tensor, residual_connection: torch.Tensor):
        """
        Args:
            hidden_states (torch.Tensor): The hidden states from the attention matrix.
            residual_connection (torch.Tensor): The residual connection to add before the LayerNorm operator.
        """
        # pre-layernorm
        hidden_states = self.layernorm(hidden_states)
        # compute the activation
        hidden_states = self.act(self.gated_layer(hidden_states)) * self.non_gated_layer(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.ffn_layernorm is not None:
            # NormFormer adds a layernorm before the last FC layer (although NormFormer is not exactly defined for GLUs)
            hidden_states = self.ffn_layernorm(hidden_states)
        # multiply by the second matrix
        hidden_states = self.wo(hidden_states)
        # add the residual connection and post-LN
        hidden_states = hidden_states + residual_connection
        if self.final_layernorm is not None:
            # pre-ln architectures follow the final residual connection with a layernorm
            hidden_states = self.final_layernorm(hidden_states)
        return hidden_states
