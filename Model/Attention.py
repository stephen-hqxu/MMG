from Model import Setting
from Model.Setting import DropoutSetting

import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from typing import Optional

class MultiHeadAttention(Module):
    """
    The classic MHA model from the original paper, with improvement such as flash attention and causal attention masking.
    """

    def __init__(this):
        super().__init__()
        this.Attention: nn.MultiheadAttention = nn.MultiheadAttention(Setting.EMBEDDED_FEATURE_SIZE, Setting.ATTENTION_HEAD_COUNT,
            dropout = DropoutSetting.MULTIHEAD_ATTENTION, batch_first = True)

    def forward(this, q: Tensor, k: Tensor, v: Tensor,
            padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None, causal: bool = False) -> Tensor:
        """
        Input Q,K,V: (batch, sequence, embedded feature);
            need to flatten time and note axes because attention works on raw sequence,
            also feature is now at the end.
            In fact, the shape of the sequence doesn't matter at this point, because we have position embedding.
        Output: same
        """
        # TODO: return attention output weight if we want to visualise the attention matrix
        attn_output, _ = this.Attention(q, k, v, key_padding_mask = padding_mask, need_weights = False, attn_mask = attn_mask, is_causal = causal)
        return attn_output