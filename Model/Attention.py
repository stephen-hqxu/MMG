from Model import Setting

import torch
from torch import Tensor
from torch.nn import Module, Linear, Softmax

import math
from typing import Optional

class MultiHeadAttention(Module):
    """
    The classic MHA model from the original paper.
    """

    def __init__(this):
        super().__init__()
        # QKV linear projection
        projection_param = (Setting.EMBEDDED_FEATURE_SIZE, Setting.EMBEDDED_FEATURE_SIZE)
        this.QProjection: Linear = Linear(*projection_param)
        this.KProjection: Linear = Linear(*projection_param)
        this.VProjection: Linear = Linear(*projection_param)
        # concatenation of QKV values
        this.QKVJoint: Linear = Linear(*projection_param)

        this.Normaliser: Softmax = Softmax(dim = -1)

    def forward(this, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Input Q,K,V: (batch, sequence, embedded feature);
            need to flatten time and note axes because attention works on raw sequence,
            also feature is now at the end.
            In fact, the shape of the sequence doesn't matter at this point, because we have position embedding.
        Output: same
        """
        batch, sequence, feature = q.shape
        # linear projection
        q = this.QProjection(q)
        k = this.KProjection(k)
        v = this.VProjection(v)

        # head splitting
        def split(x: Tensor) -> Tensor:
            """
            Output: (batch, head, sequence, split feature)
            """
            # batch, sequence, head, split feature
            x = x.reshape(batch, sequence, Setting.ATTENTION_HEAD_COUNT, Setting.EMBEDDED_FEATURE_PER_HEAD)
            # swap so we are paying attention on sequence of features rather than individual head
            return x.swapaxes(1, 2)
        q = split(q)
        k = split(k)
        v = split(v)

        # scale dot product attention matrix and its score
        def calcSDPA() -> Tensor:
            # dot product Q with pow(K, T) to compute similarity
            k_t: Tensor = k.swapaxes(-1, -2)
            # scale the dot product by the split feature size
            score: Tensor = torch.matmul(q, k_t) / math.sqrt(Setting.EMBEDDED_FEATURE_PER_HEAD)

            # masking, give it a very low score to force to not pay any attention
            if mask is not None:
                score = score.masked_fill(mask == 0, -1e9)

            # normalisation so sum is [0, 1]
            score = this.Normaliser(score)

            # scale V by score
            return torch.matmul(score, v)
        similarity: Tensor = calcSDPA()

        # swap the axes back, concatenate heads and do a linear projection
        joint_head: Tensor = similarity.swapaxes(1, 2).reshape(batch, sequence, feature)
        return this.QKVJoint(joint_head)