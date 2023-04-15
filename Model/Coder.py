from Model import Setting

import Model.Attention as Atn
from Model.Attention import MultiHeadAttention
from Model.Auxiliary import FeedForward, Residual

from torch import Tensor
from torch.nn import Module

from typing import List

class Encoder(Module):
    """
    The encoder of the transformer.
    """

    class Layer(Module):
        """
        One encoder layer.
        """

        def __init__(this):
            super().__init__()
            this.Attention: Residual[MultiHeadAttention] = Residual(MultiHeadAttention())
            this.MLP: Residual[FeedForward] = Residual(FeedForward())

        def forward(this, x: Tensor) -> Tensor:
            """
            Input: same as attention.
            Output: same as feed-forward network.
            """
            # Typically, the encoder attention should use a padding mask to ignore any padded input.
            # In our model, the input is a latent vector (recall dimensionality reduction during embedding),
            # such that we can't really do any padding manually, but letting the model to learn about the it.
            # Instead, we use causal self-attention.
            x = this.Attention(x, x, x, x, causal = True) # Q, K, V are identical from the input for encoder
            return this.MLP(x, x)

    def __init__(this):
        super().__init__()
        this.EncoderLayer: List[Encoder.Layer] = [Encoder.Layer()] * Setting.CODER_LAYER_COUNT

    def forward(this, x: Tensor) -> Tensor:
        for enc in this.EncoderLayer:
            x = enc(x)
        return x
    
class Decoder(Module):
    """
    The decoder for the transformer.
    """

    class Layer(Module):
        """
        One decoder layer.
        """

        def __init__(this):
            super().__init__()
            this.MaskedAttention: Residual[MultiHeadAttention] = Residual(MultiHeadAttention())
            this.Attention: Residual[MultiHeadAttention] = Residual(MultiHeadAttention())
            this.MLP: Residual(FeedForward) = Residual(FeedForward())

        def forward(this, dec_input: Tensor, enc_output: Tensor) -> Tensor:
            """
            Input: same as encoder layer.
            Output: also the same.

            Encoder output, obviously, the output from running the entire encoder.
            """
            sequenceLength: int = dec_input.size(1)
            # create output mask for the decoder attention
            outputMask: Tensor = Atn.makeNoPeekMask(sequenceLength, sequenceLength)

            # Q, K, V are identical here as well
            x: Tensor = this.MaskedAttention(dec_input, dec_input, dec_input, dec_input, attn_mask = outputMask)
            # Q, K are from the encoder output this time
            # same situation as in the encoder, we don't have padding mask
            x = this.Attention(x, x, enc_output, enc_output, causal = True)
            return this.MLP(x, x)
        
    def __init__(this):
        super().__init__()
        this.DecoderLayer: List[Decoder.Layer] = [Decoder.Layer()] * Setting.CODER_LAYER_COUNT

    def forward(this, dec_input: Tensor, enc_output: Tensor) -> Tensor:
        x: Tensor = dec_input
        for dec in this.DecoderLayer:
            x = dec(x, enc_output)
        return x