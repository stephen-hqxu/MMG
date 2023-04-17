from Model.Setting import EmbeddingSetting, TransformerSetting, DropoutSetting

import Model.Auxiliary as Aux

from torch import Tensor
from torch.nn import Module, TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer

from typing import List, TypeVar

L = TypeVar("L")
def createCoderLayer(layer_t: L) -> L:
    """
    @brief Create a coder layer.
    
    @tparam layer_t The type of layer.
    :return The coder layer instance.
    """
    return layer_t(EmbeddingSetting.EMBEDDED_FEATURE_SIZE,
        TransformerSetting.ATTENTION_HEAD_COUNT, TransformerSetting.FEED_FORWARD_LATENT_SIZE, DropoutSetting.CODER, batch_first = True)

L = TypeVar("L")
C = TypeVar("C")
def createCoder(layer_t: L, coder_t: C) -> C:
    """
    @brief Create a coder.

    @tparam layer_t The type of layer.
    @tparam coder_t The type of coder.
    @return The coder instance.
    """
    return coder_t(createCoderLayer(layer_t), TransformerSetting.CODER_LAYER_COUNT)

class Encoder(Module):
    """
    The encoder of the transformer.
    """

    def __init__(this):
        super().__init__()
        this.EncoderBlock: TransformerEncoder = createCoder(TransformerEncoderLayer, TransformerEncoder)

    def forward(this, x: Tensor) -> Tensor:
        """
        Input: (batch, sequence, embedded feature);
            need to flatten time window and note axes because attention works on raw sequence,
            also feature is now at the end.
            In fact, the shape of the sequence doesn't matter at this point, because we have position embedding.
        Output: same
        """
        # Typically, the encoder attention should use a padding mask to ignore any padded input.
        # In our model, the input is a latent vector (recall dimensionality reduction during embedding),
        # such that we can't really do any padding manually, but letting the model to learn about the it.
        # Instead, we use causal self-attention.
        return this.EncoderBlock(x, is_causal = True)
    
class Decoder(Module):
    """
    The decoder for the transformer.
    """

    def __init__(this):
        super().__init__()
        # not using the built-in decoder because it doesn't allow using causal attention (for some reasons)
        this.DecoderLayer: List[TransformerDecoderLayer] = [createCoderLayer(TransformerDecoderLayer) for _ in range(TransformerSetting.CODER_LAYER_COUNT)]

    def forward(this, dec_input: Tensor, enc_output: Tensor) -> Tensor:
        """
        Input: same as encoder layer.
        Output: also the same.

        @param enc_output Obviously, the output from running the entire encoder.
        """
        sequenceLength: int = dec_input.size(1)
        # create output mask for the decoder attention
        outputMask: Tensor = Aux.makeNoPeekMask(sequenceLength, sequenceLength)

        output: Tensor = dec_input
        for dec in this.DecoderLayer:
            # same situation as in the encoder, we don't have padding mask
            output = dec(output, enc_output, tgt_mask = outputMask, memory_is_causal = True)
        return output