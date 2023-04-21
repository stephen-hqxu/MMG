from Model.Setting import EmbeddingSetting, TransformerSetting, DropoutSetting

from torch import Tensor
from torch.nn import Module, TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer

from dataclasses import dataclass
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

@dataclass
class CoderMask:
    """
    @brief Masks used for calculating attention in encoder/decoder.
    To reduce the number of total memory allocation, it's recommended to preallocate a large array;
    the coder implementation will slice the array based on the current size of input.

    A mask can be none to indicate a mask is not used.

    Regarding the shape and size of each array, consult PyTorch documentation on transformer.
    """
    SourcePadding: Tensor = None
    # we don't need to mask out attention for source

    TargetPadding: Tensor = None
    TargetAttention: Tensor = None

class Encoder(Module):
    """
    The encoder of the transformer.
    """

    def __init__(this):
        super().__init__()
        this.EncoderBlock: TransformerEncoder = createCoder(TransformerEncoderLayer, TransformerEncoder)

    def forward(this, x: Tensor, mask: CoderMask) -> Tensor:
        """
        Input: (batch, sequence, embedded feature);
            need to flatten time window and note axes because attention works on raw sequence,
            also feature is now at the end.
            In fact, the shape of the sequence doesn't matter at this point, because we have position embedding.
        Output: same
        """
        sourceLength: int = x.size(1)
        srcPad: Tensor = mask.SourcePadding[:, :sourceLength] if mask.SourcePadding is not None else None

        return this.EncoderBlock(x, src_key_padding_mask = srcPad, is_causal = TransformerSetting.CAUSAL_ATTENTION_MASK)
    
class Decoder(Module):
    """
    The decoder for the transformer.
    """

    def __init__(this):
        super().__init__()
        # not using the built-in decoder because it doesn't allow using causal attention (for some reasons)
        this.DecoderLayer: List[TransformerDecoderLayer] = [createCoderLayer(TransformerDecoderLayer) for _ in range(TransformerSetting.CODER_LAYER_COUNT)]

    def forward(this, dec_input: Tensor, enc_output: Tensor, mask: CoderMask) -> Tensor:
        """
        Input: same as encoder layer.
        Output: also the same.

        @param enc_output Obviously, the output from running the entire encoder.
        """
        sourceLength: int = enc_output.size(1) # sequence length of source and memory is the same
        targetLength: int = dec_input.size(1)
        
        srcPad: Tensor = mask.SourcePadding[:, :sourceLength] if mask.SourcePadding is not None else None
        tgtPad: Tensor = mask.TargetPadding[:, :targetLength] if mask.TargetPadding is not None else None
        tgtMask: Tensor = mask.TargetAttention[:targetLength, :targetLength] if mask.TargetAttention is not None else None

        output: Tensor = dec_input
        for dec in this.DecoderLayer:
            output = dec(output, enc_output, tgt_mask = tgtMask,
                tgt_key_padding_mask = tgtPad, memory_key_padding_mask = srcPad,
                tgt_is_causal = TransformerSetting.CAUSAL_ATTENTION_MASK, memory_is_causal = TransformerSetting.CAUSAL_ATTENTION_MASK)
        return output