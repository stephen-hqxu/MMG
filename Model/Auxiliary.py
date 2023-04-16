import torch
from torch import Tensor

def makeNoPeekMask(len_q: int, len_k: int) -> Tensor:
    """
    @brief Create mask that prevents paying attention to the sequence beyond the current input.

    @param len_q, len_k The sequence length of Q and K.
    @return The boolean mask, with size (len_q, len_k)
    """
    mask: Tensor = torch.ones((len_q, len_k), dtype = torch.bool)
    # create triangular matrix
    return torch.tril(mask, out = mask)