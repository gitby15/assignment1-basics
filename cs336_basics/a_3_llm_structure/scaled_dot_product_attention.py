import torch
from cs336_basics.a_3_llm_structure.softmax import softmax
from einops import einsum

def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = q.shape[-1]

    score = q@k.transpose(-2, -1)
    score = score / torch.sqrt(torch.tensor(d_k, device=q.device, dtype=q.dtype))
    if mask is not None:        
        score = score.masked_fill(mask == False, float('-inf'))  # noqa: E712
    score = softmax(score, dim=-1)
    attention = einsum(score, v, '... t1 t2, ... t2 d -> ... t1 d')
    return attention
