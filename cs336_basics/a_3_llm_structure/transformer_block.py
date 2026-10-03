import torch
from torch import nn
from cs336_basics.a_3_llm_structure.rope import Rope
from cs336_basics.a_3_llm_structure.mh_attention import Causal_MHA
from cs336_basics.a_3_llm_structure.rmsnorm import RMSNorm
from cs336_basics.a_3_llm_structure.swi_glu_ffn import SwiGluFFN


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: Rope | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope = rope
        self.mha = Causal_MHA(d_model, num_heads, rope)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGluFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # x 的形状是 (..., seq_len, d_model)
        normalized_x = self.ln1(x)
        attention = x + self.mha(
            normalized_x,
            normalized_x,
            normalized_x,
            token_positions,
        )
        return attention + self.ffn(self.ln2(attention))
