import torch
from torch import nn

from cs336_basics.a_3_llm_structure.embedding import Embed
from cs336_basics.a_3_llm_structure.linear import Linear
from cs336_basics.a_3_llm_structure.rmsnorm import RMSNorm
from cs336_basics.a_3_llm_structure.rope import Rope
from cs336_basics.a_3_llm_structure.transformer_block import TransformerBlock


class TransfomerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.embedding = Embed(vocab_size, d_model)
        self.rope = Rope(rope_theta, d_model // num_heads, context_length)
        self.mha_block = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    rope=self.rope,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x 的形状是 (..., seq_len, d_model)
        x = self.embedding(x)
        for block in self.mha_block:
            x = block(x, token_positions)
        return self.lm_head(self.ln_final(x))
