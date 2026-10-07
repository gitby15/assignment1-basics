import torch
from torch import nn
from cs336_basics.a_3_llm_structure.linear import Linear
from cs336_basics.a_3_llm_structure.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.a_3_llm_structure.rope import Rope


class Causal_MHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: Rope | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "头数需要能够被整除"
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.rope = rope

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # q 的形状是 (..., seq_len, d_model)
        # FLOPs = 6BT * C^2 = 6BTC^2
        # 内存占用：新q/k/v一共3 * 4BTC，旧的q/k/v，如果都是同一份数据，就只有4BTC, 另外，还有4C^2(weight)
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        query_len = q.shape[-2]
        key_len = k.shape[-2]
        # 内存占用：4BT(mask)
        causal_mask = torch.ones(
            query_len, key_len, device=q.device, dtype=torch.bool
        ).tril()

        # [B, T, C（H*D）] -> [B, T, H, D] -> [B, H, T, D]
        # 这里不需要参与浮点数运算
        q = q.unflatten(-1, (self.num_heads, self.d_k)).transpose(-2, -3)
        k = k.unflatten(-1, (self.num_heads, self.d_k)).transpose(-2, -3)
        v = v.unflatten(-1, (self.num_heads, self.d_v)).transpose(-2, -3)

        # token_positions的形状是：(B, seq_len)
        if self.rope is not None:
            # prefill场景
            if token_positions is None:
                token_positions = torch.arange(q.shape[-2], device=q.device)
            # decode场景
            elif token_positions.ndim == q.ndim - 2:
                token_positions = token_positions.unsqueeze(-2)

            # rope的Flops：3BTC，这里就是(3 * 2) * B * (token_positions_len) * C
            # 在prefill的时候是6B(T_q + T_k)C，在Decode的时候，大约是6BC
            # 简化计算复杂度，我们假设QKV的T相等，所以Flops = 6BTC
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # 注意力计算的Flops大约是4BTTC => 4BT^2C
        # 内存占用
        attention = scaled_dot_product_attention(q, k, v, causal_mask)
        attention = attention.transpose(-3, -2).flatten(start_dim=-2, end_dim=-1)
        # 线性层Flops = 2BTC^2
        # 所以总的Flops = 6BTCC + 6BTC + 4BTTC + 2BTCC ≈ 8BTCC + 6BTC + 4BTTC
        return self.output_proj(attention)
