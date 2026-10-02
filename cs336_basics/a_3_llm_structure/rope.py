import torch
from torch import nn

# rope属于静态计算，可以在LLM中使用一个单独的实例，来优化性能
# self.register_buffer(persistent=False)
class Rope(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        _rope_group = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        # 长度是d_k / 2
        _freq = 1.0 / theta ** (_rope_group / d_k)

        # 长度是seq_len
        _positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        # 长度是(seq_len, d_k / 2)
        _freq_x = torch.outer(_positions, _freq)

        self.cos = torch.cos(_freq_x)
        self.sin = torch.sin(_freq_x)



    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x的形状是(..., seq_len, d_k)
        # 根据 token position 取对应的 cos / sin
        # (..., seq_len, d_k / 2)
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # 偶数维 / 奇数维
        # (..., seq_len, d_k / 2)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # RoPE rotation
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        # 重新交错
        out = torch.stack(
            [out_even, out_odd],
            dim=-1,
        )

        # (..., seq_len, d_k)
        return out.flatten(-2)
