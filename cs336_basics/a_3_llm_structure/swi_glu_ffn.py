import torch
from torch import nn
from cs336_basics.a_3_llm_structure.linear import Linear


class SwiGluFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # 分支1，跟silu一起组成门控
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        # 分支2，纯线性变换
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        # 分支1和分支2逐元素相乘，然后通过w3线性变化到d_mode维度
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        # 为了适应单元测试，这里w2作为汇总层

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = self.w1(x)
        silu_w1_x = w1_x * torch.sigmoid(w1_x)
        return self.w2(silu_w1_x * self.w3(x))
