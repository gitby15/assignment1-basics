import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # RMSNorm的初始权重就设置成1
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        temp = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        return torch.sqrt(temp)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x/self._rms(x) * self.weight