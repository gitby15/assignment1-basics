import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # RMSNorm的初始权重就设置成1
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    # rms(x) = 求平方 -> 计算均值 -> 开根号 + eps
    # x的形状是(B, T, C)
    # FLOPs
    # 开平方：BTC， 求平均值：BTC， 开根号：BT(因为求完均值，x的形状只剩BT了)、+ep: BT
    # (2c+2) * BT 约等于 BTC
    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        # 输入形状(B, T, C)
        # 输出形状(B, T)
        temp = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        return torch.sqrt(temp)
    
    # 总FLOPs = 2BTC + 2BTC = 4BTC
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 约等于2BTC
        return x/self._rms(x) * self.weight



# Todo: 计算了一把FLOPs和峰值内存后，对性能优化有了一些感觉
# 大概能感受到，算子融合是干啥的了
# 一方面，是真实推理的时候，不需要考虑代码可读性，减少中间变量，减少内存搬运，gpu内存的启动和销毁
# 另一方面，在数学公式上，可以做一些等值替换