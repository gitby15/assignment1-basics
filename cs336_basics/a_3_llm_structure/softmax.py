import torch

# 这个实现在数学公式上是对的，但是没有处理数值溢出的问题
def softmax_origin(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.exp(x) / torch.exp(x).sum(dim=dim, keepdim=True)



# 利用e的两个特性：
# 1. e^a/e^b = e^(a-b)
# 2. 当a<0时，0 < e^a < 1
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)