import math

import torch
from torch import nn
from einops import einsum

class Linear(nn.Module):
    """
    测试接口要求如下：
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        # 讲义3.2.1中, 要求使用列向量法，因为这样更方便数学解释
        # weight形状要求[d_out, d_in]
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, device=device, dtype=dtype))
        self._checked_weight_shape = self.weight.shape
        
        _theta = 2 / (in_features + out_features)
        _theta_sqrt = math.sqrt(_theta)
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=_theta_sqrt, a = -3 * _theta_sqrt, b = 3 * _theta_sqrt)
        
    
    # 在讲义中，数学的计算公式，希望是Y = W@x
    # 但是x的形状会是(..., d_in)，所以实际上会变成 Y = x @ W.T
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self._checked_weight_shape == self.weight.shape
        return einsum(self.weight, x, 'out_f in_f, ... in_f -> ... out_f')


if __name__ == "__main__":
    d_in = 5
    d_out = 7
    linear = Linear(d_in, d_out)
    x = torch.randn(d_out, d_in)
    print(linear(x))
