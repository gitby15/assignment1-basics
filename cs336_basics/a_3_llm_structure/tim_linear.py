import torch
from torch import nn
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        # 讲义3.2.1中, 要求使用列向量法，因为这样更方便数学解释
        # Y = W@x
        self.weight = nn.Parameter(torch.randn(in_features, out_features, device=device, dtype=dtype))
        print(self.weight.shape)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入的形状是[out_features, in_feature]
        # return einsum(x, self.weight, 'i o, i o->i o')
        
        # return x @ self.weight.T
        return einsum(x, self.weight, '... in_f, out_f in_f -> ... out_f')


if __name__ == "__main__":
    linear = Linear(3, 4)
    x = torch.randn(3, 5)
    print(linear(x))