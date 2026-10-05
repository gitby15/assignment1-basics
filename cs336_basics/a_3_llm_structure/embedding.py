import torch
from torch import nn

# Embed本质上是查表
# 所以，猜测Embed实际上是一个CPU操作，在推理的时候，可以不送入GPU运行？
class Embed(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # self.weight = (V, C) | 输入的结构是(B, T)
        # (V, C)[B, T] => (B, T, C) => 我对pytorch还是不够熟练，有时间要整体学习一下
        return self.weight[token_ids]
        
        
        