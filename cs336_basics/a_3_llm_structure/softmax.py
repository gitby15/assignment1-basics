import torch


def softmax(x: torch.Tensor, dim) -> torch.Tensor:
    return torch.exp(x) / torch.exp(x).sum(dim=dim, keepdim=True)