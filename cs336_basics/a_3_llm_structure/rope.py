import torch
from torch import nn


class Rope(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
