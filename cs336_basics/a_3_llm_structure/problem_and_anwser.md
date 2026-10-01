# Assignment 3. LLM Structure
## Problem (linear):  Implementing the linear module (1 point)
Deliverable: Implement a Linear class that inherits from torch.nn.Module and performs a linear transformation. Your implementation should follow the interface of PyTorch’s built-in nn.Linear module,except for not having a bias argument or parameter. We recommend the following interface
```python
def __init__(self, in_features, out_features, device=None, dtype=None)
"""
Construct a linear transformation module. This function should accept the following parameters:
- in_features: int  final dimension of the input 
- out_features: int  final dimension of the output
- device: torch.device | None = None  Device to store the parameters on
- dtype: torch.dtype | None = None  Data type of the parameters
"""
def forward(self, x: torch.Tensor) -> torch.Tensor
"""
Apply the linear transformation to the input
"""
```
完成了Linear的实现：
1. 