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
1. 接受列向量的表示方法
2. 权重初始化，N(mean, std) => mean = 0, std = sqrt(2 / (in_features + out_features))
3. 前向传播，Y = x @ W.T
通过测试

# Problem (embedding):  Implement the embedding module (1 point)
Deliverable: Implement the Embedding class that inherits from torch.nn.Module and performs an embedding lookup. Your implementation should follow the interface of PyTorch’s built-in nn.Embedding module. We recommend the following interface:
```python
def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None)
# Construct an embedding module. This function should accept the following parameters:
# - num_embeddings: int  Size of the vocabulary
# - embedding_dim: int  Dimension of the embedding vectors, i.e., 𝑑model
# - device: torch.device | None = None  Device to store the parameters on
# - dtype: torch.dtype | None = None  Data type of the parameters
def forward(self, token_ids: torch.Tensor) -> torch.Tensor
# Lookup the embedding vectors for the given token IDs.
```
Make sure to:
• subclass nn.Module
• call the superclass constructor
• initialize your embedding matrix as an nn.Parameter
• store the embedding matrix with the d_model being the final dimension
• of course, don’t use nn.Embedding or nn.functional.embedding
Again, use the settings from above for initialization, and use torch.nn.init.trunc_normal_ to initialize the weights.
To test your implementation, implement the test adapter at [adapters.run_embedding] . Then, run
```bash
uv run pytest -k test_embedding.
```
完成了Embedding的实现
1. 本质上是一个查表动作
2. 完成了初始化
3. 是一个重CPU的操作

# Problem (rmsnorm):  Root Mean Square Layer Normalization (1 point)
Deliverable: Implement RMSNorm as a torch.nn.Module. We recommend the following interface:
```python
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)
# Construct the RMSNorm module. This function should accept the following parameters:
# - d_model: int  Hidden dimension of the model
# - eps: float = 1e-5  Epsilon value for numerical stability
# - device: torch.device | None = None  Device to store the parameters on
# - dtype: torch.dtype | None = None  Data type of the parameters
def forward(self, x: torch.Tensor) -> torch.Tensor
# Process an input tensor of shape(batch_size, sequence_length, d_model) and return a tensor of the same shape.
```
Note: Remember to upcast your input to torch.float32 before performing the normalization(and later downcast to the original dtype), as described above.
To test your implementation, implement the test adapter at [adapters.run_rmsnorm] . Then, run uv run pytest -k test_rmsnorm.