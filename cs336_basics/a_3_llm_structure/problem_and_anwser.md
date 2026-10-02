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
[Deliver]
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
[Deliver]
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
[Deliver]:
完成了rms_norm，并通过测试
1. 里面是一个rms计算+权重缩放
2. rms就是求平方 -> 求均值 -> 开方 -> 加一个小数字防止0除
3. 用一个可训练的权重做缩放
4. x/self._rms(x) * self.weight

# Problem (positionwise_feedforward):  Implement the position-wise feed-forward network (2points)
Deliverable: Implement the SwiGLU feed-forward network, composed of a SiLU activation
function and a GLU.
Note: in this particular case, you should feel free to use torch.sigmoid in your implementation
for numerical stability.
You should set 𝑑ff to approximately 8/3 × 𝑑model in your implementation, while ensuring that the dimensionality of the inner feed-forward layer is a multiple of 64 to make good use of your hardware. To test your implementation against our provided tests, you will need to implement the test adapter at [adapters.run_swiglu] . Then, run uv run pytest -k test_swiglu to test your implementation.
[Deliver]:
Todo

# Problem (rope):  Implement RoPE (2 points)
Deliverable: Implement a class RotaryPositionalEmbedding that applies RoPE to the input tensor.
The following interface is recommended
```python
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) 
# Construct the RoPE module and create buffers if needed.
# - theta: float  Θ value for the RoPE
# - d_k: int  dimension of query and key vectors
# - max_seq_len: int  Maximum sequence length that will be input
# - device: torch.device | None = None  Device to store the buffer on
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor
# Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note that you should tolerate 𝑥 with an arbitrary number of batch dimensions. You should assume that the token positions are a tensor of shape (..., seq_len) specifying the token positions of 𝑥 along the sequence dimension.
```
You should use the token positions to slice your (possibly precomputed) cos and sin tensors along the sequence dimension.
To test your implementation, complete [adapters.run_rope] and make sure it passes uv run pytest -k test_rope.