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
[Deliver:]
Todo

# Problem (scaled_dot_product_attention):  Implement scaled dot-product attention (5points)
Deliverable: Implement the scaled dot-product attention function. Your implementation should handle keys and queries of shape (batch_size, ..., seq_len, d_k) and values of shape (batch_size, ..., seq_len, d_v), where ... represents any number of other batch-like dimensions (if provided). The implementation should return an output with the shape (batch_size, ..., seq_len, d_v). See Section 3.2 for a discussion on batch-like dimensions.
Your implementation should also support an optional user-provided boolean mask of shape (seq_len, seq_len). The attention probabilities of positions with a mask value of True should collectively sum to 1, and the attention probabilities of positions with a mask value of False should be zero

To test your implementation against our provided tests, you will need to implement the test adapter at [adapters.run_scaled_dot_product_attention] . uv run pytest -k
test_scaled_dot_product_attention tests your implementation on third-order input tensors, while `uv run pytest -k test_4d_scaled_dot_product_attention` tests your implementation on fourthorder input tensors.
[Deliver]:
手写了一个attention计算模块，现在还只支持传统的注意力计算，对于GQA、以及其他注意力机制还没支持

# Problem (multihead_self_attention):  Implement causal multi-head self-attention (5 points)
Deliverable: Implement causal multi-head self-attention as a torch.nn.Module. Your implementation should accept (at least) the following parameters:
- d_model: int Dimensionality of the Transformer block inputs.
- num_heads: int Number of heads to use in multi-head self-attention.
Following [A. Vaswani et al.][8], set 𝑑𝑘 = 𝑑𝑣 = 𝑑model/ℎ. To test your implementation against our provided tests, implement the test adapter at [adapters.run_multihead_self_attention] . Then, run `uv run pytest -k test_multihead_self_attention` to test your implementation.
[Deliver]:
手写了MHA，因为之前做过，所以比较顺利，token_positions应该是为了后面kv_cache设计的
我觉得后面可能要回来分析性能，这里对QKV都做了形状变换，后面需要测试一下，这里的形状变化后，只是调整了步长和shape，还是在内存中做了重新排列

# Problem (transformer_block):  Implement the Transformer block (3 points)
Implement the pre-norm Transformer block as described in Section 3.4 and illustrated in Figure 2.
Your Transformer block should accept (at least) the following parameters.
- `d_model`: int Dimensionality of the Transformer block inputs.
- `num_heads`: int Number of heads to use in multi-head self-attention.
- `d_ff`: int Dimensionality of the position-wise feed-forward inner layer.
To test your implementation, implement the adapter [adapters.run_transformer_block] . Then
run `uv run pytest -k test_transformer_block` to test your implementation.
Deliverable: Transformer block code that passes the provided tests.
[Deliver]:
完成了Transformer Block，把前面结果组件拼接到一起，残差链接 + 归一化 + 激活函数 + FFN

# Problem (transformer_lm):  Implementing the Transformer LM (3 points)
Time to put it all together! Implement the Transformer language model as described in Section 3.1 and illustrated in Figure 1. At minimum, your implementation should accept all the aforementioned construction parameters for the Transformer block, as well as these additional
parameters:
- `vocab_size`: int The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix.
- `context_length`: int The maximum context length, necessary for determining the dimensionality of the RoPE sin and cos buffer.
- `num_layers`: int The number of Transformer blocks to use.
To test your implementation against our provided tests, you will first need to implement the test
adapter at [adapters.run_transformer_lm] . Then, run `uv run pytest -k test_transformer_lm` to
test your implementation.
Deliverable: A Transformer LM module that passes the above tests
[Deliver]:
把所有零件都拼起来



# Problem (transformer_accounting):  Transformer LM resource accounting (5 points)
> Rule: Given 𝐴 ∈ ℝ(𝑚×𝑛) and 𝐵 ∈ ℝ(𝑛×𝑝), the matrix-matrix product 𝐴𝐵 requires 2𝑚𝑛𝑝 FLOPs.
(a)Consider a GPT-2 XL-sized model using our assignment architecture, which has the following configuration:
- `vocab_size`:  50,257
- `context_length`:  1,024
- `num_layers`:  48
- `d_model`:  1,600
- `num_heads`:  25
- `d_ff`:  4,288 (the nearest multiple of 64 to 8/3 × 1, 600)
Suppose we constructed our model using this configuration. How many trainable parameters
would our model have? Assuming each parameter is represented using single-precision floating
point, how much memory is required to just load this model?
Deliverable: A one-to-two sentence response.

(b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has context_length tokens.
Deliverable: A list of matrix multiplies (with descriptions), and the total number of FLOPs
required.

(c) Based on your analysis above, which parts of the model require the most FLOPs?
Deliverable: A one-to-two sentence response.

(d) Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium
(24 layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As
the model size increases, which parts of the Transformer LM take up proportionally more or
less of the total FLOPs?
Deliverable: For each model, provide a breakdown of model components and its associated
FLOPs (as a proportion of the total FLOPs required for a forward pass). In addition, provide
a one-to-two sentence description of how varying the model size changes the proportional
FLOPs of each component.

(e) Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How does the relative contribution of FLOPs of the model components
change?
Deliverable: A one-to-two sentence response.