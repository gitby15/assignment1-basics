import torch


# 0-1之间取值
# x = 1 / (1 + exp(x))
# 缺点是，它的倒数最大值是0.25， 层级深了，反向传播时容易梯度衰减，因为总要乘以一个小于0.25的值
def sigmoid(x: torch.Tensor) -> torch.Tensor:
    exp = torch.exp(-x)
    return 1 / (1 + exp)

# sigmoid的改版，在[-1, 1]之间取值
# x = (exp(x) - 1) / (exp(x) + 1)
# 优点：输出以0为中心，训练更稳定、倒数最大值是1，梯度消失会好一丢丢
# 缓解了sigmoid的问题，但是没有根治
def tanh(x: torch.Tensor) -> torch.Tensor:
    exp = torch.exp(x)
    return (exp - 1) / (exp + 1)


# max(0, x)
# 优点: 计算性能好、大于0时，倒数恒为1，不会梯度消失，在CNN大量使用
# 缺点：神经元死亡，如果某个值算下来小于0，倒数就一直是0了，就不更新了
def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(x, torch.zeros_like(x))

# x = x * sigmoid(x), relu和sigmod的结合体
# 在小于0的一小段里面，会存在一些负值，不至于下于0就等于0，目的是保持relu的优点的情况下，缓解神经元死亡
# 这个数全程可求导，relu在0点处不可求导
def silu(x: torch.Tensor) -> torch.Tensor:
    exp = torch.exp(-x)
    return x / (1 + exp)

# x = x * φ(x) | 图像跟silu很像，但是在负值有值的区间会更短
# φ(x) = 正态分布函数
def gelu(x: torch.Tensor) -> torch.Tensor:
    # Todo: 暂不实现
    return x

# Swish-Gated Linear Unit门控线性单元

def swiglu(x: torch.Tensor) -> torch.Tensor:
    # Todo: 暂不实现
    return x