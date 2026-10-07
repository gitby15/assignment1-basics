import torch

# 这个实现在数学公式上是对的，但是没有处理数值溢出的问题
def softmax_origin(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.exp(x) / torch.exp(x).sum(dim=dim, keepdim=True)


'''
算数强度计算：
需要搬运的内存数据：
计算量：假设参数量是N，那么一次Softmax的Flops是5N
但是这里做了大量的数据搬运（产生了很多次中间变量的生成）
所以需要做算子融合，减少数据搬运的次数，回头让我研究一下这个算子融合要怎么做
'''



# 利用e的两个特性来防止上溢出：
# 1. e^a/e^b = e^(a-b)
# 2. 当a<0时，0 < e^a < 1
# 内存占用：BTC
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Flops：假设形状是BTC，dim=-1，那么有BT组，每组做C-1次比较，就是BTC
    # 临时内存 = BT
    x_temp = x.max(dim=dim, keepdim=True).values
    # Flops: 减法计算了BTC次，exp做了BCT次指数运算，所以总和是2BTC
    # 临时内存 = 2BTC
    x_temp = torch.exp(x - x_temp)
    # Flops：Sum做了BTC次求和，除法做了BTC次运算
    # 所以softmax的Flops = 5BTC
    # 临时内存 = BTC
    # 所以峰值内存是4bytes * 3BTC
    return x_temp / x_temp.sum(dim=dim, keepdim=True)


# 现在懂了，为什么在计算的时候，大家喜欢复用变量名，是为了尽早释放运算过程中的中间结果，节省显存
# 在推理的时候，这样干是对的，但在训练的时候，torch的autograd会保存中间结果，所以部分内存就没有被释放干净