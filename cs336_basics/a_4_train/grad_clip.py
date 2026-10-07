# 在训练的过程中，有时候会产生很大的梯度，如果直接把这个梯度扔进优化器，可以这一个step的步子会迈太大
# 梯度裁剪技术，在实践中用于环节这个问题
# 核心思想是，完成反向传播后，在优化器更新前，对梯度的数值设置一个上限（最大能迈多大步）
# 假设所有参数的梯度，组成一个整体梯度 g。 首先计算L2范数：||g||2 - 也就是平方和开根号
# 如果||g||2 > max_norm，那么就对g进行缩放，使||g||2 = max_norm
# 然后对每个参数的梯度，都乘以max_norm/||g||2
from collections.abc import Iterable

import torch
def grad_clip(parameters: Iterable[torch.nn.Parameter], max_norm=1.0, eps=1e-6):
    parameters = list(parameters)

    total_norm_sq = 0.0

    # 计算所有参数梯度平方和，再开根号，就是L2范数（几何大小）
    # L1范数是绝对值相加
    # L♾️范数，就是取最大值
    # 吐槽一下，为啥要起这么高大上的名字
    for p in parameters:
        if p.grad is not None:
            total_norm_sq += p.grad.pow(2).sum().item() # .sum()是对所有维度的所有元素求和

    total_norm = total_norm_sq ** 0.5

    # 如果超过 max_norm，统一缩放
    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps) # 加eps是为了避免除0，因为这些算子的算数强度都很小，所以不怕浪费

        for p in parameters:
            if p.grad is not None:
                p.grad *= scale