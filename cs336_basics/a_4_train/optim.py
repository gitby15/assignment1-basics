# 直接学习AdamW，有点摸不着头脑，索性从SGD开始，把各种优化器都实现一遍
from collections.abc import Iterator

import torch


# θ(t+1) = θ(t) - lr * gt
# gt = ∇L(θ(t);Bt) = ∂L/∂θ(t)
# L就是损失函数
class SGD:
    def __init__(self, params: Iterator[torch.Tensor], lr=1e-3):
        assert lr > 0, f"学习率需要大于0, 现在是:{lr}"
        self.lr = lr
        self.params = list(params)

    def step(self):
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                # 更新参数，p.grad是参数执行backward的时候自动计算的
                p -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# θ(t+1) = θ(t) - v(t)
# v(t) = momentum * v(t-1) + lr * gt
# gt = ∇L(θ(t);Bt) = ∂L/∂θ(t)
class Momentum:
    def __init__(self, params: Iterator[torch.Tensor], lr=1e-3, momentum=0.9):
        assert lr > 0, f"学习率需要大于0, 现在是:{lr}"
        assert momentum >= 0 and momentum < 1, f"动量因子需要在0到1之间, 现在是:{momentum}"
        self.lr = lr
        self.momentum = momentum
        self.params = list(params)
        self.last_v = {}

    def step(self):
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                # 更新参数，p.grad是参数执行backward的时候自动计算的
                v_t = self.momentum * self.last_v.get(p, 0) + self.lr * p.grad
                self.last_v[p] = v_t
                p -= v_t

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# RMSProp引入了“自适应学习率”的概念
# θ(t+1) = θ(t) - m(t)
# m(t) = [lr / (sqrt(v(t)) + ε)] * g(t)
# v(t) = β*v(t-1) + (1-β)*g(t)^2
# gt = ∇L(θ(t);Bt) = ∂L/∂θ(t)
class RMSprop:
    def __init__(self, params: Iterator[torch.Tensor], lr=1e-3, beta=0.999, eps=1e-8):
        assert lr > 0, f"学习率需要大于0, 现在是:{lr}"
        assert beta >= 0 and beta < 1, f"动量因子需要在0到1之间, 现在是:{beta}"
        assert eps > 0, f"小数因子需要大于0, 现在是:{eps}"
        self.lr = lr
        self.params = list(params)
        self.beta = beta
        self.eps = eps
        self.last_v = {}

    def step(self):
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                gt = p.grad
                vt = self.beta * self.last_v.get(p, 0) + (1 - self.beta) * gt**2
                mt = self.lr / (vt.sqrt() + self.eps) * gt
                p -= mt
                self.last_v[p] = vt

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


def test_optim(optimizer_cls):
    w = torch.randn((10, 10), requires_grad=True)

    optimizer = optimizer_cls([w], lr=1)

    for step in range(10):
        loss = ((w - 3) ** 2).mean()  # 假设目标是w = 3
        optimizer.zero_grad()
        loss.backward()  # 求导，并放入每个参数的grad变量中
        optimizer.step()
        print(f"=====[{optimizer_cls}]===== step: {step}, loss: {loss}")


if __name__ == "__main__":
    # 实验下来，在学习率一样的情况下，越先进的优化器越牛
    test_optim(SGD)
    test_optim(Momentum)
    test_optim(RMSprop)
