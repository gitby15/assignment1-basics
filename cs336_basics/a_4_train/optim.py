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
        # 这是声明接下来的计算不用存入计算图，后面有空了去了解一下pytorch是怎么保存计算图的
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


# θ(t+1) = θ(t) - lr * v(t)
# v(t) = momentum * v(t-1) + (1 - momentum) * gt
# gt = ∇L(θ(t);Bt) = ∂L/∂θ(t)
class Momentum:
    def __init__(self, params: Iterator[torch.Tensor], lr=1e-3, momentum=0.9):
        assert lr > 0, f"学习率需要大于0, 现在是:{lr}"
        assert momentum >= 0 and momentum < 1, f"动量因子需要在0到1之间, 现在是:{momentum}"
        self.lr = lr
        self.momentum = momentum
        self.params = list(params)
        self.last_v = {}
        self._current_step = 0

    def step(self):
        self._current_step += 1
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                # 更新参数，p.grad是参数执行backward的时候自动计算的
                v_t = self.momentum * self.last_v.get(p, 0) + (1 - self.momentum) * p.grad
                self.last_v[p] = v_t
                v_t = v_t / (1 - self.momentum**self._current_step)
                p -= self.lr * v_t

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
        self._current_step = 0

    def step(self):
        self._current_step += 1
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                gt = p.grad
                vt = self.beta * self.last_v.get(p, 0) + (1 - self.beta) * gt**2
                self.last_v[p] = vt
                vt = vt / (1 - self.beta**self._current_step)
                mt = self.lr / (vt.sqrt() + self.eps) * gt
                p -= mt

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# Adam的特性是Momentum + RMSProp + Bias Correction
# 一阶矩：像Momentum一样记录动量：
# m(t) = β1*m(t-1) + (1-β1)*g(t)
# g(t) = ∇L(θ(t);Bt) = ∂L/∂θ(t)
# 二阶矩：像RMSProp一样做自适应学习率：
# v(t) = β2*v(t-1) + (1-β2)*g(t)^2
# Bias Correction: x(t) = x(t) / (1 - β^t), 即M和V都除以(1 - β^t)
# m管方向，v管大小
# 最后，更新公式：
# θ(t+1) = θ(t) - lr*A(t)
# A(t) = m(t) / (sqrt(v(t)) + ε)
class Adam:
    def __init__(
        self,
        params: Iterator[torch.Tensor],
        lr: float = 1e-3,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.last_m = {}
        self.last_v = {}
        self._current_step = 0

    def step(self):
        self._current_step += 1
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                gt = p.grad
                mt = self.beta_1 * self.last_m.get(p, 0) + (1 - self.beta_1) * gt
                self.last_m[p] = mt
                mt = mt / (1 - self.beta_1**self._current_step)

                vt = self.beta_2 * self.last_v.get(p, 0) + (1 - self.beta_2) * gt**2
                self.last_v[p] = vt
                vt = vt / (1 - self.beta_2**self._current_step)
                at = mt / (vt.sqrt() + self.eps)
                p -= self.lr * at

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# AdamW就是在adam的基础上增加了权重衰减
# adam: θ(t+1) = θ(t) - lr*A(t)
# adamW: θ(t+1) = θ(t) - lr*A(t) - lr*λ*θ(t)
# λ是权重衰减因子
class AdamW:
    def __init__(
        self,
        params: Iterator[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.05,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta_1, self.beta_2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.last_m = {}
        self.last_v = {}
        self._current_step = 0

    def step(self):
        self._current_step += 1
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                gt = p.grad
                mt = self.beta_1 * self.last_m.get(p, 0) + (1 - self.beta_1) * gt
                self.last_m[p] = mt
                mt = mt / (1 - self.beta_1**self._current_step)

                vt = self.beta_2 * self.last_v.get(p, 0) + (1 - self.beta_2) * gt**2
                self.last_v[p] = vt
                vt = vt / (1 - self.beta_2**self._current_step)
                at = mt / (vt.sqrt() + self.eps)

                # 这里只在局部修改了p指针指向的位置,并没有修改到权重本身
                # p = (1 - self.lr * self.weight_decay) * p - self.lr * at
                # 跟上面的公式等价，但是修改了p本身
                p.mul_(1 - self.lr * self.weight_decay)
                p.sub_(self.lr * at)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


def test_optim(optimizer_cls):
    torch.manual_seed(42)
    w = torch.randn((10, 10), requires_grad=True)
    print(f"w 初始值抽样：w[1,1]={w[1][1]}, w[5,5]={w[5][5]}")

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
    test_optim(Adam)
    test_optim(AdamW)
