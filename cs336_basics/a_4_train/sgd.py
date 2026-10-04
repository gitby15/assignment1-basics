from collections.abc import Callable
import torch
import math

# 这个是讲义上的SGD
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"学习率需要大于0, 现在是:{lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get('t', 0)
                grad = p.grad.data
                # p.data -= lr * grad # 不会衰减
                p.data -= lr/math.sqrt(t + 1) * grad # 自动衰减
                

                state['t'] = t + 1
        return loss


def test_example():
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr = 1)
    for t in range(10):
        opt.zero_grad() # 清空梯度
        loss = (weights ** 2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()

if __name__ == "__main__":
    test_example()