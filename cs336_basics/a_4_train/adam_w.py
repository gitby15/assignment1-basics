from collections.abc import Callable

import torch


# adamW: θ(t+1) = θ(t) - lr*A(t) - lr*λ*θ(t)
# A(t) = m(t) / (sqrt(v(t)) + ε)
# m(t) = β1*m(t-1) + (1-β1)*g(t)
# v(t) = β2*v(t-1) + (1-β2)*g(t)^2
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas: tuple[float, float]=(0.9, 0.999), eps=1e-8, weight_decay=0.05):
        defaults = {
            "lr": lr,
            "beta_1": betas[0],
            "beta_2": betas[1],
            "eps": eps,
            "weight_decay": weight_decay,
            "current_step": 0,
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                group["current_step"] += 1
                step = group["current_step"]

                lr = group["lr"]
                beta_1 = group["beta_1"]
                beta_2 = group["beta_2"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    grad = p.grad
                    state = self.state[p]

                    # 初始化参数的优化器状态
                    if len(state) == 0:
                        state["m"] = torch.zeros_like(p)
                        state["v"] = torch.zeros_like(p)

                    m = state["m"]
                    v = state["v"]

                    # 更新一阶、二阶动量
                    m.mul_(beta_1).add_(grad, alpha=1 - beta_1)
                    v.mul_(beta_2).addcmul_(
                        grad, grad, value=1 - beta_2
                    )

                    # 偏差修正
                    m_hat = m / (1 - beta_1 ** step)
                    v_hat = v / (1 - beta_2 ** step)

                    # AdamW 更新
                    p.mul_(1 - lr * weight_decay)
                    p.addcdiv_(
                        m_hat,
                        v_hat.sqrt().add_(eps),
                        value=-lr,
                    )

        return loss

