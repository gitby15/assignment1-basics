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
            "last_m": {},
            "last_v": {},
        }
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        with torch.no_grad():
            for group in self.param_groups:
                group["current_step"] += 1
                lr = group["lr"]
                beta_1 = group["beta_1"]
                beta_2 = group["beta_2"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                current_step = group["current_step"]
                last_m = group["last_m"]
                last_v = group["last_v"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    gt = p.grad
                    mt = beta_1 * last_m.get(p, 0) + (1 - beta_1) * gt
                    last_m[p] = mt
                    mt = mt / (1 - beta_1 ** current_step)
                    vt = beta_2 * last_v.get(p, 0) + (1 - beta_2) * gt**2
                    last_v[p] = vt
                    vt = vt / (1 - beta_2 ** current_step)
                    at = mt / (vt.sqrt() + eps)
                    p.mul_(1 - lr * weight_decay)
                    p.sub_(lr * at)
        return loss

