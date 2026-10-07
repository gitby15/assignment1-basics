import math

def cosine_lr_schedule(t, lr_max, lr_min, tw, tc) -> float:
    lr = lr_min
    # 预热阶段
    if t < tw:
        lr = t/tw * lr_max
    # 退火阶段
    elif tw <= t and t <= tc:
        gap = lr_max - lr_min
        temp = (t - tw)/(tc - tw) * math.pi
        temp = math.cos(temp)
        temp += 1
        temp *= 0.5
        lr = lr_min + gap * temp
    # 收汁阶段
    elif t > tc:
        lr = lr_min
    return lr