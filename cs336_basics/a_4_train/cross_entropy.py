import torch

# 这个是按照公式去实现的交叉熵素损失，但是会存在下溢出的问题
def cross_entropy_old(
    inputs: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    probs = torch.softmax(inputs, dim=-1)
    probs = probs[torch.arange(probs.size(0), device=inputs.device), targets]
    result = -torch.log(probs).mean()
    return result

# 利用了log(a/b) = log(a) - log(b)的特性
def cross_entropy(
    inputs: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    # inputs: [batch_size, vocab_size]
    # targets: [batch_size]

    max_logits = inputs.max(dim=-1, keepdim=True).values
    shifted_logits = inputs - max_logits
    log_sum_exp = torch.log(
        torch.exp(shifted_logits).sum(dim=-1)
    )

    target_logits = inputs[
        torch.arange(inputs.size(0), device=inputs.device),
        targets
    ]

    # Cross Entropy
    # -x + max(x) + log(sum(exp(x)))
    # 这样可以避免数值下溢出和上溢出的问题
    loss = (
        -target_logits
        + max_logits.squeeze(-1)
        + log_sum_exp
    )

    return loss.mean()

# 困惑度 perplexity = e^(loss.mean())
