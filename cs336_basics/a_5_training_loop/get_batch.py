import torch
import numpy.typing as npt
import numpy as np

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:

    max_start_idx = len(dataset) - context_length # 确保不会超界

    start_idx = np.random.randint(0, max_start_idx, size=batch_size)
    x = np.stack([
        dataset[s : s + context_length]
        for s in start_idx
    ])

    y = np.stack([
        dataset[s + 1 : s + context_length + 1]
        for s in start_idx
    ])

    # 转换为 Tensor 并放到指定设备
    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)

    return x, y
