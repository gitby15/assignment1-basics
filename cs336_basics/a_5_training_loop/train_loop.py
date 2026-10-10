from dataclasses import dataclass
import torch
import numpy.typing as npt

from cs336_basics.a_4_train.cross_entropy import cross_entropy
from cs336_basics.a_4_train.grad_clip import grad_clip
from cs336_basics.a_5_training_loop.checkpoint import save_checkpoint
from cs336_basics.a_5_training_loop.get_batch import get_batch


@dataclass
class TrainConfig:
    batch_size: int = 32
    context_length: int = 512
    max_iters: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_interval: int = 100
    checkpoint_interval: int = 100
    checkpoint_path: str = "checkpoints"


def train_loop(model: torch.nn.Module, optimizer: torch.optim.Optimizer, train_dataset: npt.NDArray, config: TrainConfig, device: str):
    model.train()
    model.to(device)

    for i in range(config.max_iters):
        inputs, labels = get_batch(train_dataset, config.batch_size, config.context_length, device)
        logits = model(inputs)
        loss = cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        grad_clip(model.parameters(), config.grad_clip)
        optimizer.step()
        if (i % config.log_interval == 0):
            print(f"step: {i}, loss: {loss.item()}")
        if (i % config.checkpoint_interval == 0):
            save_checkpoint(model, optimizer, i, config.checkpoint_path)




