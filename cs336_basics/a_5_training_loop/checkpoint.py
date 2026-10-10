import torch
import os
import typing

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, outout: str | os.PathLike | typing.BinaryIO | typing.IO[bytes] ):
    optimizer_state = optimizer.state_dict()
    model_weight = model.state_dict()
    torch.save({
        "optimizer_state": optimizer_state,
        "model_weight": model_weight,
        "iteration": iteration
    }, outout)



def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    checkpoint = torch.load(src)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    model.load_state_dict(checkpoint["model_weight"])
    return checkpoint["iteration"]
