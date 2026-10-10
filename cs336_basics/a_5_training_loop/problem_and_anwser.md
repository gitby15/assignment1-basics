# Problem (data_loading):  Implement data loading (2 points)
Deliverable: Write a function that takes a numpy array 𝑥 (integer array with token IDs), a batch_size, a context_length and a PyTorch device string (e.g., 'cpu' or 'cuda:0'), and returns a pair of tensors: the sampled input sequences and the corresponding next-token targets. Both tensors should have shape (batch_size, context_length) containing token IDs, and both should be placed on the requested device. To test your implementation against our provided tests, you will first need to implement the test adapter at [adapters.run_get_batch] . Then, run `uv run pytest -k test_get_batch` to test your implementation.


# Problem (checkpointing):  Implement model checkpointing (1 point)
Implement the following two functions to load and save checkpoints:
```python
def save_checkpoint(model, optimizer, iteration, out)
# should dump all the state from the model, optimizer and iteration into the file-like object out. You can use the state_dict method of both the model and the optimizer to get their relevant states and use torch.save(obj, out) to dump obj into out (PyTorch supports either a path or a file-like object here). A typical choice is to have obj be a dictionary, but you can use whatever format you want as long as you can load your checkpoint later.
# This function expects the following parameters:
# - model: torch.nn.Module  
# - optimizer: torch.optim.Optimizer  
# - iteration: int  
# - out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]  
def load_checkpoint(src, model, optimizer) 
# should load a checkpoint from src (path or file-like object), and then recover the model and optimizer states from that checkpoint. Your function should return the iteration number that was saved to the checkpoint. You can use torch.load(src) to recover what you saved in your save_checkpoint implementation, and the load_state_dict method in both the model and optimizer to return them to their previous states
# This function expects the following parameters:
# - src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]  
# - model: torch.nn.Module  
# - optimizer: torch.optim.Optimizer  
```
Implement the [adapters.run_save_checkpoint] and [adapters.run_load_checkpoint] adapters, and make sure they pass `uv run pytest -k test_checkpointing`.