# 4. Training a Transformer LM

## Problem (cross_entropy):  Implement cross-entropy (1 point)
Deliverable: Write a function to compute the cross-entropy loss, which takes in predicted logits (𝑜𝑖) and targets (𝑥𝑖+1) and computes the cross-entropy ℓ𝑖 = − log softmax(𝑜𝑖)[𝑥𝑖+1]. Your function should handle the following:
• Subtract the largest element for numerical stability.
• Cancel out log and exp whenever possible.
• Handle any additional batch dimensions and return the average across the batch. As with
Section 3.2, we assume batch-like dimensions always come first, before the vocabulary size
dimension.
Implement [adapters.run_cross_entropy] , then run uv run pytest -k test_cross_entropy to test your implementation.

# Problem (learning_rate_tuning):  Tuning the learning rate (1 point)
As we will see, one of the hyperparameters that affects training the most is the learning rate.
Let’s see that in practice in our toy example. Run the SGD example above with three other values
for the learning rate: 1e1, 1e2, and 1e3, for just 10 training iterations. What happens with the loss
for each of these learning rates? Does it decay faster, slower, or does it diverge (i.e., increase over
the course of training)?
Deliverable: A one-to-two sentence response with the behaviors you observed.
[Deliver]:
试了一下1e1，1e2, 1e3，前两个，loss都能下降，1e2更快下降到一个低值，第三个1e3，loss会不断增大，第十次的时候已经变成一个很大很大的数字了

# Problem (adamw):  Implement AdamW (2 points)
Deliverable: Implement the AdamW optimizer as a subclass of torch.optim.Optimizer. Your class should take the learning rate 𝛼 in __init__, as well as the 𝛽, 𝜀 and 𝜆 hyperparameters. To help you keep state, the base Optimizer class gives you a dictionary self.state, which maps nn.Parameter objects to a dictionary that stores any information you need for that parameter (for AdamW, this would be the moment estimates). Implement [adapters.get_adamw_cls] and make sure it passes `uv run pytest -k test_adamw`.
[Deliver]:
手撸了一系列优化器：SGD -> momentum -> RMSProps -> Adam -> AdamW，在optim.py
按照作业也实现了AdamW
两个优化器都通过了测试，不过我没太花时间研究torch.optim.Optimizer 的用法，先学原理，后面再说

# Problem (adamw_accounting):  Resource accounting for training with AdamW (2 points)
Let us compute how much memory and compute running AdamW requires. Assume we are using float32 for every tensor.
(a) How much peak memory does running AdamW require? Decompose your answer based on the memory usage of the parameters, activations, gradients, and optimizer state. Express your answer in terms of the batch_size and the model hyperparameters (vocab_size, context_length, num_layers, d_model, num_heads). Assume 𝑑ff = 8/3 × 𝑑model.
For simplicity, when calculating memory usage of activations, consider only the following components:
- Transformer block
    - RMSNorm(s)
    - Multi-head self-attention sublayer: 𝑄𝐾𝑉 projections, 𝑄𝐾⊤ matrix multiply, softmax,weighted sum of values, output projection.
    - Position-wise feed-forward (SwiGLU): 𝑊1, 𝑊2, SiLU on the gate branch, element-wise product, 𝑊3
    - final RMSNorm
    - output embedding
    - cross-entropy on logits
Deliverable: An algebraic expression for each of parameters, activations gradients, and optimizer state, as well as the total.

(b) Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends on the batch_size. What is the maximum batch size you can use and still fit within 80GB memory?
Deliverable: An expression that looks like 𝑎 ⋅ batch_size + 𝑏 for numerical values 𝑎, 𝑏, and a number representing the maximum batch size.

(c) How many FLOPs does running one step of AdamW take?
Deliverable: An algebraic expression, with a brief justification.

(d) Model FLOPs utilization (MFU) is defined as the ratio of observed throughput (tokens per second) relative to the hardware’s theoretical peak FLOP throughput
[A. Chowdhery et al., 2022]. An NVIDIA H100 GPU has a theoretical peak of 495 teraFLOP/s for “float32” (actually TensorFloat-32, which in reality is “bfloat19”) operations. Assuming you are able to get 50% MFU, how long would it take to train a GPT-2 XL for 400K steps and a batch size of 1024 on a single H100? Following J. Kaplan et al. [25] and J. Hoffmann et al. [26], assume that the backward pass has twice the FLOPs of the forward pass.
Deliverable: The number of hours training would take, with a brief justification


# Problem (learning_rate_schedule):  Implement cosine learning rate schedule with warmup(1 point)
Write a function that takes 𝑡, 𝛼max, 𝛼min, 𝑇𝑤 and 𝑇𝑐, and returns the learning rate 𝛼𝑡 according to
the scheduler defined above. Then implement [adapters.get_lr_cosine_schedule] and make sure it passes uv run pytest -k test_get_lr_cosine_schedule.


# Problem (learning_rate_schedule):  Implement cosine learning rate schedule with warmup (1 point)
Write a function that takes 𝑡, 𝛼max, 𝛼min, 𝑇𝑤 and 𝑇𝑐, and returns the learning rate 𝛼𝑡 according to the scheduler defined above. Then implement [adapters.get_lr_cosine_schedule] and make sure
it passes `uv run pytest -k test_get_lr_cosine_schedule`.
[Deliver]:
其实就是一个基于总步长 + 当前step + 超参数的一个学习率调整的函数
讲义里面没有讲，为什么要这样调整学习率，我觉得应该是大量实践总结下来的经验，这样往往效果更好

# Problem (gradient_clipping):  Implement gradient clipping (1 point)
Write a function that implements gradient clipping. Your function should take a list of parameters
and a maximum ℓ2-norm. It should modify each parameter gradient in place. Use 𝜀 = 10^−6 (thePyTorch default). Then, implement the adapter [adapters.run_gradient_clipping] and make sure
it passes `uv run pytest -k test_gradient_clipping`.


# Problem (training_together):  Put it together (4 points)
Deliverable: Write a script that runs a training loop to train your model on user-provided input.
In particular, we recommend that your training script allow for (at least) the following:
• Ability to configure and control the various model and optimizer hyperparameters.
• Memory-efficient loading of large training and validation datasets with np.memmap.
• Serializing checkpoints to a user-provided path.
• Periodically logging training and validation performance (e.g., to console and/or an external
service like Weights and Biases).