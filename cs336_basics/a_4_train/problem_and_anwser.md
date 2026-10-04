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