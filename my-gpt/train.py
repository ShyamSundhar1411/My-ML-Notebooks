import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model and return the final loss (rounded to 4 decimals).
        #
        # Steps:
        # 1. Create an AdamW optimizer with the given learning rate
        # 2. For each epoch:
        #    a. Use torch.manual_seed(epoch) for reproducibility
        #    b. Sample random start indices with torch.randint
        #    c. Build X (input) and Y (target) batches, each (batch_size, context_length)
        #       Y is X shifted right by 1
        #    d. Forward pass: logits = model(X), shape (batch_size, context_length, vocab_size)
        #    e. Reshape for cross_entropy: logits to (B*T, C), targets to (B*T)
        #    f. Compute loss = F.cross_entropy(logits_flat, targets_flat)
        #    g. optimizer.zero_grad(), loss.backward(), optimizer.step()
        # 3. Return the final loss value rounded to 4 decimals
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr = lr
        )
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            indices = torch.randint(len(data)-context_length,(batch_size,))
            X = torch.stack([data[i:i+context_length] for i in indices])
            Y = torch.stack([data[i+1:i+1+context_length] for i in indices])
            logits = model(X)
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            Y = Y.view(B*T)
            loss = F.cross_entropy(logits,Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return round(loss.item(), 4)