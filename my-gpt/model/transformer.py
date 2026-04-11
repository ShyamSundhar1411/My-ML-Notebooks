import torch
import torch.nn as nn
from torchtyping import TensorType

# Uses Pre-LN architecture: LayerNorm is applied BEFORE each sub-layer, not after.
# This differs from the original "Attention is All You Need" diagram but trains better.
class TransformerBlock(nn.Module):

    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        # Instantiate in this order:
        # 1. self.MultiHeadedSelfAttention(model_dim, num_heads)
        # 2. self.VanillaNeuralNetwork(model_dim)
        # 3. Two nn.LayerNorm(model_dim) instances
        self.multihead = self.MultiHeadedSelfAttention(model_dim,num_heads)
        self.nn = self.VanillaNeuralNetwork(model_dim)
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        x = embedded+self.multihead(self.ln1(embedded))
        x = x+self.nn(self.ln2(x))
        return torch.round(x,decimals=4)


    class MultiHeadedSelfAttention(nn.Module):

        class SingleHeadAttention(nn.Module):
            def __init__(self, model_dim: int, head_size: int):
                super().__init__()
                torch.manual_seed(0)
                self.key_gen = nn.Linear(model_dim, head_size, bias=False)
                self.query_gen = nn.Linear(model_dim, head_size, bias=False)
                self.value_gen = nn.Linear(model_dim, head_size, bias=False)

            def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                k = self.key_gen(embedded)
                q = self.query_gen(embedded)
                v = self.value_gen(embedded)

                scores = q @ torch.transpose(k, 1, 2) # @ is the same as torch.matmul()
                context_length, attention_dim = k.shape[1], k.shape[2]
                scores = scores / (attention_dim ** 0.5)

                lower_triangular = torch.tril(torch.ones(context_length, context_length))
                mask = lower_triangular == 0
                scores = scores.masked_fill(mask, float('-inf'))
                scores = nn.functional.softmax(scores, dim = 2)

                return scores @ v

        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            torch.manual_seed(0)
            self.att_heads = nn.ModuleList()
            for i in range(num_heads):
                self.att_heads.append(self.SingleHeadAttention(model_dim, model_dim // num_heads))

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            head_outputs = []
            for head in self.att_heads:
                head_outputs.append(head(embedded))
            concatenated = torch.cat(head_outputs, dim = 2)
            return concatenated

    class VanillaNeuralNetwork(nn.Module):

        def __init__(self, model_dim: int):
            super().__init__()
            torch.manual_seed(0)
            self.up_projection = nn.Linear(model_dim, model_dim * 4)
            self.relu = nn.ReLU()
            self.down_projection = nn.Linear(model_dim * 4, model_dim)
            self.dropout = nn.Dropout(0.2) # using p = 0.2

        def forward(self, x: TensorType[float]) -> TensorType[float]:
            torch.manual_seed(0)
            return self.dropout(self.down_projection(self.relu(self.up_projection(x))))
