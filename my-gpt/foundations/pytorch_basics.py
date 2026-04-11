import torch
import torch.nn
from torchtyping import TensorType

# Round all answers to 4 decimal places: torch.round(tensor, decimals=4)
class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        # Reshape (M, N) tensor to (M*N/2, 2)
        # Use torch.reshape(tensor, new_shape)
        shape = to_reshape.shape
        new_shape = ((shape[0]*shape[1])//2,2)
        return torch.reshape(to_reshape,new_shape)

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        # Compute column-wise mean (average across rows)
        # Use torch.mean(tensor, dim=0)
        return torch.mean(to_avg,dim=0)

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        # Join two tensors side-by-side along dim=1
        # Use torch.cat((a, b), dim=1)
        return torch.cat((cat_one,cat_two),dim = 1)

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        # Compute Mean Squared Error between prediction and target
        # Use torch.nn.functional.mse_loss(prediction, target)
        return torch.nn.functional.mse_loss(prediction,target)
