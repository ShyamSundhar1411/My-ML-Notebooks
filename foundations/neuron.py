import numpy as np
from numpy.typing import NDArray


class Solution:
    def sigmoid(self,z: NDArray[np.float64]) -> np.float64:
        return 1/(1+np.exp(-z))
    def relu(self,z:NDArray[np.float64]) -> np.float64:
        return np.maximum(0,z)
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        z = np.dot(w,x)+b
        if activation == "sigmoid":
            z = self.sigmoid(z)
        else:
            z = self.relu(z)
        return np.round(z,5)
