import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        y_pred = y_pred+epsilon
        n = len(y_pred)
        numerator = np.sum((y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred)))
        loss = -numerator/n
        return np.round(loss,4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        y_pred = y_pred+epsilon
        n = len(y_pred)
        numerator = np.sum((y_true)*np.log(y_pred))
        loss = -numerator/n
        return np.round(loss,4)
