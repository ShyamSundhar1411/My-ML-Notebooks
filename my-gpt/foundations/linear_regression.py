import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        Y_HAT = np.dot(X,weights)
        return np.round(Y_HAT,5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        n = len(ground_truth)
        error = np.sum(np.square(model_prediction-ground_truth))/n
        return np.round(error,5)
