import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def sigmoid(self,z:NDArray[np.float64]) -> np.float64:
        return 1/(1+np.exp(-z))
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        forward = np.dot(w,x)+b
        y_hat = self.sigmoid(forward)
        L = np.square(y_hat-y_true)/2
        dl_dw = (y_hat-y_true)*(y_hat)*(1-y_hat)*x
        dl_db = (y_hat-y_true)*(y_hat)*(1-y_hat)
        return (np.round(dl_dw,5),np.round(dl_db,5))
