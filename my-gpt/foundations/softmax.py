import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        max_value = np.max(z)
        denom = np.sum(np.exp(z-max_value))
        return np.round(np.exp(z-max_value)/denom,4)
