import numpy as np
from numpy.typing import NDArray


class Solution:
    def lookup(self, embeddings: NDArray[np.float64], token_ids: NDArray[np.int64]) -> NDArray[np.float64]:
        # embeddings: (vocab_size, embed_dim) matrix
        # token_ids: 1D array of integer token IDs
        # Return the embedding vectors for the given token IDs
        # return np.round(your_answer, 5)
        embedding_vector = [np.round(embeddings[idx],5) for idx in token_ids]
        return embedding_vector
