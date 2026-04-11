import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        combined = positive+negative
        vocab = set()
        for sentence in combined:
            for word in sentence.split():
                vocab.add(word)
        tokens = {word:idx+1 for idx,word in enumerate(sorted(list(vocab)))}
        tensors = []
        for sentence in combined:
            encoded = []
            for word in sentence.split():
                encoded.append(tokens[word])
            tensors.append(torch.tensor(encoded))

        tensors = nn.utils.rnn.pad_sequence(tensors,batch_first = True)
        return tensors
