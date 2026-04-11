from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        stoi = {}
        itos = {}
        text = sorted(list(text))
        counter = 0
        for i in range(len(text)):
            if text[i] not in stoi:
                stoi[text[i]] = counter
                itos[counter] = text[i]
                counter+=1
        return (stoi,itos)


    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        res = []
        for i in text:
            res.append(stoi[i])
        print(res)
        return res

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        res = []
        for i in ids:
            res.append(itos[i])
        print(res)
        return "".join(res)
