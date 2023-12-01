from torch.utils.data import Dataset
import torch
import os
import json

class CodeGiverDataset(Dataset):
    def __init__(self, code_dir: str, game_dir: str):
        super().__init__()
        
        self.code_dir = code_dir
        self.game_dir = game_dir

        self.data = json.load(self.code_dir)
        self.code_dict = self._create_word_dict(self.data)

    def _create_word_dict(self, data: json):
        """TODO: Change embeddings name"""
        return { word: embedding for word, embedding in zip(data['codewords'], data['code_embeddings'])}

    def __len__(self):
        return len(self.data[''])
    


    
