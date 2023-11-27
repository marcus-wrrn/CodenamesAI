from torch.utils.data import Dataset
import torch
import os
import json

class CodeGiverDataset(Dataset):
    def __init__(self, dir: str):
        super().__init__()
        self.dir = dir
        self.data = json.load(self.dir)

    def __len__(self):
        return len(self.data[''])