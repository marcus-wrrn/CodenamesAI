from torch.utils.data import Dataset, DataLoader
import torch
import os
import json
from sentence_transformers import SentenceTransformer

class CodeGiverDataset(Dataset):
    def __init__(self, code_dir: str, game_dir: str):
        super().__init__()
        
        self.code_dir = code_dir
        self.game_dir = game_dir

        with open(self.code_dir, 'r') as fp:
            self.code_words = json.load(fp)
        self.code_dict = self._create_word_dict(self.code_words)
        
        with open(self.game_dir, 'r') as fp:
            self.data = json.load(fp)
        
        self._process_game_data(self.data)        


    def _create_word_dict(self, data: json):
        return { word: embedding for word, embedding in zip(data['codewords'], data['code_embeddings']) }
    
    def _process_game_data(self, data: json):
        self.positive_sents = data['positive']
        self.negative_sents = data['negative']
        self.neutral_sets = data['neutral']

    def __len__(self):
        return len(self.positive_sents)
    
    def __getitem__(self, index):
        pos_sent = self.positive_sents[index]
        neg_sent = self.negative_sents[index]

        # Get embeddings
        pos_embeddings = [self.code_dict[word] for word in pos_sent.split(' ')]
        neg_embeddings = [self.code_dict[word] for word in neg_sent.split(' ')]

        return pos_sent, neg_sent, pos_embeddings, neg_embeddings


def test_dataloader():
    dataset = CodeGiverDataset(code_dir="../data/words.json", game_dir="../data/three_word_data.json")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = SentenceTransformer('all-mpnet-base-v2')

    for data in dataloader:
        pos_sents, neg_sents, pos_embs, neg_embs = data
        encs = model.encode(pos_sents)
        print(encs.shape)

if __name__ == "__main__":
    test_dataloader()
    
    
