from torch.utils.data import Dataset, DataLoader
import torch
import json
from sentence_transformers import SentenceTransformer

class CodeGiverDataset(Dataset):
    def __init__(self, code_dir: str, game_dir: str):
        super().__init__()
        
        self.code_dir = code_dir
        self.game_dir = game_dir

        with open(self.code_dir, 'r') as fp:
            self.word_data = json.load(fp)
        self.code_dict = self._create_word_dict(self.word_data)
        self.guess_dict = self._create_guess_dict(self.word_data)
        
        with open(self.game_dir, 'r') as fp:
            self.game_data = json.load(fp)
        
        self._process_game_data(self.game_data)        
    
    # Intializers
    def _create_word_dict(self, data: json):
        return { word: embedding for word, embedding in zip(data['codewords'], data['code_embeddings']) }
    
    def _create_guess_dict(self, data: json):
        return { word: embedding for word, embedding in zip(data['guesses'], data['guess_embeddings']) }
    
    def _process_game_data(self, data: json):
        self.positive_sents = data['positive']
        self.negative_sents = data['negative']
        self.neutral_sets = data['neutral']

    # Accessors
    def get_vocab(self, guess_data=True):
        words = []
        embeddings = []
        data = self.guess_dict.items() if guess_data else self.code_dict.items()
        for key, value in data:
            words.append(key)
            embeddings.append(value)
        return words, embeddings
    
    def __len__(self):
        return len(self.positive_sents)
    
    def __getitem__(self, index):
        pos_sent = self.positive_sents[index]
        neg_sent = self.negative_sents[index]

        # Get embeddings
        pos_embeddings = torch.stack([torch.tensor(self.code_dict[word]) for word in pos_sent.split(' ')])
        neg_embeddings = torch.stack([torch.tensor(self.code_dict[word]) for word in neg_sent.split(' ')])

        return pos_sent, neg_sent, pos_embeddings, neg_embeddings
    

class CodeGiverDatasetCombinedSent(CodeGiverDataset):
    def __init__(self, code_dir: str, game_dir: str):
        super().__init__(code_dir, game_dir)

    def __getitem__(self, index):
        pos_sents, neg_sents, pos_embeddings, neg_embeddings = super().__getitem__(index)

        # Combine sentences
        combined_sents = f"{pos_sents}\n{neg_sents}" # Experiment with different seperators (look into <SEP> token, but text should allow for multiple inputs)
        return combined_sents, pos_embeddings, neg_embeddings


def testing_dataloader():
    dataset = CodeGiverDataset(code_dir="../data/words.json", game_dir="../data/three_word_data.json")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = SentenceTransformer('all-mpnet-base-v2')

    for data in dataloader:
        pos_sents, neg_sents, pos_embs, neg_embs = data
        encs = model.encode(pos_sents)
        print(encs.shape)

if __name__ == "__main__":
    testing_dataloader()
    
    
