from sentence_transformers import SentenceTransformer, InputExample
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from processing.processing import Processing
from transformers import DebertaConfig, DebertaModel, DebertaTokenizer
import numpy as np

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class SentenceEncoder():
    """
    Sentence Transformer used for encoding input sentences

    TODO: Add max_token length
    """
    def __init__(self, 
                 device="cpu",
                 tokenizer_path="sentence-transformers/all-mpnet-base-v2", 
                 model_path="sentence-transformers/all-mpnet-base-v2" 
                ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.device = device

    def tokenize_sentences(self, sentences):
        return self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)
    
    def get_token_embeddings(self, tokenized_sentences):
        return self.model(**tokenized_sentences)

    def encode(self, sentences, normalize=True) -> torch.Tensor:
        tokenized_sents = self.tokenize_sentences(sentences)
        token_embeddings = self.get_token_embeddings(tokenized_sents)
        sentence_embeddings = mean_pooling(token_embeddings, tokenized_sents['attention_mask'])
        # Consider not adding normalization (does that improve performance?)
        return F.normalize(sentence_embeddings, p=2, dim=1) if normalize else sentence_embeddings


class SentenceEncoder(nn.Module):
    def __init__(self, model_name='all-mpnet-base-v2'):
        super(SentenceEncoder, self).__init__()

        self.name = model_name
        self.encoder = SentenceTransformer(self.name)
        self.fc = nn.Sequential(
            nn.Linear(768, 500),
            nn.ReLU(),
            nn.Linear(500, 256),
        )
    
    def forward(self, text: torch.Tensor):
        encodings = self.encoder.encode(text, convert_to_tensor=True)
        out = self.fc(encodings)
        return out


class EncoderLayer(nn.Module):
    """
    A neural network module for encoding sentences using the SentenceTransformer model.

    This class supports encoding sentences with different labels (e.g., positive, negative, neutral, assassin).
    Each label type can be optionally included in the encoding process.

    Attributes:
        pos_encoder (SentenceTransformer): Encoder for positive sentences.
        neg_encoder (SentenceTransformer): Encoder for negative sentences.
        neutral_encoder (SentenceTransformer): Encoder for neutral sentences.
        assassin_encoder (SentenceTransformer): Encoder for assassin words.
    """

    def __init__(self, has_pos=True, has_neg=True, has_neutral=True, has_assassin=False) -> None:
        super().__init__()

        model_name = 'all-mpnet-base-v2'
        self.encoders = {
            'pos': SentenceEncoder(model_name) if has_pos else None,
            'neg': SentenceEncoder(model_name) if has_neg else None,
            'neutral': SentenceEncoder(model_name) if has_neutral else None,
            'assassin': SentenceEncoder(model_name) if has_assassin else None
        }

    def forward(self, pos_sents='', neg_sents='', neutral_sents='', assassin_word=''):
        encoded = {
            'pos': self.encoders['pos'](pos_sents) if self.encoders['pos'] else None,
            'neg': self.encoders['neg'](neg_sents) if self.encoders['neg'] else None,
            'neutral': self.encoders['neutral'](neutral_sents) if self.encoders['neutral'] else None,
            'assassin': self.encoders['assassin'](assassin_word) if self.encoders['assassin'] else None
        }

        return encoded['pos'], encoded['neg'], encoded['neutral'], encoded['assassin']

class SimpleEncoderLayer(nn.Module):
    def __init__(self, model_name='all-mpnet-base-v2'):
        super().__init__()
        self.name = model_name
        
        self.pos_encoder = SentenceEncoder(model_name)
        self.neg_encoder = SentenceEncoder(model_name)

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 758)
        )
    
    def forward(self, pos_texts, neg_texts):
        pos_emb = self.pos_encoder(pos_texts)
        neg_emb = self.neg_encoder(neg_texts)

        concatenated = torch.cat((pos_emb, neg_emb), 0)
        return self.fc(concatenated)



class CodeGiver(nn.Module):
    def __init__(self, model_name="microsoft/deberta-base", device=torch.device('cpu')) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.deberta = DebertaModel.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = DebertaTokenizer.from_pretrained(self.model_name)
        
    def tokenize(self, positive_text: str, negative_text: str) -> torch.Tensor:
        return self.tokenizer(positive_text, negative_text, add_special_tokens=True, return_tensors='pt').to(self.device)

    def forward(self, positive_text: str, negative_text: str):
        inputs = self.tokenize(positive_text, negative_text)
        logits = self.deberta(**inputs).last_hidden_state
        # Mean pool output
        pooled = logits.mean(dim=1)
        # Normalize output
        return F.normalize(pooled, p=2, dim=1)


def main():
    positive_words = "cat dog house"
    negative_words = "car train sun"

    model = CodeGiver()

    out = model(positive_words, negative_words)

    print(out.shape)

if __name__ == "__main__":
    main()