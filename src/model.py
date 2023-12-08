from sentence_transformers import SentenceTransformer, InputExample
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from processing.processing import Processing
from transformers import DebertaConfig, DebertaModel, DebertaTokenizer
import numpy as np
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Take attention mask into account for correct averaging
def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def mean_pooling_ein(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    mask = attention_mask.unsqueeze(-1)  # Add an extra dimension for broadcasting
    sum_embeddings = torch.einsum('ijk,ij->ik', token_embeddings, mask.float())
    sum_mask = mask.sum(1)
    return sum_embeddings / torch.clamp(sum_mask, min=1e-9)

class SentenceEncoderRaw():
    """
    Sentence Transformer used for encoding input sentences. Does not use sentence_transformers library

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
        sentence_embeddings = mean_pooling_ein(token_embeddings, tokenized_sents['attention_mask'])
        return F.normalize(sentence_embeddings, p=2, dim=1) if normalize else sentence_embeddings
    




class SentenceEncoderLib(nn.Module):
    def __init__(self, model_name='all-mpnet-base-v2'):
        super(SentenceEncoderLib, self).__init__()

        self.name = model_name
        self.encoder = SentenceTransformer(self.name)
        
    def forward(self, text: torch.Tensor):
        encodings = self.encoder.encode(text, convert_to_tensor=True)
        if encodings.ndim == 1:
            encodings = encodings.unsqueeze(0)
        # out = self.fc(encodings)
        return encodings


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
            'pos': SentenceEncoderLib(model_name) if has_pos else None,
            'neg': SentenceEncoderLib(model_name) if has_neg else None,
            'neutral': SentenceEncoderLib(model_name) if has_neutral else None,
            'assassin': SentenceEncoderLib(model_name) if has_assassin else None
        }

    def forward(self, pos_sents='', neg_sents='', neutral_sents='', assassin_word=''):
        encoded = {
            'pos': self.encoders['pos'](pos_sents) if self.encoders['pos'] else None,
            'neg': self.encoders['neg'](neg_sents) if self.encoders['neg'] else None,
            'neutral': self.encoders['neutral'](neutral_sents) if self.encoders['neutral'] else None,
            'assassin': self.encoders['assassin'](assassin_word) if self.encoders['assassin'] else None
        }

        return encoded['pos'], encoded['neg'], encoded['neutral'], encoded['assassin']


class SimpleCodeGiver(nn.Module):
    """Only encodes positive and negative sentences"""
    def __init__(self, model_name='all-mpnet-base-v2'):
        super().__init__()
        self.name = model_name
        
        self.pos_encoder = SentenceEncoderLib(model_name)
        self.neg_encoder = SentenceEncoderLib(model_name)

        self.fc = nn.Sequential(
            nn.Linear(1536, 1250),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1250, 1000),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            #nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(1000, 768)
        )
    
    def forward(self, pos_texts: str, neg_texts: str):
        pos_emb = self.pos_encoder(pos_texts)
        neg_emb = self.neg_encoder(neg_texts)
        
        concatenated = torch.cat((pos_emb, neg_emb), 1)
        out = self.fc(concatenated)
        return F.normalize(out, p=2, dim=1)


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
    # testing
    positive_words = "cat dog house"
    negative_words = "car train sun"
    device = torch.device('cuda')

    encoder = SimpleCodeGiver()
    encoder.to(device)
    vals = encoder(positive_words, negative_words)

    print(vals.shape)
    print("Done")

if __name__ == "__main__":
    main()