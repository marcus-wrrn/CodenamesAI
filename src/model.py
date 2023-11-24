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


class CodeGiver(nn.Module):
    def __init__(self, model_name="microsoft/deberta-base", device=torch.device('cpu')) -> None:
        super().__init__()
        self.model_name = model_name
        self.deberta = DebertaModel.from_pretrained(self.model_name)
        self.tokenizer = DebertaTokenizer.from_pretrained(self.model_name)
        self.device = device

    def tokenize_sentences(self, sentences):
        return self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)

    
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