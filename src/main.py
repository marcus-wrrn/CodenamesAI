from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from processing.processing import Processing
from transformers import DebertaConfig, DebertaModel, AutoTokenizer
import numpy as np

class MainModel(nn.Module):
    def __init__(self, model_name="microsoft/deberta-base") -> None:
        super().__init__()
        self.model_name = model_name
        self.model = DebertaModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def tokenize_sentences(self, sentences):
        return self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(self.device)

    def get_tokens(self, text: str):
        tokens = self.tokenizer(text, add_special_tokens=False)
        return tokens.data['input_ids']
    
    @torch.no_grad()
    def tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, add_special_tokens=True, return_tensors='pt')

    def forward(self, text: str):
        inputs = self.tokenize(text)
        logits = self.model(**inputs).last_hidden_state
        # Mean pool output
        pooled = logits.mean(dim=1)
        # Normalize output
        return F.normalize(pooled, p=2, dim=1)

def main():
    encoder = SentenceTransformer('all-mpnet-base-v2')
    proc = Processing(encoder, filepath="/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/data/words.json", download=False)
    
    # Build model
    #config = DebertaConfig()
    model_name = "microsoft/deberta-base"
    model = MainModel(model_name=model_name)

    tokens = model("onomatopeia").detach().numpy()[0]

    magnitude = sum([x**2 for x in tokens])**0.5
    print(f"{magnitude:.5f}")
    #pred = [model.model.config.id2label[t.item()] for t in tokens[0]]
    

if __name__ == "__main__":
    main()