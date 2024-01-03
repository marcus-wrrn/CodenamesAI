from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
#from transformers import DebertaConfig, DebertaModel, DebertaTokenizer
from utils.vector_search import VectorSearch
import numpy as np
import os
from models.model import SentenceEncoder

class MORSpyMaster(nn.Module):
    """Multi-Objective Retrieval model for codenames"""
    def __init__(self, vocab: VectorSearch, device: torch.device, backbone='all-mpnet-base-v2', vocab_size=50):
        super().__init__()
        self.encoder = SentenceEncoder(backbone)
        self.vocab_size = vocab_size
        
        self.fc = nn.Sequential(
            nn.Linear(2304, 1800),
            nn.ReLU(),
            nn.Linear(1800, 1250),
            nn.ReLU(),
            nn.Linear(1250, 900),
            nn.ReLU(),
            nn.Linear(900, 768),
        )
        self.vocab = vocab
        self.device = device

    def _process_embeddings(self, embs: Tensor):
        out = embs.mean(dim=1)
        out = F.normalize(out, p=2, dim=1)
        return out
    
    def forward(self, pos_embs: Tensor, neg_embs: Tensor, neut_embs: Tensor):
        neg_emb = self._process_embeddings(neg_embs)
        neut_emb = self._process_embeddings(neut_embs)
        pos_emb = self._process_embeddings(pos_embs)

        concatenated = torch.cat((neg_emb, neut_emb, pos_emb), dim=1)
        model_out = self.fc(concatenated)
        model_out = F.normalize(model_out, p=2, dim=1)

        words, word_embeddings, dist = self.vocab.search(model_out, num_results=self.vocab_size)
        word_embeddings = torch.tensor(word_embeddings).to(self.device).squeeze(1)
        
        # Find both highest and lowest scoring words
        pos_score = F.cosine_similarity(word_embeddings, pos_emb.unsqueeze(1), dim=2)
        neg_score = F.cosine_similarity(word_embeddings, neg_emb.unsqueeze(1), dim=2)
        neut_score = F.cosine_similarity(word_embeddings, neut_emb.unsqueeze(1), dim=2)
        
        # Calculate loss 
        loss = neg_score/2 + neut_score/2 - pos_score

        index_min = torch.argmin(loss, dim=1)
        index_max = torch.argmax(loss, dim=1)
        search_out_min = word_embeddings[torch.arange(word_embeddings.shape[0]), index_min]
        search_out_max = word_embeddings[torch.arange(word_embeddings.shape[0]), index_max]

        if self.training:
            return model_out, search_out_max, search_out_min
        return words, word_embeddings, dist


