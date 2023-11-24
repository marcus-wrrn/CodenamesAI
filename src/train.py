from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from processing.processing import Processing
from transformers import DebertaConfig, DebertaModel, AutoTokenizer
from model import SentenceEncoder
import numpy as np



def main():
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Initialize sentence encoders
    pos_encoder = SentenceEncoder(device=device)
    neg_encoder = SentenceEncoder(device=device)


    
    

if __name__ == "__main__":
    main()