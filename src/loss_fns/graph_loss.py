from vector_search import VectorSearch
from dataset import CodeGiverDataset
from sentence_transformers import util
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from loss_fns.loss import CATLoss
import numpy as np



class CATCluster(CATLoss):
    def __init__(self, dataset: CodeGiverDataset, device: torch.device, margin=1, weighting=1, neg_weighting=-2):
        super().__init__(device, margin, weighting, neg_weighting)
        # Set the vector search object to use the codename vocab instead of the guess word vocab
        self.db = VectorSearch(dataset, useGuessData=False)
        
