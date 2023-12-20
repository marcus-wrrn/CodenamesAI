from sentence_transformers import util
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

def triplet_loss(out_embedding, pos_embeddings, neg_embeddings, margin=0):
    pos_sim = util.cos_sim(out_embedding, pos_embeddings)/len(pos_embeddings)
    neg_sim = util.cos_sim(out_embedding, neg_embeddings)/len(neg_embeddings)
    return max(pos_sim - neg_sim + margin, 0)

class CombinedTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(CombinedTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, pos_encs, neg_encs):
        # Add extra dimension to anchor to align with the pos and neg encodings shape
        anchor_expanded = anchor.unsqueeze(1)   # [batch, emb_size] -> [batch, 1, emb_size]

        pos_score = F.cosine_similarity(anchor_expanded, pos_encs, dim=2)
        pos_score = torch.mean(pos_score, dim=1)

        neg_score = F.cosine_similarity(anchor_expanded, neg_encs, dim=2)
        neg_score = torch.mean(neg_score, dim=1) * 3

        loss = neg_score - pos_score + self.margin
        return F.relu(loss).mean()

class TripletMeanLossL2Distance(CombinedTripletLoss):
    """
    Performs worse than CombinedTripletLoss, potentially due to MPNET being finetuned on cossine similarity not distance

    Interestingly loss seems about the same, even though it tests slightly lower
    More testing is required
    """
    def __init__(self, margin=1.0):
        super(TripletMeanLossL2Distance, self).__init__(margin)
    
    def forward(self, anchor, pos_encs, neg_encs):
        # Add extra dimension to anchor to align with the pos and neg encodings shape
        anchor_expanded = anchor.unsqueeze(1)  # [batch, emb_size] -> [batch, 1, emb_size]

        # Calculate L2 distance for positive and negative pairs
        pos_distance = torch.norm(anchor_expanded - pos_encs, p=2, dim=2)
        neg_distance = torch.norm(anchor_expanded - neg_encs, p=2, dim=2)

        # Calculate mean of distances
        avg_pos_distance = torch.mean(pos_distance, dim=1)
        avg_neg_distance = torch.mean(neg_distance, dim=1) # Attempting to weight the negative more highly,

        # Calculate triplet loss
        loss = F.relu(avg_pos_distance - avg_neg_distance + self.margin)

        return loss.mean()

class CATLoss(CombinedTripletLoss):
    """Combined Asymmetric Triplet Loss"""
    def __init__(self, device, margin=1, weighting=1, neg_weighting=-2):
        super().__init__(margin)
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)
        self.device = device
        self.weighting = weighting
        self.neg_weighting = neg_weighting

    def forward(self, anchor: torch.Tensor, pos_encs: torch.Tensor, neg_encs: torch.Tensor):
        anchor_expanded = anchor.unsqueeze(1)

        pos_score = F.cosine_similarity(anchor_expanded, pos_encs, dim=2)
        neg_score = F.cosine_similarity(anchor_expanded, neg_encs, dim=2)
        scores = torch.cat((pos_score, neg_score), dim=1)

        scores, indices = scores.sort(dim=1, descending=True)
        # Set all positive values to -1 and negative values to 1
        # use a larger negative 
        modified_indices = torch.where(indices < 3, self.neg_weighting, self.weighting)
        scores = torch.mul(scores, modified_indices)
    
        weights = 1/torch.arange(1, scores.shape[1] + 1).to(self.device) + 1
        scores = torch.mul(scores, weights)
        scores = scores.mean(dim=1)
        loss = F.relu(scores + self.margin)
        return loss.mean()
    
class CATLossNormalDistribution(CATLoss):
    """Uses a normal distribution for the weighting function"""
    def __init__(self, stddev: float, mean=0.0, device="cpu", margin=1, weighting=1, neg_weighting=-1, constant=7, list_size=6):
        super().__init__(device, margin, weighting, neg_weighting)
        self.mean = mean
        self.stddev = stddev
        self.constant = constant
        
        self.negative_weighting = neg_weighting
        self._norm_distribution = Normal(self.mean, self.stddev)
        

    def norm_sample(self, num_elements):
        inp_tensor = torch.arange(0, num_elements).to(self.device)
        return torch.exp(self._norm_distribution.log_prob(inp_tensor))

    def forward(self, anchor: torch.Tensor, pos_encs: torch.Tensor, neg_encs: torch.Tensor):
        anchor_expanded = anchor.unsqueeze(1)

        pos_score = F.cosine_similarity(anchor_expanded, pos_encs, dim=2)
        neg_score = F.cosine_similarity(anchor_expanded, neg_encs, dim=2)
        scores = torch.cat((pos_score, neg_score), dim=1)

        scores, indices = scores.sort(dim=1, descending=True)
        modified_indices = torch.where(indices < 3, self.neg_weighting, self.weighting)

        scores = torch.mul(scores, modified_indices)

        # Apply normal distribution
        norm_distribution = self.norm_sample(scores.shape[1]) 
        scores = torch.mul(scores, norm_distribution).sum(1)
        loss = F.relu(scores + self.margin)
        return loss.mean()