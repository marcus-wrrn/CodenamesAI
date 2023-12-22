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

    def _calc_cos_scores(self, anchor, pos_encs, neg_encs):
        """Finds cosine similarity between all word embeddings and the model output"""
        pos_score = F.cosine_similarity(anchor, pos_encs, dim=2)
        neg_score = F.cosine_similarity(anchor, neg_encs, dim=2)
        return pos_score, neg_score

    def forward(self, anchor, pos_encs, neg_encs):
        # Add extra dimension to anchor to align with the pos and neg encodings shape
        anchor_expanded = anchor.unsqueeze(1)   # [batch, emb_size] -> [batch, 1, emb_size]

        pos_score, neg_score = self._calc_cos_scores(anchor_expanded, pos_encs, neg_encs)
        # Combine scores
        pos_score = torch.mean(pos_score, dim=1)
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
        self.std = stddev
        self.constant = constant
        
        self.negative_weighting = neg_weighting
        self._norm_distribution = Normal(self.mean, self.std)
        

    def norm_sample(self, indicies):
        norm_dist = torch.exp(self._norm_distribution.log_prob(indicies))
        mean = norm_dist.mean()
        return (norm_dist - mean) / self.std

    def forward(self, anchor: torch.Tensor, pos_encs: torch.Tensor, neg_encs: torch.Tensor):
        anchor_expanded = anchor.unsqueeze(1)

        pos_score = F.cosine_similarity(anchor_expanded, pos_encs, dim=2)
        neg_score = F.cosine_similarity(anchor_expanded, neg_encs, dim=2)
        scores = torch.cat((pos_score, neg_score), dim=1)

        sorted_scores, indices = scores.sort(dim=1, descending=True)
        
        # Create loss mask
        size = scores.shape[1]
        mask = torch.ones(size).to(self.device)
        mask[size//2:] = -1
        # mask = mask.unsqueeze(1)
        # Apply mask
        scores = torch.mul(scores, mask)
        # Find normal distribution 
        norm_distribution = self.norm_sample(indices)
        # Apply weights 
        scores = torch.mul(scores, norm_distribution).sum(1)
        loss = F.relu(scores + self.margin)
        return loss.mean()

class ScoringLoss(CombinedTripletLoss):
    def __init__(self, margin=1, device='cpu'):
        super().__init__(margin)
        self.device = device

    def forward(self, anchor: torch.Tensor, pos_encs: torch.Tensor, neg_encs: torch.Tensor):
        anchor_expanded = anchor.unsqueeze(1)
        pos_scores, neg_scores = self._calc_cos_scores(anchor_expanded, pos_encs, neg_encs)

        pos_sorted, _ = pos_scores.sort(descending=True, dim=1)
        neg_sorted, _ = neg_scores.sort(descending=False, dim=1)
        comparison = torch.where(neg_sorted > pos_sorted, 1.0, 0.0).to(self.device)
        
        margin = comparison.sum(1, keepdim=True)
        loss = F.relu(neg_scores - pos_scores + margin + self.margin)
        return loss.mean()
        