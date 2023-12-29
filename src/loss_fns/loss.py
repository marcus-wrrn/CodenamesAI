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

    It could also be due to the fact that the outputs are normalized and thus all have the same magnitude (exist on hypersphere) which is accounted for by just looking at the angle but leads to 
    information loss when only comparing distance
    """
    def __init__(self, margin=1.0):
        super(TripletMeanLossL2Distance, self).__init__(margin)
    
    def forward(self, anchor: torch.Tensor, pos_encs: torch.Tensor, neg_encs: torch.Tensor):
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

class ScoringLoss(CombinedTripletLoss):
    def __init__(self, margin=1, device='cpu', normalize=True):
        super().__init__(margin)
        self.normalize = normalize
        self.device = device

    def _process_shape(self, pos_tensor: torch.Tensor, neg_tensor: torch.Tensor):
        pos_dim = pos_tensor.shape[1]
        neg_dim = neg_tensor.shape[1]

        if pos_dim > neg_dim:
            dif = pos_dim - neg_dim
            pos_tensor = pos_tensor[:, dif:]
        elif neg_dim > pos_dim:
            dif = neg_dim - pos_dim
            neg_tensor = neg_tensor[:, dif:]
        
        return pos_tensor, neg_tensor
    
    def _calc_score(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor):
        """
        Compares the values of negative triplet scores to positive triplet scores.
        Returns the total number of negative scores greater than positivefor each batch
        """
        pos_sorted, _ = pos_scores.sort(descending=True, dim=1)
        neg_sorted, _ = neg_scores.sort(descending=False, dim=1)
        
        pos_sorted, neg_sorted = self._process_shape(pos_sorted, neg_sorted)
        
        comparison = torch.where(neg_sorted > pos_sorted, 1.0, 0.0).to(self.device)
        
        final_score = comparison.sum(1, keepdim=True)

        if self.normalize:
            final_score = final_score * 1/comparison.shape[1]

        return final_score

    def forward(self, anchor: torch.Tensor, pos_encs: torch.Tensor, neg_encs: torch.Tensor):
        anchor_expanded = anchor.unsqueeze(1)
        pos_scores, neg_scores = self._calc_cos_scores(anchor_expanded, pos_encs, neg_encs)

        total_score = self._calc_score(pos_scores, neg_scores)
        loss = F.relu(neg_scores - pos_scores + total_score + self.margin)
        return loss.mean(), total_score.mean(dim=0)
        

class ScoringLossWithModelSearch(ScoringLoss):
    def __init__(self, margin=1, device='cpu', normalize=True):
        super().__init__(margin, device, normalize)
    
    def forward(self, model_out: torch.Tensor, search_out: torch.Tensor, pos_encs: torch.Tensor, neg_encs: torch.Tensor):
        model_expanded = model_out.unsqueeze(1)
        search_expanded = search_out.unsqueeze(1)
        # calculate scores compared to the search output, not the model output
        s_pos_scores, s_neg_scores = self._calc_cos_scores(search_expanded, pos_encs, neg_encs)
        m_pos_scores, m_neg_scores = self._calc_cos_scores(model_expanded, pos_encs, neg_encs)

        total_score = self._calc_score(s_pos_scores, s_neg_scores)

        loss = F.relu(m_neg_scores.mean(dim=1) - m_pos_scores.mean(dim=1) + total_score + self.margin)
        # loss_select = F.relu(s_neg_scores - s_pos_scores + total_score + self.margin)
        # loss = loss + loss_select
        return loss.mean(), total_score.mean(dim=0)

class MultiObjectiveScoringLoss(ScoringLoss):
    def __init__(self, margin=1, device='cpu', normalize=True):
        super().__init__(margin, device, normalize)
    
    def forward(self, anchor: torch.Tensor, pos_encs: torch.Tensor, neg_encs: torch.Tensor, neutral_encs: torch.Tensor):
        ...