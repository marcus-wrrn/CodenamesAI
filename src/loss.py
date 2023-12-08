from sentence_transformers import util
import torch
import torch.nn as nn
import torch.nn.functional as F

def triplet_loss(out_embedding, pos_embeddings, neg_embeddings, margin=0):
    pos_sim = util.cos_sim(out_embedding, pos_embeddings)/len(pos_embeddings)
    neg_sim = util.cos_sim(out_embedding, neg_embeddings)/len(neg_embeddings)
    return max(pos_sim - neg_sim + margin, 0)

class TripletMeanLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletMeanLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, pos_encs, neg_encs):
        # Add extra dimension to anchor to align with the pos and neg encodings shape
        anchor_expanded = anchor.unsqueeze(1)   # [batch, emb_size] -> [batch, 1, emb_size]

        pos_score = F.cosine_similarity(anchor_expanded, pos_encs, dim=2)
        pos_score = torch.mean(pos_score, dim=1)

        neg_score = F.cosine_similarity(anchor_expanded, neg_encs, dim=2)
        neg_score = torch.mean(neg_score, dim=1)

        loss = neg_score - pos_score + self.margin
        return F.relu(loss).mean()

class TripletMeanLossL2Distance(TripletMeanLoss):
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
        avg_neg_distance = torch.mean(neg_distance, dim=1)

        # Calculate triplet loss
        loss = F.relu(avg_pos_distance - avg_neg_distance + self.margin)

        return loss.mean()

class CombinedAsymmetricTripletLoss(TripletMeanLoss):
    def __init__(self, margin=1):
        super().__init__(margin)
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)
    
    def forward(self, anchor, pos_encs, neg_encs):
        """TODO: Encorporate triplet margin loss across all values"""
        ...