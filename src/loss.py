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
        pos_score = sum([util.cos_sim(anchor, enc) for enc in pos_encs]) / len(pos_encs)
        neg_score = sum(util.cos_sim(anchor, enc) for enc in neg_encs) / len(neg_encs)
        loss = pos_score - neg_score + self.margin
        return F.relu(loss)
    
class TriplietCombinedLoss(TripletMeanLoss):
    def __init__(self, margin=1):
        super().__init__(margin)
        self.triplet_loss = nn.TripletMarginLoss(margin=self.margin, p=2)
    
    def forward(self, anchor, pos_encs, neg_encs):
        """TODO: Encorporate triplet margin loss across all values in the """
        ...