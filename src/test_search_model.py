from model import  CodeSearchDualNet
from datasets.dataset import CodeDatasetDualModel
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import numpy as np
from utils.vector_search import VectorSearch
import utils.utilities as utils

def calc_cos_score(anchor, pos_encs, neg_encs):
    anchor = anchor.unsqueeze(1)
    pos_score = F.cosine_similarity(anchor, pos_encs, dim=2)
    neg_score = F.cosine_similarity(anchor, neg_encs, dim=2)
    
    return pos_score, neg_score

def process_shape( pos_tensor: torch.Tensor, neg_tensor: torch.Tensor):
        pos_dim = pos_tensor.shape[1]
        neg_dim = neg_tensor.shape[1]
        dif = 0
        if pos_dim > neg_dim:
            dif = pos_dim - neg_dim
            pos_tensor = pos_tensor[:, dif:]
        elif neg_dim > pos_dim:
            dif = neg_dim - pos_dim
            neg_tensor = neg_tensor[:, dif:]
        
        return pos_tensor, neg_tensor, dif

@torch.no_grad()
def test_loop(model: CodeSearchDualNet, dataloader, dataset: CodeDatasetDualModel, device: torch.device, verbose=False):

    total_score = 0

    for i, data in enumerate(dataloader):
        pos_sents, neg_sents, game_state, pos_embs, neg_embs = data
        pos_embs, neg_embs = pos_embs.to(device), neg_embs.to(device)
        words, out, dist = model.infer(pos_sents, neg_sents)
        pos_score, neg_score = calc_cos_score(out, pos_embs, neg_embs)

        pos_score, _ = pos_score.sort(descending=True)
        neg_score, _ = neg_score.sort(descending=False)
        pos_score, neg_score, dif = process_shape(pos_score, neg_score)

        comparison = torch.where(pos_score > neg_score, 1., 0.).to(device)
        score = comparison.sum(dim=1).mean().item() + dif
        if verbose: 
            #print(f"Words: {words}")
            #print(f"Pos Scores Sorted: {pos_score}\nNeg Scores Sorted: {neg_score}")
            print(f"Score: {score}")
        total_score += score
        i += 1
    print(f"Average Score: {total_score/len(dataloader)}")
        
def main(args):
    device = utils.get_device(args.cuda)
    verbose = True if args.v.lower() == 'y' else False
    
    # Initialize data
    test_dataset = CodeDatasetDualModel(code_dir=args.code_dir, game_dir=args.geuss_dir)
    dataloader = DataLoader(test_dataset, batch_size=200)
    vector_db = VectorSearch(test_dataset, prune=True)

    # Initialize model
    model = CodeSearchDualNet(vector_db, device)
    model.load_state_dict(torch.load(args.m))
    model.to(device)
    model.eval()

    test_loop(model, dataloader, test_dataset, device, verbose)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-code_dir', type=str, help='Dataset Path', default="/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/data/words.json")
    parser.add_argument('-geuss_dir', type=str, help="", default="/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/data/codewords_word_data_mini.json")
    parser.add_argument('-m', type=str, help='Model Path', default="/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/test_multi_new_mask.pth")
    parser.add_argument('-raw', type=str, help="Use the Raw Sentence Encoder, [y/N]", default='N')
    parser.add_argument('-cuda', type=str, help="Whether to use CPU or Cuda, use Y or N", default='Y')
    parser.add_argument('-v', type=str, help="Verbose [y/N]", default='Y')
    args = parser.parse_args()
    main(args)