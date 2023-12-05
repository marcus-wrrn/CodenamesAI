from model import SimpleCodeGiver
from dataset import CodeGiverDataset
import torch
import torch.nn.functional as F
import argparse
import faiss
import numpy as np

def get_device(is_cuda: str):
    if (is_cuda.lower() == 'y' and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")

def calc_cos_score(anchor, pos_encs, neg_encs):
    pos_score = F.cosine_similarity(anchor, pos_encs, dim=1)
    #pos_score = torch.mean(pos_score, dim=1)

    neg_score = F.cosine_similarity(anchor, neg_encs, dim=1)
    #neg_score = torch.mean(neg_score, dim=1)

    
    print(f"Pos Score: {pos_score}")
    print(f"Neg Score: {neg_score}\n")

@torch.no_grad()
def test_loop(model: SimpleCodeGiver, dataset: CodeGiverDataset, device: torch.device):
    vocab_words, vocab_embeddings = dataset.get_vocab()
    vocab_embeddings = np.array(vocab_embeddings).astype(np.float32)
    index = faiss.IndexHNSWFlat(768, 32)
    index.add(vocab_embeddings)

    for data in dataset:
        pos_sents, neg_sents, pos_embs, neg_embs = data
        pos_embs, neg_embs = pos_embs.to(device), neg_embs.to(device)
        logits = model(pos_sents, neg_sents)
        D, I = index.search(logits.cpu().numpy(), 20)
        words = [vocab_words[i] for i in I[0]]
        for i, word in enumerate(words):
            print(f"{i + 1}: {word}: {D[0][i]}")
        
        out_word = None
        anchor = None
        for i, word in enumerate(words):
            if word not in pos_sents.split(' '):
                out_word = word
                anchor = torch.tensor(vocab_embeddings[I[0][i]]).unsqueeze(0)
                anchor = anchor.to(device)
                break
        print(f"Output: {out_word}")
        print(f"Positive: {pos_sents}\nNegative: {neg_sents}")
        calc_cos_score(anchor, pos_embs, neg_embs)
        print('')


def main(args):
    device = get_device(args.cuda)
    model = SimpleCodeGiver()
    model.load_state_dict(torch.load(args.m))
    model.to(device)
    model.eval()


    test_dataset = CodeGiverDataset(code_dir=args.code_dir, game_dir=args.geuss_dir)

    test_loop(model, test_dataset, device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-code_dir', type=str, help='Dataset Path', default="/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/data/words.json")
    parser.add_argument('-geuss_dir', type=str, help="", default="/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/data/three_word_data.json")
    parser.add_argument('-m', type=str, help='Model Path', default="/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/saved_models/first_model_medium_validation.out")
    parser.add_argument('-cuda', type=str, help="Whether to use CPU or Cuda, use Y or N", default='Y')
    args = parser.parse_args()
    main(args)