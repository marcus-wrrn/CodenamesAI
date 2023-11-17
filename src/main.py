from sentence_transformers import SentenceTransformer
import torch
from processing.processing import Processing

def main():
    encoder = SentenceTransformer('all-mpnet-base-v2')
    lines = Processing(encoder, filepath="../data/words.json", download=False)
    lines.to_json("../data/words.json")


if __name__ == "__main__":
    main()