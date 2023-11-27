import random
import json
from sentence_transformers import SentenceTransformer
import torch

class GameBoard:
    def __init__(self, words: list, pos_num=9, neg_num=9, has_assassin=False):
        nwords = len(words)
        assert nwords > pos_num + neg_num + int(has_assassin)

        random.shuffle(words)
        self.pos_words = words[:pos_num]
    
        self.pos_words = words[:pos_num]
        self.neg_words = words[pos_num:pos_num + neg_num:]
    
        if has_assassin:
            self.assassin = words[-1]
            self.neutral = words[pos_num + neg_num:-1]
        else:
            self.assassin = None
            self.neutral = words[pos_num + neg_num:]

    def is_assassin(self, choice: str) -> bool:
        return self.assassin != None and self.assassin == choice

    def remove_word(self, word: str):
        """
        0 -: Word has been removed
        1 -: Word belongs to other team
        -1 -: Assassin has been called end game (should not get here)
        """
        
        for word_list in [self.pos_words, self.neg_words, self.neutral]:
            if word in word_list:
                word_list.remove(word)
                return 0
        if not self.is_assassin(word): return 1

        return -1
    
    def get_words_string(self, words_list: list):
        return ' '.join(words_list)
    
    def print_board(self):
        print(f"Board:")
        print(f"Positive Words: {self.pos_words}")
        print(f"Negative Words: {self.neg_words}")
        print(f"Neutral Words: {self.neutral}")
        print(f"Assassin: {self.assassin}")
    

class GameManager:
    def __init__(self, wordfile: str, encoder: SentenceTransformer) -> None:
        self.words = self._get_words(wordfile, allWords=False)
        self.board = GameBoard(self.words)
        self.encoder = encoder

    def _get_words(self, wordfile: str, allWords: bool, numOfWords=25):
        with open(wordfile, 'r') as fp:
            data = json.load(fp)
        words = data['codewords']
        random.shuffle(words)
        if allWords:
            return words
        return words[:numOfWords]
    
    def get_sentences(self):
        pos = " ".join(self.board.pos_words)
        neg = " ".join(self.board.neg_words)
        neutral = " ".join(self.board.neutral)
        return pos, neg, neutral
    
    def get_encoding(self):
        pos, neg, neutral = self.get_sentences()
        with torch.no_grad():
            pos_emb = self.encoder.encode(pos)
            neg_emb = self.encoder.encode(neg, )
            neutral_emb = self.encoder.encode(neutral)
        return pos_emb, neg_emb, neutral_emb

def main():
    wordfile = "/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/data/words.json"
    encoder = SentenceTransformer('all-mpnet-base-v2')
    manager = GameManager(wordfile, encoder)

    encs = manager.get_encoding()

    print("Done")

    
if __name__ == "__main__":
    main()