import random
from enum import Enum

class WordLabel(Enum):
    RED = 0
    BLUE = 1
    ASSASSIN = 2

class GameBoard:
    def __init__(self, words: list, pos_num=9, neg_num=9, has_assassin=False):
        nwords = len(words)
        assert nwords < pos_num + neg_num + int(has_assassin)

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
        return self.assassin != None and self.assassin != choice

    def remove_word(self, word: str):
        """
        0 -: Word has been removed
        1 -: Word belongs to other team
        2 -: Assassin has been called end game (should not get here)
        """
        
        for word_list in [self.pos_words, self.neg_words, self.neutral]:
            if word in word_list:
                word_list.remove(word)
                return 0
        if word != self.assassin: return 1
        return 2

class Word:
    def __init__(self, text: str, label: int) -> None:
        self.text = text
        
        assert label >= 0 and label <= 3
        self.choice = label

        