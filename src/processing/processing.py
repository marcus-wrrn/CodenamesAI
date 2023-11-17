from sentence_transformers import SentenceTransformer
import json



class Processing:
    def __init__(self, encoder: SentenceTransformer | None, filepath="../data/wordlist-eng.txt", download=False) -> None:
        print("What?")
        if download:
            self.words = self._get_words(filepath)
            self.embeddings = self._get_embeddings(encoder, self.words)
        else:
            with open(filepath, 'r') as file:
                data = json.load(file)
            self.words = data['words']
            self.embeddings = data['embeddings']
        

    # Initializers
    def _get_words(self, filepath):
        with open(filepath, 'r') as file:
            lines = [line.replace('\n', '').lower() for line in file.readlines()]
        return lines
    
    def _get_embeddings(self, encoder: SentenceTransformer | None, words: list):
        if type(encoder) == SentenceTransformer:
            return encoder.encode(words)
        
        encoder = SentenceTransformer("all-mpnet-base-v2")
        return encoder.encode(words)
    
    def to_json(self, filepath: str):
        data = {
            'words': [word for word in self.words],
            'embeddings': [[float(x) for x in embedding] for embedding in self.embeddings]
        }

        with open(filepath, 'w') as file:
            json.dump(data, file)

if __name__ == "__main__":
    proc = Processing(encoder=None, filepath="/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/data/words.json")
    ...
