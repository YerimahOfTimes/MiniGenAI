from src.config import DATA_PATH


class CharTokenizer:
    def __init__(self, stoi=None, itos=None):
        if stoi is not None and itos is not None:
            self.stoi = stoi
            self.itos = itos
            self.vocab_size = len(stoi)
        else:
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                text = f.read()

            chars = sorted(list(set(text)))

            self.vocab_size = len(chars)
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text):
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids):
        return "".join([self.itos[int(i)] for i in ids])
