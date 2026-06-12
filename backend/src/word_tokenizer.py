import re
from src.config import DATA_PATH


class WordTokenizer:
    def __init__(self, stoi=None, itos=None):
        if stoi is not None and itos is not None:
            self.stoi = stoi
            self.itos = itos
            self.vocab_size = len(stoi)
        else:
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                text = f.read()

            tokens = self._basic_tokenize(text)

            vocab = sorted(list(set(tokens)))

            self.stoi = {token: i for i, token in enumerate(vocab)}
            self.itos = {i: token for token, i in self.stoi.items()}
            self.vocab_size = len(vocab)

    def _basic_tokenize(self, text):
        """
        Splits text into words, punctuation, and special tags.
        Example:
        <REPORT> Project Title: MiniGenAI
        becomes:
        ['<REPORT>', 'Project', 'Title', ':', 'MiniGenAI']
        """
        pattern = r"<[^>]+>|\w+|[^\w\s]"
        return re.findall(pattern, text)

    def encode(self, text):
        tokens = self._basic_tokenize(text)
        return [self.stoi[token] for token in tokens if token in self.stoi]

    def decode(self, ids):
        tokens = [self.itos[int(i)] for i in ids]

        text = " ".join(tokens)

        text = text.replace(" .", ".")
        text = text.replace(" ,", ",")
        text = text.replace(" :", ":")
        text = text.replace(" ;", ";")
        text = text.replace(" !", "!")
        text = text.replace(" ?", "?")
        text = text.replace("( ", "(")
        text = text.replace(" )", ")")
        text = text.replace("\n ", "\n")

        return text
