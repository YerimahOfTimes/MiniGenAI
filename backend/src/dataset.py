import torch
from src.config import DATA_PATH, BLOCK_SIZE, BATCH_SIZE, DEVICE
from src.tokenizer import CharTokenizer


class TextDataset:
    def __init__(self):
        self.tokenizer = CharTokenizer()

        with open(DATA_PATH, "r", encoding="utf-8") as f:
            text = f.read()

        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split):
        data = self.train_data if split == "train" else self.val_data

        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))

        x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])

        return x.to(DEVICE), y.to(DEVICE)
