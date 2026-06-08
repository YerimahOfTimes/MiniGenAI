import torch

from src.dataset import TextDataset
from src.model import MiniGPT
from src.config import DEVICE

dataset = TextDataset()
vocab_size = dataset.tokenizer.vocab_size

model = MiniGPT(vocab_size).to(DEVICE)

x, y = dataset.get_batch("train")

logits, loss = model(x, y)

print("Vocab size:", vocab_size)
print("Input shape:", x.shape)
print("Logits shape:", logits.shape)
print("Loss:", loss.item())
