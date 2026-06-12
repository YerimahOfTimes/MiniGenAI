from src.word_dataset import WordTextDataset

dataset = WordTextDataset()

x, y = dataset.get_batch("train")

print("Vocabulary size:", dataset.tokenizer.vocab_size)
print("x shape:", x.shape)
print("y shape:", y.shape)
print("First input sample:", x[0])
print("First target sample:", y[0])
