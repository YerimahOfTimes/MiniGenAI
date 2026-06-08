from src.dataset import TextDataset

dataset = TextDataset()

x, y = dataset.get_batch("train")

print("x shape:", x.shape)
print("y shape:", y.shape)
print("First input sample:", x[0])
print("First target sample:", y[0])
