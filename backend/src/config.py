import torch

BATCH_SIZE = 8
BLOCK_SIZE = 64
MAX_ITERS = 1000
EVAL_INTERVAL = 100
LEARNING_RATE = 3e-4
EVAL_ITERS = 10

N_EMBED = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH = "checkpoints/minigenai_model.pt"
DATA_PATH = "data/input.txt"
