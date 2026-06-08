import torch
from tqdm import tqdm

from src.config import (
    MAX_ITERS,
    EVAL_INTERVAL,
    EVAL_ITERS,
    LEARNING_RATE,
    DEVICE,
    CHECKPOINT_PATH
)

from src.dataset import TextDataset
from src.model import MiniGPT


@torch.no_grad()
def estimate_loss(model, dataset):
    out = {}

    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)

        for k in range(EVAL_ITERS):
            x, y = dataset.get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()

    return out


def main():
    dataset = TextDataset()
    vocab_size = dataset.tokenizer.vocab_size

    model = MiniGPT(vocab_size).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print(f"Training on device: {DEVICE}")
    print(f"Vocabulary size: {vocab_size}")

    for step in tqdm(range(MAX_ITERS)):
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, dataset)

            print(
                f"Step {step}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        x, y = dataset.get_batch("train")

        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "stoi": dataset.tokenizer.stoi,
            "itos": dataset.tokenizer.itos,
        },
        CHECKPOINT_PATH
    )

    print(f"Model saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
