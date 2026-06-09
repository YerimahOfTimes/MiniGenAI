import torch

from src.config import DEVICE, CHECKPOINT_PATH
from src.model import MiniGPT
from src.tokenizer import CharTokenizer


def main():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    tokenizer = CharTokenizer(
        stoi=checkpoint["stoi"],
        itos=checkpoint["itos"]
    )

    model = MiniGPT(checkpoint["vocab_size"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt = "<REPORT>\nProject Title: MiniGenAI\n"

    context = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=DEVICE
    )

    generated_ids = model.generate(
        context,
        max_new_tokens=400,
        temperature=0.3,
        top_k=5
    )

    generated_text = tokenizer.decode(
        generated_ids[0].tolist()
    )

    print("\nGenerated Text:\n")
    print(generated_text)


if __name__ == "__main__":
    main()
