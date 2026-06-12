import torch

from src.config import DEVICE
from src.model import MiniGPT
from src.word_tokenizer import WordTokenizer


CHECKPOINT_PATH = "checkpoints/minigenai_word_model.pt"


def main():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    tokenizer = WordTokenizer(
        stoi=checkpoint["stoi"],
        itos=checkpoint["itos"]
    )

    model = MiniGPT(checkpoint["vocab_size"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt = "<CODE>\nTask:"

    context = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=DEVICE
    )

    stop_token_id = tokenizer.stoi.get("<END>")

    generated_ids = model.generate(
        context,
        max_new_tokens=120,
        temperature=0.4,
        top_k=10,
        stop_token_id=stop_token_id
    )

    generated_text = tokenizer.decode(
        generated_ids[0].tolist()
    )

    generated_text = generated_text.replace("<END>", "").strip()

    print("\nGenerated Text:\n")
    print(generated_text)


if __name__ == "__main__":
    main()
