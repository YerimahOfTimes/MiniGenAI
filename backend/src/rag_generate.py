import torch

from src.config import DEVICE
from src.model import MiniGPT
from src.word_tokenizer import WordTokenizer
from src.retriever import SimpleRetriever
import re


CHECKPOINT_PATH = "checkpoints/minigenai_word_model.pt"


def load_model():
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    tokenizer = WordTokenizer(
        stoi=checkpoint["stoi"],
        itos=checkpoint["itos"]
    )

    model = MiniGPT(checkpoint["vocab_size"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer


def build_prompt(question, context):
    return f"""<ANSWER>
Context: {context}
Question: {question}
Answer:"""


def main():
    question = "What is Retrieval-Augmented Generation?"

    retriever = SimpleRetriever()
    retrieved_context = retriever.retrieve(question, top_k=2)

    context = " ".join(retrieved_context)

    model, tokenizer = load_model()

    prompt = build_prompt(question, context)

    input_ids = tokenizer.encode(prompt)

    input_tensor = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=DEVICE
    )

    stop_token_id = tokenizer.stoi.get("<END>")

    generated_ids = model.generate(
        input_tensor,
        max_new_tokens=80,
        temperature=0.4,
        top_k=10,
        stop_token_id=stop_token_id
    )

    generated_text = tokenizer.decode(
        generated_ids[0].tolist()
    )

    generated_text = generated_text.replace("<END>", "").strip()

    match = re.search(r"Answer\s*:\s*(.*)", generated_text)

    if match:
        generated_text = match.group(1).strip()

    print("\nQuestion:")
    print(question)

    print("\nRetrieved Context:")
    for item in retrieved_context:
        print("-", item)

    print("\nGenerated Answer:")
    print(generated_text)


if __name__ == "__main__":
    main()
