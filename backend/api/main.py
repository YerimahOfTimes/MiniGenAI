import re
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from src.config import DEVICE
from src.model import MiniGPT
from src.word_tokenizer import WordTokenizer
from src.retriever import SimpleRetriever


CHECKPOINT_PATH = "checkpoints/minigenai_word_model.pt"

app = FastAPI(
    title="MiniGenAI API",
    description="A Transformer-based Generative AI model built from scratch.",
    version="1.0.0"
)


class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 80


class QuestionRequest(BaseModel):
    question: str


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


model, tokenizer = load_model()
retriever = SimpleRetriever()


def generate_text(prompt, max_tokens=80):
    input_ids = tokenizer.encode(prompt)

    input_tensor = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=DEVICE
    )

    stop_token_id = tokenizer.stoi.get("<END>")

    generated_ids = model.generate(
        input_tensor,
        max_new_tokens=max_tokens,
        temperature=0.4,
        top_k=10,
        stop_token_id=stop_token_id
    )

    generated_text = tokenizer.decode(
        generated_ids[0].tolist()
    )

    generated_text = generated_text.replace("<END>", "").strip()

    return generated_text


def clean_answer(text):
    match = re.search(r"Answer\s*:\s*(.*)", text)
    if match:
        return match.group(1).strip()
    return text.strip()


@app.get("/")
def home():
    return {
        "message": "MiniGenAI API is running",
        "model": "Word-level Transformer from scratch"
    }


@app.post("/generate")
def generate(request: PromptRequest):
    output = generate_text(request.prompt, request.max_tokens)
    return {"output": output}


@app.post("/report")
def report(request: PromptRequest):
    prompt = f"<REPORT>\nProject Title: {request.prompt}\n"
    output = generate_text(prompt, request.max_tokens)
    return {"output": output}


@app.post("/summarize")
def summarize(request: PromptRequest):
    prompt = f"<SUMMARY>\nLong Text: {request.prompt}\n"
    output = generate_text(prompt, request.max_tokens)
    return {"output": output}


@app.post("/code")
def code(request: PromptRequest):
    prompt = f"<CODE>\nTask: {request.prompt}\n"
    output = generate_text(prompt, request.max_tokens)
    return {"output": output}


@app.post("/ask")
def ask(request: QuestionRequest):
    retrieved_context = retriever.retrieve(request.question, top_k=2)
    context = " ".join(retrieved_context)

    prompt = f"""<ANSWER>
Context: {context}
Question: {request.question}
Answer:"""

    output = generate_text(prompt, max_tokens=80)
    answer = clean_answer(output)

    return {
        "question": request.question,
        "retrieved_context": retrieved_context,
        "answer": answer
    }
