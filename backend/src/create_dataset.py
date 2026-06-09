import os


examples = []

text_topics = [
    "Artificial intelligence is changing how people build software.",
    "Machine learning helps computers learn patterns from data.",
    "Deep learning uses neural networks with many layers.",
    "Generative AI can create text, code, summaries, and reports.",
    "Transformers use attention to understand relationships in data.",
    "A tokenizer converts text into numbers for a machine learning model.",
    "A dataset provides examples that a model can learn from.",
    "Training is the process of updating model weights using data.",
    "A checkpoint stores the learned parameters of a trained model.",
    "PyTorch is a deep learning framework used to build neural networks.",
]

for text in text_topics:
    examples.append(f"<TEXT>\n{text}\n")


summary_pairs = [
    (
        "Artificial intelligence allows machines to perform tasks that normally require human intelligence such as reasoning, learning, and decision making.",
        "Artificial intelligence enables machines to perform intelligent tasks."
    ),
    (
        "Machine learning is a subset of artificial intelligence where systems learn patterns from data instead of being explicitly programmed.",
        "Machine learning helps systems learn from data."
    ),
    (
        "Generative AI creates new content such as text, images, code, reports, summaries, music, and videos.",
        "Generative AI creates new content."
    ),
    (
        "Transformers are deep learning architectures that use attention mechanisms to process sequence data effectively.",
        "Transformers use attention to understand sequences."
    ),
    (
        "A tokenizer converts raw text into numerical tokens that can be processed by a neural network.",
        "A tokenizer converts text into numbers."
    ),
]

for long_text, summary in summary_pairs:
    examples.append(
        f"<SUMMARY>\nLong Text: {long_text}\nSummary: {summary}\n"
    )


reports = [
    {
        "title": "MiniGenAI",
        "problem": "Many beginners use AI tools without understanding how generative models work internally.",
        "method": "A decoder-only Transformer was built from scratch using PyTorch and trained with next-character prediction.",
        "result": "The model learned to generate simple AI-related text.",
        "conclusion": "MiniGenAI demonstrates the foundation of GPT-style generative AI."
    },
    {
        "title": "Fraud Detection AI",
        "problem": "Financial institutions need to identify fraudulent transactions quickly.",
        "method": "A machine learning model was trained using transaction features and evaluated with precision, recall, and F1-score.",
        "result": "The model detected suspicious transaction patterns.",
        "conclusion": "The system can support financial fraud monitoring."
    },
    {
        "title": "Medical Report Generator",
        "problem": "Medical professionals need fast and consistent report generation support.",
        "method": "A vision-language model was designed to generate reports from medical images.",
        "result": "The system produced structured draft medical reports.",
        "conclusion": "Generative AI can assist clinical documentation workflows."
    },
]

for r in reports:
    examples.append(
        f"""<REPORT>
Project Title: {r['title']}
Problem Statement: {r['problem']}
Methodology: {r['method']}
Result: {r['result']}
Conclusion: {r['conclusion']}
"""
    )


code_examples = [
    (
        "Create a Python function that adds two numbers.",
        "def add_numbers(a, b):\n    return a + b"
    ),
    (
        "Create a Python function that checks if a number is even.",
        "def is_even(number):\n    return number % 2 == 0"
    ),
    (
        "Create a Python function that calculates the square of a number.",
        "def square(number):\n    return number * number"
    ),
    (
        "Create a FastAPI health endpoint.",
        'from fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get("/health")\ndef health_check():\n    return {"status": "ok"}'
    ),
]

for task, code in code_examples:
    examples.append(
        f"<CODE>\nTask: {task}\nCode:\n{code}\n"
    )


# Repeat examples to increase training signal
dataset_text = "\n".join(examples * 100)

os.makedirs("data", exist_ok=True)

with open("data/input.txt", "w", encoding="utf-8") as f:
    f.write(dataset_text)

print("Dataset created successfully.")
print("Total characters:", len(dataset_text))
