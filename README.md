# 🤖 MiniGenAI

MiniGenAI is a Transformer-based Generative AI system built completely from scratch using PyTorch.

The project demonstrates the core concepts behind modern Generative AI systems, including tokenization, Transformer architectures, text generation, Retrieval-Augmented Generation (RAG), API development, and frontend integration.

---

## 🚀 Features

### Generative AI Tasks

* Report Generation
* Summary Generation
* Code Generation
* Question Answering

### AI Components

* Custom Word-Level Tokenizer
* Transformer Decoder Model
* Training Pipeline
* Checkpoint Saving and Loading
* Retrieval-Augmented Generation (RAG)

### Application Components

* FastAPI Backend
* Streamlit Frontend
* Knowledge Base Search
* Interactive User Interface

---

## 🏗️ Project Architecture

```text
User
 │
 ▼
Streamlit Frontend
 │
 ▼
FastAPI Backend
 │
 ├── Report Generator
 ├── Summary Generator
 ├── Code Generator
 └── RAG Question Answering
          │
          ▼
     Retriever
          │
          ▼
   Knowledge Base
          │
          ▼
   MiniGenAI Model
          │
          ▼
 Generated Response
```

---

## 🧠 Model Architecture

MiniGenAI uses a GPT-style Transformer architecture.

Pipeline:

```text
Dataset
   ↓
Tokenizer
   ↓
Word IDs
   ↓
Embeddings
   ↓
Transformer Blocks
   ↓
Attention Mechanism
   ↓
Prediction Layer
   ↓
Generated Output
```

---

## 📂 Project Structure

```text
MiniGenAI/
│
├── api/
│   └── main.py
│
├── frontend/
│   └── app.py
│
├── knowledge_base/
│   └── mini_knowledge.txt
│
├── checkpoints/
│   └── minigenai_word_model.pt
│
├── src/
│   ├── model.py
│   ├── tokenizer.py
│   ├── word_tokenizer.py
│   ├── dataset.py
│   ├── word_dataset.py
│   ├── train.py
│   ├── train_word.py
│   ├── generate.py
│   ├── generate_word.py
│   ├── retriever.py
│   └── rag_generate.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Technologies Used

* Python
* PyTorch
* FastAPI
* Streamlit
* Scikit-Learn
* Uvicorn
* Requests

---

## 🔍 Retrieval-Augmented Generation (RAG)

MiniGenAI supports Retrieval-Augmented Generation.

Workflow:

```text
Question
   ↓
Retriever
   ↓
Knowledge Base
   ↓
Relevant Context
   ↓
MiniGenAI
   ↓
Generated Answer
```

Example:

Question:

```text
What is Retrieval-Augmented Generation?
```

Answer:

```text
RAG means Retrieval-Augmented Generation. It retrieves relevant external information before generating an answer.
```

---

## 📊 Supported Tasks

### Report Generation

Input:

```text
MiniGenAI
```

Output:

```text
Project Title
Problem Statement
Methodology
Result
Conclusion
```

### Summary Generation

Input:

```text
Long Text
```

Output:

```text
Short Summary
```

### Code Generation

Input:

```text
Task Description
```

Output:

```python
def function():
    pass
```

### Question Answering

Input:

```text
Question
```

Output:

```text
Answer generated using retrieved knowledge.
```

---

## ▶️ Running the Backend

```bash
uvicorn api.main:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## ▶️ Running the Frontend

```bash
streamlit run frontend/app.py
```

Frontend URL:

```text
http://localhost:8501
```

---

## 🎯 Learning Objectives

This project was built to understand:

* Transformer Architecture
* Tokenization
* Attention Mechanisms
* Generative AI
* Retrieval-Augmented Generation
* API Development
* Frontend Integration
* End-to-End AI Systems

---

## 👨‍💻 Author

Yerimah Emmanuel Ogenahotse

Machine Learning Engineer | Generative AI Engineer | AI Systems Builder

GitHub: https://github.com/YerimahOfTimes
