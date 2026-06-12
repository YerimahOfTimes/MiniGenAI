import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="MiniGenAI Studio",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 MiniGenAI Studio")
st.write("A Transformer-based Generative AI model built from scratch using PyTorch.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Report",
    "📝 Summary",
    "💻 Code",
    "🔎 Ask with RAG"
])


with tab1:
    st.header("Report Generator")

    project_title = st.text_input("Project Title", "MiniGenAI")

    if st.button("Generate Report"):
        response = requests.post(
            f"{API_URL}/report",
            json={
                "prompt": project_title,
                "max_tokens": 120
            }
        )

        if response.status_code == 200:
            st.success("Report generated")
            st.write(response.json()["output"])
        else:
            st.error("Something went wrong")


with tab2:
    st.header("Summary Generator")

    long_text = st.text_area(
        "Enter text to summarize",
        "Transformers are deep learning architectures that use attention mechanisms to process sequence data effectively."
    )

    if st.button("Generate Summary"):
        response = requests.post(
            f"{API_URL}/summarize",
            json={
                "prompt": long_text,
                "max_tokens": 100
            }
        )

        if response.status_code == 200:
            st.success("Summary generated")
            st.write(response.json()["output"])
        else:
            st.error("Something went wrong")


with tab3:
    st.header("Code Generator")

    code_task = st.text_area(
        "Describe the code you want",
        "Create a Python function that checks if a number is even."
    )

    if st.button("Generate Code"):
        response = requests.post(
            f"{API_URL}/code",
            json={
                "prompt": code_task,
                "max_tokens": 100
            }
        )

        if response.status_code == 200:
            st.success("Code generated")
            st.code(response.json()["output"], language="python")
        else:
            st.error("Something went wrong")


with tab4:
    st.header("Ask with RAG")

    question = st.text_input(
        "Ask a question",
        "What is Retrieval-Augmented Generation?"
    )

    if st.button("Ask Question"):
        response = requests.post(
            f"{API_URL}/ask",
            json={
                "question": question
            }
        )

        if response.status_code == 200:
            data = response.json()

            st.success("Answer generated")
            st.subheader("Answer")
            st.write(data["answer"])

            st.subheader("Retrieved Context")
            for item in data["retrieved_context"]:
                st.info(item)
        else:
            st.error("Something went wrong")
