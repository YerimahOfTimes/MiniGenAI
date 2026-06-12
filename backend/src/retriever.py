from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimpleRetriever:
    def __init__(self, knowledge_path="knowledge_base/mini_knowledge.txt"):
        with open(knowledge_path, "r", encoding="utf-8") as f:
            text = f.read()

        self.chunks = [
            chunk.strip()
            for chunk in text.split("\n\n")
            if chunk.strip()
        ]

        self.vectorizer = TfidfVectorizer()
        self.chunk_vectors = self.vectorizer.fit_transform(self.chunks)

    def retrieve(self, query, top_k=2):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.chunk_vectors)[0]

        top_indices = scores.argsort()[-top_k:][::-1]

        return [self.chunks[i] for i in top_indices]
