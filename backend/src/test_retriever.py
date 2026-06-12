from src.retriever import SimpleRetriever

retriever = SimpleRetriever()

query = "What is Retrieval-Augmented Generation?"

results = retriever.retrieve(query)

print("Query:", query)
print("\nRetrieved Context:")
for result in results:
    print("-", result)
