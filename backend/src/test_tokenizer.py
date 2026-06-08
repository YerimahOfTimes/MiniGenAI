from src.tokenizer import CharTokenizer

tokenizer = CharTokenizer()

text = "Generative AI"

encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)

print("Vocabulary size:", tokenizer.vocab_size)
print("Original:", text)
print("Encoded:", encoded)
print("Decoded:", decoded)
