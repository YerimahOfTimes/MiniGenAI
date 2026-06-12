from src.word_tokenizer import WordTokenizer

tokenizer = WordTokenizer()

text = "<REPORT>\nProject Title: MiniGenAI"

encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)

print("Vocabulary size:", tokenizer.vocab_size)
print("Original:", text)
print("Encoded:", encoded)
print("Decoded:", decoded)
