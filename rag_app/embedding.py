from langchain_huggingface import HuggingFaceEmbeddings

text="hello world"
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

results = get_embeddings(text)
print(results)