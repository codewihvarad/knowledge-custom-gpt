from langchain_community.document_loaders import PyPDFLoader

def load_text():
    loader = PyPDFLoader(r"data\LLM Engineer.pdf")
    documents = loader.load()
    return documents

results = load_text()
print(results)