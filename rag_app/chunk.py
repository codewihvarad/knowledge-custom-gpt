from langchain_text_splitters import RecursiveCharacterTextSplitter

text="""The LangSmith Python SDK provides a convenient way to interact with the LangSmith API, allowing you to"""

def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10,
        chunk_overlap=2,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.create_documents([text])

results = chunk_text(text)
print(results)