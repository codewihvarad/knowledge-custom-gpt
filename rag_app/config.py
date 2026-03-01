from langchain_chroma import Chroma

def get_chroma_client():
    return Chroma(collection_name="test_collection")

results = get_chroma_client()
print(results)