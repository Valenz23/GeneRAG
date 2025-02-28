from langchain_ollama import OllamaEmbeddings

def get_embedding_function(model):
    embeddings =  OllamaEmbeddings(
        model=model
    )    
    return embeddings