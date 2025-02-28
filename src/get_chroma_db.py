from langchain_chroma import Chroma

def get_chroma_db(path, embedding_function):
    return Chroma(persist_directory=path, embedding_function=embedding_function)