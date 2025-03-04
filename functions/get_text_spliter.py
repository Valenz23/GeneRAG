from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(documents: list[Document], chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)