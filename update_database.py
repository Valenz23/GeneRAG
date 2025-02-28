

from src.add_to_chroma import add_to_chroma
from src.get_document_loader import load_pdf_documents
from src.get_embedding_function import get_embedding_function
from src.get_text_spliter import split_text

from enum import Enum

CHROMA_PDF_PATH = "chroma/pdf"
DATA_PDF_PATH = "data/pdf"

class EMBEDDING(Enum):
    NOMIC = "nomic-embed-text"

def main():
    documents = load_pdf_documents(DATA_PDF_PATH)
    # for doc in documents:
    #     print(doc.metadata)
    chunks = split_text(documents, 500, 50)
    # print(chunks)
    add_to_chroma(chunks, CHROMA_PDF_PATH, get_embedding_function(model=EMBEDDING.NOMIC))

if __name__ == "__main__":
    main()