

from functions.add_to_chroma import add_to_chroma
from functions.get_document_loader import load_pdf_documents, load_xml_documents, load_web_documents, load_web_documents2
from functions.get_embedding_function import get_embedding_function
from functions.get_text_spliter import split_text

from enum import Enum
from timeit import default_timer as timer

CHROMA_PDF_PATH = "chroma/pdf"
CHROMA_XML_PATH = "chroma/xml"
CHROMA_WEB_PATH = "chroma/web"

# DATA_PDF_PATH = "data/pdf"
# DATA_XML_PATH = "data/xml"
# DATA_WEB_PATH = "data/web"

DATA_PDF_PATH = "miniset/pdf"
DATA_XML_PATH = "miniset/xml"
DATA_WEB_PATH = "miniset/web"

class EMBEDDING(Enum):
    NOMIC = "nomic-embed-text"
    MXBAI = "mxbai-embed-large"
    SNOWFLAKEv2 = "snowflake-arctic-embed2"
    JINA = "jina/jina-embeddings-v2-base-es"

def main():
    start = timer()

    # documents = load_pdf_documents(DATA_PDF_PATH)
    # documents = load_xml_documents(DATA_XML_PATH)    
    documents = load_web_documents(DATA_WEB_PATH)

    # documents = load_web_documents2(DATA_XML_PATH) # DELETE

    chunks = split_text(documents, 500, 50)

    # for doc in documents:
    #     print(doc.metadata)
    print(chunks)

    end = timer()
    print("%.2fs" % (end-start))

    # add_to_chroma(chunks, CHROMA_PDF_PATH, get_embedding_function(model=EMBEDDING.NOMIC))

if __name__ == "__main__":
    main()