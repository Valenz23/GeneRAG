from functions.add_to_chroma import add_to_chroma
from functions.get_document_loader import load_pdf_documents, load_xml_documents, load_web_documents
from functions.get_embedding_function import get_embedding_function
from functions.get_text_spliter import split_text

from enum import Enum
from timeit import default_timer as timer
import streamlit as st

CHROMA_PDF_PATH = "chroma/pdf"
CHROMA_XML_PATH = "chroma/xml"
CHROMA_WEB_PATH = "chroma/web"

DATA_PDF_PATH = "data/pdf"
DATA_XML_PATH = "data/xml"
DATA_WEB_PATH = "data/web"

class EMBEDDING(Enum):
    NOMIC = "nomic-embed-text"
    MXBAI = "mxbai-embed-large"
    SNOWFLAKEv2 = "snowflake-arctic-embed2"
    JINA = "jina/jina-embeddings-v2-base-es"



def main():
    st.set_page_config(page_title="Index", page_icon="images/icon_blue.png")
    st.logo("images/horizontal_blue.png", icon_image="images/icon_blue.png")

    st.markdown("<h1 style='text-align: center;'>Actualizar DB</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Pulse el botón para actualizar la DB</h2>", unsafe_allow_html=True)

    if st.button("Actualizar", icon=":material/database:"):
        update_db()
    
def update_db():
    with st.spinner("Actualizando DB, espere ...", show_time=True):
        
        size, overlap = 500, 25

        container1 = st.container(border=True)
        container1.write("• pdf db")
        start = timer()
        documents = load_pdf_documents(DATA_PDF_PATH)
        chunks = split_text(documents, size, overlap)
        # for doc in documents:
        #     print(doc.metadata)
        # print(chunks)
        end = timer()
        container1.write("Carga & Split: %.2fs" % (end-start))
        start = timer()
        add_to_chroma(chunks, CHROMA_PDF_PATH, get_embedding_function(model=EMBEDDING.NOMIC), container1)
        end = timer()
        container1.write("Actualización finalizada: %.2fs" % (end-start))
        
        container2 = st.container(border=True)
        container2.write("• xml db")
        start = timer()
        documents = load_xml_documents(DATA_XML_PATH) 
        chunks = split_text(documents, size, overlap)
        end = timer()
        container2.write("Carga & Split: %.2fs" % (end-start))
        start = timer()
        add_to_chroma(chunks, CHROMA_XML_PATH, get_embedding_function(model=EMBEDDING.NOMIC), container2)
        end = timer()
        container2.write("Actualización finalizada: %.2fs" % (end-start))
        
        container3 = st.container(border=True)
        container3.write("• web db")
        start = timer()
        documents = load_web_documents(DATA_WEB_PATH)
        chunks = split_text(documents, size, overlap)
        end = timer()
        container3.write("Carga & Split: %.2fs" % (end-start))
        start = timer()
        add_to_chroma(chunks, CHROMA_WEB_PATH, get_embedding_function(model=EMBEDDING.NOMIC), container3)
        end = timer()
        container3.write("Actualización finalizada: %.2fs" % (end-start))    

    st.success("DB actualizada!")

if __name__ == "__main__":
    main()