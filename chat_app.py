from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

from functions.get_chroma_db import get_chroma_db
from functions.get_embedding_function import get_embedding_function
from functions.get_llm_model import get_llm_model

import random
from enum import Enum

CHROMA_PDF_PATH = "chroma/pdf"
CHROMA_XML_PATH = "chroma/xml"
CHROMA_WEB_PATH = "chroma/web"

class LLM(Enum):
    LLAMA32 = "llama3.2"        # meta 3b 2gb
    DEEPSEEKR1 = "deepseek-r1"  #ds 7b 4.7gb
    GEMMA3 = "gemma3"           # google dm 4b 3.3gb
    QWEN25 = "qwen2.5"          # alibaba 7b 4.7gb
    MISTRAL = "mistral"         # mistral ai 7b 4.1gb    
    HERMES3 = "hermes3"         # nous research 8b 4.7gb

class EMBEDDING(Enum):
    NOMIC = "nomic-embed-text"                  # nomic team
    MXBAI = "mxbai-embed-large"                 # mixed bread
    SNOWFLAKEv2 = "snowflake-arctic-embed2"     # snowflake
    JINA = "jina/jina-embeddings-v2-base-es"    # jina ai

PROMPT_TEMPLATE = """
Responde la pregunta: {question}
Tienes que basarte ÚNICAMENTE en el siguiente contexto: {context}
---
Todas las preguntas que haga el usuario estarán relacionadas con la DANA ocurrida en España.  
Debes proporcionar una respuesta detallada y bien estructurada, organizando la información en párrafos y listas si es necesario.  
Si la respuesta no se encuentra en el contexto, indícalo claramente y no inventes datos.
"""


WELCOME_MESSAGES = [
    "Mensaje de bienvenida de prueba.",
]

def query(question: str, sel_llm: str):

    # conexion base de datos y consulta
    db = get_chroma_db(CHROMA_PDF_PATH, get_embedding_function(model=EMBEDDING.NOMIC))
    # db = get_chroma_db(CHROMA_XML_PATH, get_embedding_function(model=EMBEDDING.NOMIC))
    # db = get_chroma_db(CHROMA_WEB_PATH, get_embedding_function(model=EMBEDDING.NOMIC))
    results = db.similarity_search_with_score(question, k=5)

    context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])    # contexto
    metadata = [    # metadatos
        {            
            "score": _score,
            "author": doc.metadata.get("author", None),
            "creator": doc.metadata.get("creator", None),
            "id": doc.metadata.get("id", None),
            "keywords": doc.metadata.get("keywords", None),
            "source": doc.metadata.get("source", None),
            "subject": doc.metadata.get("subject", None),
            "title": doc.metadata.get("title", None)
        }
        for doc, _score in results
    ]
    sources_set = {item["id"] for item in metadata if item.get("id")}   # recursos(set)
    # sources_set = {item["source"] for item in metadata if item.get("source")}   # recursos(set)
    sources = "---\n\n**Recursos**:\n\n" + "\n".join(f"\t🔗 {src}" for src in sources_set)
    
    # Prompt & chain
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = get_llm_model(model=sel_llm)
    chain = prompt | llm | StrOutputParser()

    # generacion de respuesta
    response_text = st.write_stream(chain.stream({"context": context, "question": question}))
    st.write(sources)
    
    return response_text + "\n\n" + sources

#################################################

def main():
    st.set_page_config(page_title="Index", page_icon="images/icon_blue.png")
    st.logo("images/horizontal_blue.png", icon_image="images/icon_blue.png")

    st.markdown("<h1 style='text-align: center;'>¡Bienvenido!</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Usa este chat para obtener información</h2>", unsafe_allow_html=True)

    select_llm = st.sidebar.selectbox(
        label="Selecciona un modelo",
        options=([model.value for model in LLM]),        
        placeholder="Seleccione una opción",
        index=0
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content=random.choice(WELCOME_MESSAGES))]
        
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
            
    user_query = st.chat_input("Escribe tu mensaje aquí ...")

    if user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
        with st.chat_message("Human"):
            st.markdown(user_query)
            
        with st.chat_message("AI"):
            with st.spinner("Pensando ...", show_time=True):
                response = query(user_query, select_llm)
            
        st.session_state.chat_history.append(AIMessage(content=response))   

if __name__ == "__main__":
    main()