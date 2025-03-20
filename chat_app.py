from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

from functions.get_chroma_db import get_chroma_db
from functions.get_embedding_function import get_embedding_function
from functions.get_llm_model import get_llm_model, get_hf_model

import random
from enum import Enum

CHROMA_PDF_PATH = "chroma/pdf"
CHROMA_XML_PATH = "chroma/xml"
CHROMA_WEB_PATH = "chroma/web"

class LLM(Enum):
    llama_3_2_3B = "llama3.2"        # meta 3b 2gb
    deepseek_R1_7B = "deepseek-r1"  #ds 7b 4.7gb
    gemma3_4B = "gemma3"           # google dm 4b 3.3gb
    qwen_2_5_7B = "qwen2.5"          # alibaba 7b 4.7gb
    mistral_7B = "mistral"         # mistral ai 7b 4.1gb    
    hermes_3_8B = "hermes3"         # nous research 8b 4.7gb

class LLM_HF(Enum):
    meta_llama32_3B_Instruct = "meta-llama/Llama-3.2-3B-Instruct"    # 3.21B params
    mistral_7B_Instruct_v03 ="mistralai/Mistral-7B-Instruct-v0.3"   # 7.25B params
    Qwen_QwQ_32B ="Qwen/QwQ-32B"                         # 32.8B params // overloaded
    deepseekR1_Distill_Qwen_32B= "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" # 32.8B params
    # deephermes3_llama_3_3B= "NousResearch/DeepHermes-3-Llama-3-3B-Preview" # 3.21B params // temporary unavailable
    # phi_4_mini_instruct = "microsoft/Phi-4-mini-instruct"   # 3.84B params // temporary unavailable   
    # gemma3_4b = "google/gemma-3-4b-pt"  # 4.3B params // temporary unavailable
    

class EMBEDDING(Enum):
    NOMIC = "nomic-embed-text"                  # nomic team
    MXBAI = "mxbai-embed-large"                 # mixed bread
    Snowflake = "snowflake-arctic-embed2"     # snowflake
    Jina = "jina/jina-embeddings-v2-base-es"    # jina ai

PROMPT_TEMPLATE = """
Tu tarea es responder a la pregunta hecha por el usuario basandote UNÍCAMENTE en el contexto. El contexto son documentos del BOE (Boletín Oficial Español) relacionados con la DANA (Depresión Aislada en Niveles Altos), ocurrida a finales de 2024.
Tus respuestas deben de ser detalladas y bien estructuradas, organizando la información en párrafos y listas si es necesario.
Tus respuestas tienen que contener información sacada del contexto, si no puedes obtener información del contexto indícalo.
Tus respuestas NO DEBEN mencionar cosas como "de acuerdo a la información" o "según el contexto".

---
Pregunta: {question}
Contexto: {context}
"""


WELCOME_MESSAGES = [
    "Mensaje de bienvenida de prueba.",
]

def query(question: str, source_model: str, sel_llm: str):

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

    llm = ""
    if source_model == "Local":
        llm = get_llm_model(sel_llm)
    else:
        llm = get_hf_model(sel_llm)

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

    source_model = st.sidebar.radio(
        "Elige la fuente del modelo",
        ["Local", "HuggingFace"],
        index=0
        )
    
    models = ""
    if source_model == "Local":
        models = [model.value for model in LLM]
    else:
        models = [model.value for model in LLM_HF]

    select_llm = st.sidebar.selectbox(
        label="Modelos disponibles",
        options=models,        
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
                response = query(user_query, source_model, select_llm)
            
        st.session_state.chat_history.append(AIMessage(content=response))   

if __name__ == "__main__":
    main()