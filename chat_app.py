import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage

from classes.chatbot import Chatbot
from classes.LLM import LLM, EMBEDDING


def set_language_model():
    chatbot = st.session_state["chatbot"]
    chatbot.set_language_model(st.session_state["language_model"])

def set_search_type():
    chatbot = st.session_state["chatbot"]
    chatbot.set_search_type(st.session_state["search_type"])

def query(question: str):
    chatbot = st.session_state["chatbot"]

    # retriever = chatbot.get_retriever()           # normal 
    retriever = chatbot.get_compression_retriever() # reranker 

    results = retriever.invoke(question) 
    context = "\n\n---\n\n".join([doc.page_content for doc in results]) 
    metadata = [   
        {            
            "author": doc.metadata.get("author", None),
            "creator": doc.metadata.get("creator", None),
            "id": doc.metadata.get("id", None),
            "keywords": doc.metadata.get("keywords", None),
            "source": doc.metadata.get("source", None),
            "subject": doc.metadata.get("subject", None),
            "title": doc.metadata.get("title", None)
        }
        for doc in results
    ]

    sources_set = {item["id"] for item in metadata if item.get("id")}   
    sources = "---\n\n**Referencias**:\n\n" + "\n".join(f"\t🔗 {src}" for src in sources_set)
    
    response_text = st.write_stream(
        chatbot.answer_query(question, context)
        )
    st.write(sources)
    
    return response_text + "\n\n" + sources

##############################################################################################


def main_page():

    if "chatbot" not in st.session_state:
        st.session_state["chatbot"] = Chatbot(
            chroma_directory="chroma/snow_1024", 
            embedding_model=EMBEDDING.SNOWFLAKEv2,
            k=10,
            top_n=5
        )

    st.set_page_config(page_title="Chat", page_icon="images/icon_logo.png")
    st.logo("images/horizontal_logo.png", icon_image="images/icon_logo.png")
    st.markdown('<h1>¡Bienvenido a <span style="color:#06abeb;">Gene</span><span style="color:#fc6b04;">RAG</span>!</h1>', unsafe_allow_html=True)
    st.subheader("Use este   chat para obtener cualquier información relacionada con la DANA.")
    
    st.sidebar.selectbox(
        label="Chatea con uno de estos LLM",
        options=[model.value for model in LLM],        
        placeholder="Seleccione una opción",
        index=0,
        key="language_model",
        on_change=set_language_model
    )  
    
    st.sidebar.write("---")

    st.sidebar.radio(
        "Seleccione una estrategia de búsqueda",
        ["Similarity", "MMR", "TF-IDF", "BM25", "Grafo"],
        key="search_type",
        index=2,
        on_change=set_search_type
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="¡Hola! ¿En qué puedo ayudarte?")]      
        
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI", avatar="images/avatar_ai.png"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human", avatar="images/avatar_user.png"):
                st.write(message.content)
            
    user_query = st.chat_input("Escribe tu mensaje aquí ...")
    if user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))        
        with st.chat_message("Human", avatar="images/avatar_user.png"):
            st.markdown(user_query)            
        with st.chat_message("AI", avatar="images/avatar_ai.png"):
            with st.spinner("Pensando ...", show_time=True):
                response = query(user_query)            
        st.session_state.chat_history.append(AIMessage(content=response))   


if __name__ == "__main__":
    main_page()