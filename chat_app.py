import streamlit as st

from langchain_core.messages import AIMessage, HumanMessage

from classes.chatbot import Chatbot
from classes.LLM import LLM


### Cambiar el modelo de lenguaje ###
def set_language_model():
    chatbot = st.session_state["chatbot"]
    chatbot.set_language_model(st.session_state["language_model"])


### Genera la respuesta ###
def query(question: str):
    chatbot = st.session_state["chatbot"]
    results = chatbot.get_retriever().batch([question])

    context = "\n\n---\n\n".join([doc.page_content for doc in results[0]]) # contexto
    metadata = [    # metadatos
        {            
            "author": doc.metadata.get("author", None),
            "creator": doc.metadata.get("creator", None),
            "id": doc.metadata.get("id", None),
            "keywords": doc.metadata.get("keywords", None),
            "source": doc.metadata.get("source", None),
            "subject": doc.metadata.get("subject", None),
            "title": doc.metadata.get("title", None)
        }
        for doc in results[0]
    ]
    
    print(metadata)

    sources_set = {item["id"] for item in metadata if item.get("id")}   # recursos(set)
    sources = "---\n\n**Recursos**:\n\n" + "\n".join(f"\t🔗 {src}" for src in sources_set)
    
    response_text = st.write_stream(
        chatbot.answer_query(question, context)
        )
    st.write(sources)
    
    return response_text + "\n\n" + sources

##############################################################################################


def main_page():

    if "chatbot" not in st.session_state:
        st.session_state["chatbot"] = Chatbot()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="¡Hola! Soy un asistente virtual. ¿En qué puedo ayudarte?")]

    st.set_page_config(page_title="Chat", page_icon="images/icon_blue.png")
    st.logo("images/horizontal_blue.png", icon_image="images/icon_blue.png")

    st.header("¡Bienvenido!")
    st.subheader("Usa este chat para obtener información")
    
    st.sidebar.selectbox(
        label="Chatea con uno de estos LLM",
        options=[model.value for model in LLM],        
        placeholder="Seleccione una opción",
        index=0,
        key="language_model",
        on_change=set_language_model
    )  
    
      
        
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
                response = query(user_query)            
        st.session_state.chat_history.append(AIMessage(content=response))   


if __name__ == "__main__":
    main_page()