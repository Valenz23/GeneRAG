from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
import random


############################
### CONFIGURACION PREVIA ###
############################

# Algunos de los modelos de lenguaje disponibles
MODELOS = [
    "llama3.2:1b",          # 0
    "llama3.2:latest",      # 1
    "llama3.1:latest",      # 2 
    "deepseek-r1:1.5b",     # 3
    "deepseek-r1:latest"    # 4
]

# Mensaje de bienbenida del chatbot
WELCOME_MESSAGES = [
    "¡Bienvenido al concesionario virtual! 🚗💨 Soy Carina, tu asistente personal. ¿Qué tipo de coche estás buscando hoy?",
    "¡Hola! Soy Carina, estoy aquí para ayudarte a encontrar el coche perfecto. ¿Prefieres un deportivo, familiar o eléctrico?",
    "¡Arranquemos motores! 🏎️ Soy Carina, listos para encontrar juntos tu próximo coche ideal. Cuéntame qué necesitas."
]

# Contexto, tambien agrega el historial de conversacion
CHAT_PROMPT_TEMPLATE = """
    Eres un asistente profesional de venta de coches. 
    Tu meta es ayudar a los clientes a encontrar el coche perfecto basado en sus necesidades y preferencias.
    No vuelvas a saludar al cliente.
    No abuses de las exclamaciones.
    Manten una actitud positiva y amigable.
    Usa iconos a menudo si es necesario.
    Usa euros para los precios.    
    Procura siempre responder en español.
    Responde a las preguntas teniendo en cuenta el historial de la conversación, pero no lo menciones.
    Chat history: {chat_history}
    User question: {user_question}
"""

#################
### FUNCIONES ###
#################

# funcion para obtener respuesta
def get_response(user_query, chat_history):    

    llm = ChatOllama(
        model=MODELOS[1],                   # <---- cambiar modelo
        base_url="http://ollama:11434",
        temperature=0.7
    )

    prompt = ChatPromptTemplate.from_template(CHAT_PROMPT_TEMPLATE)

    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "chat_history": chat_history, 
        "user_question": user_query
    })


####################################
### PAGINA Y CHATBOT (STREAMLIT) ###
####################################

st.set_page_config(page_title="DriveNet", page_icon="🚗")
st.markdown("<h1 style='text-align: center;'>¡Bienvenido a DriveNet! 🚗💨</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Encuentra el coche de tus sueños con la ayuda de Carina, tu asistente virtual. ¿Qué tipo de coche estás buscando?</h2>", unsafe_allow_html=True)

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
        response = st.write_stream(get_response(user_query, st.session_state.chat_history))
        
    st.session_state.chat_history.append(AIMessage(content=response))    