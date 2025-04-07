from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFDirectoryLoader

from tqdm import tqdm
from timeit import default_timer as timer
import time
import warnings

from dotenv import load_dotenv
import os
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

warnings.filterwarnings("ignore", category=FutureWarning)

# llm = HuggingFaceEndpoint(endpoint_url="mistralai/Mistral-7B-Instruct-v0.3")
# llm2 = OllamaLLM(model="mistral")
llm3 = ChatMistralAI(model="mistral-small-latest", mistral_api_key=MISTRAL_API_KEY,temperature=0.1, random_seed=12345)

# QA_generation_prompt = """
# Tu tarea es generar una pregunta con su respectiva respuesta basándote únicamente en el contexto aportado.
# Tus preguntas deben estar relacionadas con la DANA (Depresión Aislada en Niveles Altos) ocurrida a finales de 2024.
# Las preguntas generadas deben tener un formato de preguntas frecuentas (FAQ).
# Debes formular tus preguntas imitando el estilo que usaría una persona afectada por la catástrofe buscando información.
# Tus preguntas deben poder ser respondidas usando una pieza específica del contexto.
# No debes mencionar cosas como "de acuerdo a la información" o "según el contexto".
# Escribe en ESPAÑOL.
# ---
# Realiza tu respuestas como sigue:

# Pregunta: tu pregunta.
# Respuesta: tu respuesta.

# ---
# Contexto: {context}\n
# Output:::"""

QA_generation_prompt = """
Tu tarea es generar una ÚNICA pregunta con su respectiva respuesta basándote únicamente en el contexto aportado.
Tus preguntas deben estar relacionadas con la DANA (Depresión Aislada en Niveles Altos) ocurrida a finales de 2024.
Las preguntas generadas deben tener un formato de preguntas frecuentas (FAQ).
Debes formular tus preguntas imitando el estilo que usaría una persona afectada por la catástrofe buscando información.
Tus preguntas deben poder ser respondidas usando una pieza específica del contexto.
No debes mencionar cosas como "de acuerdo a la información" o "según el contexto".
Escribe en ESPAÑOL.
---
Realiza tu respuestas como sigue, manteniendo un formato separado por comas (csv):

"tu pregunta","tu respuesta"

---
Contexto: {context}\n
Output:::"""

documents = PyPDFDirectoryLoader("../my_data").load()
prompt = ChatPromptTemplate.from_template(QA_generation_prompt)
chain =  prompt | llm3 | StrOutputParser()

with open("preguntas_mistral.csv", "w", encoding="utf-8") as f:
    f.write("Pregunta, Respuesta\n")
    for doc in tqdm(documents, desc="Generando hoja de preguntas", unit="páginas"):
        page_content = doc.page_content
        response = chain.invoke({"context":page_content})
        f.write(response + "\n")
        time.sleep(5)   # to avoid rate limit issues