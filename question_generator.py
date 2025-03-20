from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from tqdm import tqdm
from timeit import default_timer as timer
import warnings

from functions.get_document_loader import load_pdf_documents


warnings.filterwarnings("ignore", category=FutureWarning)

llm = HuggingFaceEndpoint(
    endpoint_url="mistralai/Mistral-7B-Instruct-v0.3"
)
llm2 = OllamaLLM(
    model="mistral"
)

# QA_generation_prompt = """
# Tu tarea es generar 100 preguntas con sus respuestas basada el contexto aportado.
# Tu preguntas debe de ser formulada en el mismo estilo de preguntas que un usuario afectado pueda hacer en un motor de búsqueda.
# Tu preguntas o respuesta NO debe mencionar cosas como "de acuerdo a la información" o "segun el contexto".
# Tienes que escribir en ESPAÑOL.

# Realiza tu respuesta como sigue:

# Output:::
# Pregunta: (tu pregunta)
# Respuesta: (tu respuesta a la pregunta)

# ---
# Contexto: {context}\n
# Output:::"""

QA_generation_prompt = """
Tu tarea es generar 100 preguntas con su respuestas basadas el contexto aportado.
Tus preguntas debe de ser formuladas en el mismo estilo de preguntas que un usuario afectado pueda hacer en un motor de búsqueda. Los usuarios afectados pueden ser: ciudadanos, trabajadores, empresarios o PYMES.
Tus preguntas DEBEN ser respondidas con un trozo específico del contexto.
Tu preguntas o respuestas NO DEBEN mencionar cosas como "de acuerdo a la información" o "según el contexto".
Tienes que escribir en ESPAÑOL.

Realiza tu respuesta como sigue:

Pregunta: (tu pregunta)
Respuesta: (tu respuesta a la pregunta)
"""

QA_generation_prompt = """
Tu tarea es generar una ÚNICA pregunta con su respuesta basada el contexto aportado.
Tu pregunta debe de ser formulada en el mismo estilo de preguntas que un usuario afectado pueda hacer en un motor de búsqueda.
Tu pregunta o respuesta NO debe mencionar cosas como "de acuerdo a la información" o "contexto".
Tienes que escribir en ESPAÑOL.

Estructura tu respuesta como sigue (en una sóla línea, separando por comas pregunta y respuesta):

Output:::
"tu pregunta generada","tu respuesta a la pregunta generada";

---
Contexto: {context}\n
Output:::"""

documents = load_pdf_documents("data/pdf")

prompt = ChatPromptTemplate.from_template(QA_generation_prompt)
chain =  prompt | llm2

with open("preguntas_mistral3.csv", "w", encoding="utf-8") as f:
    f.write("Pregunta, Respuesta\n")
    for doc in tqdm(documents, desc="Generando hoja de preguntas", unit="páginas"):
        page_content = doc.page_content
        response = chain.invoke({"context":page_content})
        f.write(response + "\n")

# for chunk in chain.stream({"context":page_content[0:5], "x":N_GENERATIONS}):
#     print(chunk, end="", flush=True)

# response = chain.invoke({"context":documents[0:1], "x":N_GENERATIONS})
# print(response)