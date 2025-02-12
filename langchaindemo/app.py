from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from enum import Enum

class LLM(Enum):
    LLAMA32 = "llama3.2"
    DEEPSEEKR1 = "deepseek-r1"

template = """Question: {question}
Respuesta: Responde a lo que el usuario pida."""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(
    model=LLM.LLAMA32.value,
    base_url="http://localhost:11434"    
)

chain = prompt | model | StrOutputParser()

query={
    "question": "¿Que es langchain?"
}

for chunks in chain.stream(query):
    print(chunks, end="")