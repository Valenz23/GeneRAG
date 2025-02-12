from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, PyPDFDirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from enum import Enum

class LLM(Enum):
    LLAMA32 = "llama3.2"
    DEEPSEEKR1 = "deepseek-r1"
    
DATA_PATH = "ragdemo/data"
# DATA_PATH = "data"

# ----- CARGA DE DOCUMENTOS

# desde web
# web_loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# web_data = web_loader.load()

# desde carpeta
pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
pdf_data = pdf_loader.load() # tarda

# dividir los docs en fragmentos mas pequeños
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=100,
    length_function=len
) 
all_splits = text_splitter.split_documents(pdf_data)


# ------ CREAR VECTORSTORE
local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings) # tarda

# ------ MODELO Y PROMPT
template = """
Responde al usuario teniendo en cuenta solo la información que contienen estos documentos --> docs: {docs}.
Si haces alguna afirmación, asegúrate de citar una fuente que se encuentre en esos documentos.
Si no encuentras una fuente específica, no cites ninguna fuente.
"""

prompt = ChatPromptTemplate.from_template(template)

model = ChatOllama (
    model=LLM.LLAMA32.value,
    base_url="http://localhost:11434"  
)

# formatear los docs en strings
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# chain de procesamiento
chain = {"docs": format_docs} | prompt | model | StrOutputParser()

# ------ CONSULTA
question = "Puedes explicarme que es el procesamiento del lenguaje natural y que aplicaciones tiene?"
docs = vectorstore.similarity_search(question)

for chunks in chain.stream(docs):
    print(chunks, end="")
