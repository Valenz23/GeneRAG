from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.get_chroma_db import get_chroma_db
from src.get_embedding_function import get_embedding_function
from src.get_llm_model import get_llm_model

import argparse
from enum import Enum

CHROMA_PDF_PATH = "chroma/pdf"

class LLM(Enum):
    LLAMA32 = "llama3.2"

class EMBEDDING(Enum):
    NOMIC = "nomic-embed-text"

PROMPT_TEMPLATE = """
Responde la pregunta basandote solamente en el siguiente contexto: {context}
---
Responde la pregunta basandote en el contexto anterior: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query(query_text)

def query(question: str):

    db = get_chroma_db(CHROMA_PDF_PATH, get_embedding_function(model=EMBEDDING.NOMIC))
    results = db.similarity_search_with_score(question, k=5)
    context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])    
    sources = [
        {            
            "score": _score,
            "id": doc.metadata.get("id", None),
            "source": doc.metadata.get("source", None),
            "author": doc.metadata.get("author", None),
            "title": doc.metadata.get("title", None)
        }
        for doc, _score in results
    ]
    
    # Prompt and chain
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = get_llm_model(model=LLM.LLAMA32)
    
    chain = prompt | llm | StrOutputParser()

    print("Response:", end=" ")
    response_text = ""
    for chunk in chain.stream({"context": context, "question": question}):
        print(chunk, end="", flush=True)
        response_text += chunk          
    print(f"\n\nSources:")
    for metadata in sources:
        print(metadata)

    return response_text

if __name__ == "__main__":
    main()