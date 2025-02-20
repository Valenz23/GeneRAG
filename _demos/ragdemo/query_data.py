from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

import argparse

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context: {context}
---
Answer the question based on the above context: {question}
"""


def main():

    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)
    context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])    
    sources = [doc.metadata.get("id", None) for doc, _score in results]    
    
    # Prompt and chain
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")     
    
    chain = prompt | llm | StrOutputParser()

    print("Response:", end=" ")
    response_text = ""
    for chunk in chain.stream({"context": context, "question": query_text}):
        print(chunk, end="", flush=True)
        response_text += chunk      
    
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    print(f"\nSources: {sources}")

    return response_text

if __name__ == "__main__":
    main()