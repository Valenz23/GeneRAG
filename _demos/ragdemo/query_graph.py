from graph_retriever.strategies import Eager
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_graph_retriever import GraphRetriever
from langchain_ollama import ChatOllama

import argparse

from get_embedding_function import get_embedding_function
from plot import plot_graph_retriever

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
    edges=[("source","source"), ("author","author"), ("title","title")]
    traversal_retriever = GraphRetriever(
        store=db,
        edges=edges, 
        strategy=Eager(k=5, start_k=1, max_depth=2) 
    )
    results = traversal_retriever.invoke(query_text)
    plot_graph_retriever(results, edges, "mi_grafo.png")
    context = "\n\n---\n\n".join([doc.page_content for doc in results])
    sources = [
        {
            "id": doc.metadata.get("id", None),
            "source": doc.metadata.get("source", None),
            "author": doc.metadata.get("author", None),
            "title": doc.metadata.get("title", None)
        }
        for doc in results
    ]

    # Prompt and chain
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")     
    
    chain = prompt | llm | StrOutputParser()

    print("Response:", end=" ")
    response_text = ""
    for chunk in chain.stream({"context": context, "question": query_text}):
        print(chunk, end="", flush=True)
        response_text += chunk      
    print(f"\n\nSources:")
    for metadata in sources:
        print(metadata)

    return response_text

if __name__ == "__main__":
    main()