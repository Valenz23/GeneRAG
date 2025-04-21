from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader, WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import TFIDFRetriever, BM25Retriever
from langchain_core.documents import Document
from graph_retriever.strategies import Eager
from langchain_graph_retriever import GraphRetriever

import streamlit as st
from timeit import default_timer as timer

from dotenv import load_dotenv
import os
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# CHROMA_PDF_PATH = "chroma/pdf"
# CHROMA_XML_PATH = "chroma/xml"
# CHROMA_WEB_PATH = "chroma/web"

# DATA_PDF_PATH = "data/pdf"
# DATA_XML_PATH = "data/xml"
# DATA_WEB_PATH = "data/web"

class Chatbot:

    ### Constructor ### 
    def __init__(self, 
                 language_model: str = "llama3.2", num_ctx: int = 2048, # modelo de lenguaje //     num_ctx -->     2048      4096         8192
                 chunk_size: int = 512, chunk_overlap: int = 50,        # tamaño de los chunks //   size    -->  [[512, 50],[1024, 100], [2048, 200]]
                 embedding_model: str = "nomic-embed-text",             # modelo de embeddings
                 search_type: str = "similarity", k: int = 5,      # tipo de búsqueda //        kwargs -->    5           4             3
                 chroma_directory: str = "chroma",                      # directorio de chroma
                #  chroma_directory: str = "__chroma",                      # directorio de chroma
                 docs_directory: str = "my_data"                        # directorio de documentos                 
                 ):
        
        # self.language_model = ChatOllama(model=language_model, num_ctx=num_ctx, temperature=0, seed=12345)        
        self.language_model = ChatMistralAI(model="mistral-small-latest", mistral_api_key=MISTRAL_API_KEY,temperature=0, random_seed=12345)

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)        
        self.embedding_service = OllamaEmbeddings(model=embedding_model)        
        self.vector_store = Chroma(persist_directory=chroma_directory, embedding_function=self.embedding_service)      
        self.docs_directory = docs_directory      

        self.k = k  

        self.retriever = search_type
        if search_type == "similarity" or search_type == "mmr":
          self.retriever = self.vector_store.as_retriever(search_type=search_type,search_kwargs={"k": k})  
        elif search_type == "tfidf":
            docs = self.__get_documents_from_chroma()
            self.retriever = TFIDFRetriever.from_documents(docs, k=k)
        elif search_type == "bm25":
            docs = self.__get_documents_from_chroma()
            self.retriever = BM25Retriever.from_documents(docs, k=k)
        elif search_type == "grafo":
            self.retriever = GraphRetriever(
                store=self.vector_store,
                edges=[("keywords","keywords"),("author","author"),("subject","subject"),("title","title")],
                strategy=Eager(k=k, start_k=1, max_depth=2)
            )
            

        self.prompt_template = PromptTemplate.from_template(
            """
            Eres un asistente que responde preguntas usando SOLO y ÚNICAMENTE el contexto proporcionado.
            Si la respuesta NO está en contexto, simplemente dí que no lo sabes y no respondas o te inventes la respuesta.
            Tus respuestas DEBEN ser detalladas y estar bien estructuradas, organizando la información en párrafos y listas si es necesario.
            Tus respuestas DEBEN de ser respondidas con las partes de información sacadas del contexto.
            Tus respuestas NO DEBEN mencionar expresiones como "de acuerdo a la información" o "según el contexto".
            
            Si el usuario pide información sobre ayudas, estas también pueden ser subvenciones.            
            Si el usuario pide información sobre subvenciones, estas también pueden ser ayudas.
            Todas las preguntas realizadas por el usuario están relacionadas con la DANA (Depresión Aislada en Niveles Altos) sucedida a finales de 2024.

            Aqui tienes la pregunta y el contexto:
            Pregunta: {question}\n
            Contexto: {context}\n
            """
        )

    ################################################################################################

    ### Carga un documento pdf en la base de datos ### 
    def load_document(self, file_path: str):

        start = timer()
        documents = PyPDFLoader(file_path).load()
        chunks = self.text_splitter.split_documents(documents)
        chunks_with_ids = self.__calculate_chunk_ids(chunks)        

        existing_items = self.vector_store.get(include=[])
        existing_ids = set(existing_items["ids"])

        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            st.write(f"👉 Añadiendo nuevos documentos: **{len(new_chunks)}**")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            self.vector_store.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            st.write("✅ No hay nuevos documentos para añadir")   

        self.__update_tfidf_bm25_retrievers()             
        st.write(f"⏳ Tiempo empleado: **{round(timer() - start,2)} segundos**")

        
    ### Carga un directorio de documentos pdf  en la DB ###
    def populate_db(self, directory_path: str):
        start = timer()
        documents = PyPDFDirectoryLoader(directory_path).load()
        chunks = self.text_splitter.split_documents(documents)
        chunks_with_ids = self.__calculate_chunk_ids(chunks)

        existing_items = self.vector_store.get(include=[])
        existing_ids = set(existing_items["ids"])

        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            st.write(f"👉 Añadiendo nuevos documentos: **{len(new_chunks)}**")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            self.vector_store.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            st.write("✅ No hay nuevos documentos para añadir")

        self.__update_tfidf_bm25_retrievers()
        st.write(f"⏳ Tiempo empleado: **{round(timer() - start,2)} segundos**")


    ### Genera IDs para los chunks ###
    def __calculate_chunk_ids(self, chunks):
        """ Genera ids para los chunks
        """

        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            chunk.metadata["id"] = chunk_id

        return chunks
    
    ### actualizar los retrievers en caso de que sean tf-idf o bm25
    def __update_tfidf_bm25_retrievers(self):        
        docs = self.__get_documents_from_chroma()
        if self.retriever == "tfidf":
            self.retriever = TFIDFRetriever.from_documents(docs, k=self.k)
        elif self.retriever == "bm25":
            self.retriever = BM25Retriever.from_documents(docs, k=self.k)    
    
    ################################################################################################
    
    ### Retorna el stream con la respuesta ###
    def answer_query(self, query: str, context: str):
        query_chain = (
            {"context": self.retriever, "question": RunnablePassthrough(), "context": RunnablePassthrough()}
            | self.prompt_template
            | self.language_model
            | StrOutputParser()
        )
        return query_chain.stream({"question": query, "context": context})
    
    def answer_query2(self, query: str, context: str):
        query_chain = (
            {"context": self.retriever, "question": RunnablePassthrough(), "context": RunnablePassthrough()}
            | self.prompt_template
            | self.language_model
            | StrOutputParser()
        )
        return query_chain.invoke({"question": query, "context": context})
    
    
    # def load_web_documents(path):    
        # webs = []

        # for file in os.listdir(path):
        #     file_path = os.path.join(path, file)
        #     with open(file_path, "r") as file:
        #         url = file.read().strip()
        #         webs.append(url)

        # document_loader = WebBaseLoader(webs)
        # return document_loader.load()


    ################################################################################################
    
    ### GETTERS & SETTERS ###

    def get_vector_store(self):
        return self.vector_store
    
    
    def get_retriever(self):
        return self.retriever
    
    def set_search_type(self, search_type: str, k: int = 5):
        # print(f"Busqueda por: {search_type}")
        if search_type == "similarity" or search_type == "mmr":
          self.retriever = self.vector_store.as_retriever(search_type=search_type,search_kwargs={"k": k})  
        elif search_type == "tfidf":
            docs = self.__get_documents_from_chroma()
            self.retriever = TFIDFRetriever.from_documents(docs, k=k)
        elif search_type == "bm25":
            docs = self.__get_documents_from_chroma()
            self.retriever = BM25Retriever.from_documents(docs, k=k)
        elif search_type == "grafo":
            self.retriever = GraphRetriever(
                store=self.vector_store,
                edges=[("keywords","keywords"),("author","author"),("subject","subject"),("title","title")],
                strategy=Eager(k=k, start_k=1, max_depth=2)
            )      
    

    def get_docs_directory(self):
        return self.docs_directory
    

    def get_language_model(self):
        return self.language_model
    def set_language_model(self, language_model: str, num_ctx: int = 2048):
        # self.language_model = ChatOllama(model=language_model, num_ctx=num_ctx, temperature=0, seed=12345)        
        self.language_model = ChatMistralAI(model="mistral-small-latest", mistral_api_key=MISTRAL_API_KEY,temperature=0, random_seed=12345)

    def __get_documents_from_chroma(self):
        """ Extrae los documentos del vector store y los convierte a objetos Document """
        chroma_data = self.vector_store.get()
        documents = []

        ids = chroma_data['ids']
        contents = chroma_data['documents']
        metadatas = chroma_data['metadatas']

        # print(metadatas[0:3])

        for doc_id, content, metadata in zip(ids, contents, metadatas):
            documents.append(Document(page_content=content, metadata=metadata))

        return documents

        