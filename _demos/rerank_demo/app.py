from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever

from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

documents = TextLoader("./state_of_the_union.txt", encoding="utf-8").load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

embedding_model = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma.from_documents(texts,embedding_model)
# vector_store = Chroma(persist_directory="chroma", embedding_function=embedding_model)

retriever = vector_store.as_retriever(search_kwargs={"k": 10})

query = "What is the plan for the economy?"

### Sin reranking ###
# docs = retriever.invoke(query)
# pretty_print_docs(docs)

### Con reranking ###
### https://huggingface.co/BAAI/bge-reranker-v2-m3 ###
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3") # from hugginface
compressor = CrossEncoderReranker(model=model, top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

compressed_docs = compression_retriever.invoke(query)
pretty_print_docs(compressed_docs)