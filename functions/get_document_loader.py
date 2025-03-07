from langchain_community.document_loaders import UnstructuredXMLLoader, WebBaseLoader
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader

import os


def load_pdf_documents(path):
    document_loader = PyPDFDirectoryLoader(path)
    return document_loader.load()

def load_xml_documents(path):

    documents = []

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        document_loader = UnstructuredXMLLoader(file_path)
        documents.extend(document_loader.load())

    return documents

def load_web_documents(path):
    
    webs = []

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        with open(file_path, "r") as file:
            url = file.read().strip()
            webs.append(url)

    document_loader = WebBaseLoader(webs)
    return document_loader.load()
