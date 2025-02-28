from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader

def load_pdf_documents(path):
    document_loader = PyPDFDirectoryLoader(path)
    return document_loader.load()