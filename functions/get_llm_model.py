from langchain_ollama import OllamaLLM, ChatOllama
from langchain_huggingface import HuggingFaceEndpoint

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def get_llm_model(model):
    return OllamaLLM(
        model=model,      
        # num_ctx=2048
    )  

def get_chat_model(model):
    return ChatOllama(
        model=model
    )  

def get_hf_model(model):
    return HuggingFaceEndpoint(
        endpoint_url=model
    )