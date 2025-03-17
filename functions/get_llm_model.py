from langchain_ollama import OllamaLLM, ChatOllama
from langchain_huggingface import HuggingFaceEndpoint

from dotenv import load_dotenv
import os
import warnings

# warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

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
        endpoint_url=model,
        # max_new_tokens=512,
        top_k=10,
        top_p=0.95,
        typical_p=0.95,
        temperature=0.01,
        repetition_penalty=1.03,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )