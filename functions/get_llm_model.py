from langchain_ollama import OllamaLLM, ChatOllama

def get_llm_model(model):
    return OllamaLLM(
        model=model, 
        # base_url="http://127.0.0.1:11434",        
        num_ctx=2048
    )  

def get_chat_model(model):
    return ChatOllama(
        model=model, 
        # base_url="http://127.0.0.1:11434"
    )  