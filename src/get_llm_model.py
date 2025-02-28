from langchain_ollama import OllamaLLM

def get_llm_model(model):
    return OllamaLLM(
        model=model, 
        base_url="http://localhost:11434"
    )  