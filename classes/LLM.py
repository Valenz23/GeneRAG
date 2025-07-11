from enum import Enum

# Modelos de lenguaje locales
class LLM(Enum):
    mistral_small = "mistral-small-latest"  # MistralAI: mistral-small 
    llama_3_2_3B = "llama3.2"               # Ollama: Llama 3.2 3B Instruct (2.0 GB)
    gemma3_4B = "gemma3"                    # Ollama: Gemma 3 4B (3.3 GB)
    qwen_2_5_7B = "qwen2.5"                 # Ollama: Qwen2.5 7B Instruct (4.7 GB)
    
# Modelos de embedding locales
class EMBEDDING(Enum):
    NOMIC = "nomic-embed-text"                  # Ollama: nomic-embed-text-v1.5 (274 MB)
    SNOWFLAKEv2 = "snowflake-arctic-embed2"     # Ollama: snowflake (1.2 GB)
    JINA = "jina/jina-embeddings-v2-base-es"    # Ollama: jina ai (323 MB)
    BGEM3 = "bge-m3"                            # Ollama: BGE-M3 (1.2 GB)   
