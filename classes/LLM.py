from enum import Enum

# Modelos de lenguaje locales
class LLM(Enum):
    mistral_small = "mistral-small-latest"         # MistralAI: mistral-small 24B
    gpt_41 = "gpt-4.1"                      # OpenAI: GPT-4.1
    llama_3_2_3B = "llama3.2"               # Ollama: Llama 3.2 3B Instruct (2.0 GB)
    
# Modelos de embedding locales
class EMBEDDING(Enum):
    NOMIC = "nomic-embed-text"                  # Ollama: nomic-embed-text-v1.5 (274 MB)
    SNOWFLAKEv2 = "snowflake-arctic-embed2"     # Ollama: snowflake (1.2 GB)
    JINA = "jina/jina-embeddings-v2-base-es"    # Ollama: jina ai (323 MB)
    BGEM3 = "bge-m3"                            # Ollama: BGE-M3 (1.2 GB)   
    TE3S = "text-embedding-3-small"             # OpenAI
