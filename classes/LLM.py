from enum import Enum

# Modelos de lenguaje locales
class LLM(Enum):
    llama_3_2_3B = "llama3.2"       # Llama 3.2 3B Instruct (2.0 GB)
    gemma3_4B = "gemma3"            # Gemma 3 4B (3.3 GB)
    deepseek_R1_7B = "deepseek-r1"  # DeepSeek R1 Distill Qwen 7B (2.0 GB)
    mistral_7B = "mistral"          # Mistral-7B-Instruct-v0.3 (4.1 GB)
    qwen_2_5_7B = "qwen2.5"         # Qwen2.5 7B Instruct (4.7 GB)
    hermes_3_8B = "hermes3"         # Hermes 3 Llama 3.1 8B (4.7 GB)
    llama_3_1_8B = "llama3.1:8b"    # Meta Llama 3.1 8B Instruct (4.9 GB)
    
# Modelos de embedding locales
class EMBEDDING(Enum):
    NOMIC = "nomic-embed-text"                  # nomic-embed-text-v1.5 (274 MB)
    SNOWFLAKEv2 = "snowflake-arctic-embed2"     # snowflake (1.2 GB)
    JINA = "jina/jina-embeddings-v2-base-es"    # jina ai (323 MB)
    BGEM3 = "bge-m3"                            # BGE-M3 (1.2 GB)   
