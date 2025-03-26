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

# Modelos de lenguaje disponibles en Hugging Face
class LLM_HF(Enum):
    meta_llama32_3B_Instruct = "meta-llama/Llama-3.2-3B-Instruct"    # 3.21B params
    mistral_7B_Instruct_v03 ="mistralai/Mistral-7B-Instruct-v0.3"   # 7.25B params
    Qwen_QwQ_32B ="Qwen/QwQ-32B"                         # 32.8B params // overloaded
    deepseekR1_Distill_Qwen_32B= "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" # 32.8B params
    # deephermes3_llama_3_3B= "NousResearch/DeepHermes-3-Llama-3-3B-Preview" # 3.21B params // temporary unavailable
    # phi_4_mini_instruct = "microsoft/Phi-4-mini-instruct"   # 3.84B params // temporary unavailable   
    # gemma3_4b = "google/gemma-3-4b-pt"  # 4.3B params // temporary unavailable
    
# Modelos de embedding locales
class EMBEDDING(Enum):
    NOMIC = "nomic-embed-text"                  # nomic-embed-text-v1.5 (274 MB)
    MXBAI = "mxbai-embed-large"                 # mixed bread (669 MB)
    SNOWFLAKEv2 = "snowflake-arctic-embed2"     # snowflake (1.2 GB)
    JINA = "jina/jina-embeddings-v2-base-es"    # jina ai (323 MB)
