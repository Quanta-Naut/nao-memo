import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
    
    # Gemini Config
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GENERATIVE_MODEL = "gemini-2.5-flash-lite"
    EMBEDDING_MODEL = "models/gemini-embedding-001"
    
    # Ollama Config
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

    # Common Config
    PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
    
    # Auto-switch DB path if not set, to avoid mixing embeddings
    _db_env = os.getenv("DB_PATH")
    if _db_env:
        DB_PATH = _db_env
    else:
        DB_PATH = "memory_ollama.db" if LLM_PROVIDER == "ollama" else "memory.db"

    @staticmethod
    def validate():
        if Config.LLM_PROVIDER == "gemini":
            if not Config.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is not set in .env file (required for LLM_PROVIDER=gemini)")
        elif Config.LLM_PROVIDER == "ollama":
            # Ollama usually doesn't need a key, but we could check if URL is reachable if we wanted (skipping for now)
            pass
        else:
            raise ValueError(f"Invalid LLM_PROVIDER: {Config.LLM_PROVIDER}. Must be 'gemini' or 'ollama'")
            
    @staticmethod
    def validate_wakeword():
        if not Config.PICOVOICE_ACCESS_KEY:
            raise ValueError("PICOVOICE_ACCESS_KEY is not set in .env file")
