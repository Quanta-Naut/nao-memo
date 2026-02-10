import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
    DB_PATH = os.getenv("DB_PATH", "memory.db")
    EMBEDDING_MODEL = "models/gemini-embedding-001"
    GENERATIVE_MODEL = "gemini-2.5-flash-lite"

    @staticmethod
    def validate():
        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in .env file")
            
    @staticmethod
    def validate_wakeword():
        if not Config.PICOVOICE_ACCESS_KEY:
            raise ValueError("PICOVOICE_ACCESS_KEY is not set in .env file")
