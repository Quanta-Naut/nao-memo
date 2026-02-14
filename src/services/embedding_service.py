import google.generativeai as genai
from typing import List
from src.core.config import Config
import ollama

class EmbeddingService:
    def __init__(self):
        Config.validate()
        if Config.LLM_PROVIDER == "gemini":
            genai.configure(api_key=Config.GEMINI_API_KEY)

    def get_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding vector for the given text using the configured provider.
        """
        try:
            if Config.LLM_PROVIDER == "gemini":
                result = genai.embed_content(
                    model=Config.EMBEDDING_MODEL,
                    content=text,
                    task_type="retrieval_document"
                )
                return result['embedding']
            elif Config.LLM_PROVIDER == "ollama":
                response = ollama.embeddings(
                    model=Config.OLLAMA_EMBEDDING_MODEL,
                    prompt=text
                )
                return response['embedding']
            return []
        except Exception as e:
            # Fallback or re-raise depending on strictness. For now, print and return empty list or raise.
            print(f"Embedding Error: {e}")
            raise e
