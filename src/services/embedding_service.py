import google.generativeai as genai
from typing import List
from src.core.config import Config

class EmbeddingService:
    def __init__(self):
        Config.validate()
        genai.configure(api_key=Config.GEMINI_API_KEY)

    def get_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding vector for the given text using Gemini's embedding model.
        """
        try:
            result = genai.embed_content(
                model=Config.EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            # Fallback or re-raise depending on strictness. For now, print and return empty list or raise.
            print(f"Embedding Error: {e}")
            raise e
