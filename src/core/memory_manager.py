import numpy as np
import logging
from typing import List, Tuple
from src.services.llm_service import LLMService
from src.services.embedding_service import EmbeddingService
from src.services.storage_service import StorageService
from src.models.memory_entry import MemoryEntry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self):
        self.llm_service = LLMService()
        self.embedding_service = EmbeddingService()
        self.storage_service = StorageService()

    def p_text(self, text: str) -> None:
        """Helper to print steps if needed, for now using logging"""
        logger.info(text)

    def process_input(self, text: str) -> dict:
        """
        Processes user input:
        1. Asks LLM if it's worth remembering.
        2. If yes, check for conflicts/updates.
        3. Store/Update/Ignore based on decision.
        """
        decision, reason = self.llm_service.decide_memory_importance(text)

        if decision:
            # Check for existing similar memories
            similar_memories = [m[0] for m in self.search_memory(text, limit=5)]
            
            if similar_memories:
                action, target_ids = self.llm_service.check_memory_update(text, similar_memories)
                
                if action == "IGNORE":
                    return {"stored": False, "reason": "Information already exists.", "message": "Memory ignored (duplicate)."}
                
                if action == "UPDATE":
                    for mid in target_ids:
                        self.storage_service.delete_memory(mid)
                    reason = f"Updated memory (replaced IDs: {target_ids}). {reason}"
            
            # If ADD or UPDATE (new one needs to be stored)
            embedding = self.embedding_service.get_embedding(text)
            memory_entry = MemoryEntry(text=text, embedding=embedding)
            self.storage_service.add_memory(memory_entry)
            return {"stored": True, "reason": reason, "message": "Memory stored successfully."}
        else:
            return {"stored": False, "reason": reason, "message": "Memory not stored."}

    def search_memory(self, query: str, limit: int = 5) -> List[Tuple[MemoryEntry, float]]:
        """
        Searches for memories relevant to the query.
        Returns a list of (MemoryEntry, similarity_score).
        """
        query_embedding = self.embedding_service.get_embedding(query)
        all_memories = self.storage_service.get_all_memories()
        
        if not all_memories:
            return []

        scored_memories = []
        for memory in all_memories:
            score = self._cosine_similarity(query_embedding, memory.embedding)
            scored_memories.append((memory, score))

        # Sort by score descending
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        return scored_memories[:limit]

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        a = np.array(vec_a)
        b = np.array(vec_b)
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
