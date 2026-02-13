import numpy as np
import logging
import threading
from typing import List, Tuple
from src.services.llm_service import LLMService
from src.services.embedding_service import EmbeddingService
from src.services.storage_service import StorageService
from src.models.memory_entry import MemoryEntry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Auto-retrain after this many new memories since last training
RETRAIN_THRESHOLD = 20


class MemoryManager:
    def __init__(self):
        self.llm_service = LLMService()
        self.embedding_service = EmbeddingService()
        self.storage_service = StorageService()

        # Lazy-load the contrastive trainer (only when needed)
        self._contrastive_trainer = None

        # Auto-training state
        self._is_training = False
        self._last_trained_count = self._get_last_trained_count()

    @property
    def contrastive_trainer(self):
        """Lazy-loads the contrastive trainer to avoid importing torch on startup."""
        if self._contrastive_trainer is None:
            try:
                from src.ml.contrastive_trainer import ContrastiveTrainer
                self._contrastive_trainer = ContrastiveTrainer()
            except ImportError:
                logger.warning("sentence-transformers not installed. Using Gemini embeddings only.")
                self._contrastive_trainer = None
        return self._contrastive_trainer

    def _get_last_trained_count(self) -> int:
        """Reads the last trained memory count from the training log."""
        import os, json
        log_path = os.path.join("models", "retriever", "training_log.json")
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    log = json.load(f)
                if log.get("runs"):
                    return log["runs"][-1].get("memory_count", 0)
            except Exception:
                pass
        return 0

    def p_text(self, text: str) -> None:
        """Helper to print steps if needed, for now using logging"""
        logger.info(text)

    def process_input(self, text: str) -> dict:
        """
        Processes user input:
        1. Asks LLM if it's worth remembering.
        2. If yes, check for conflicts/updates.
        3. Store/Update/Ignore based on decision.
        4. Auto-triggers retraining if threshold reached.
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
            
            # Generate embeddings
            embedding = self.embedding_service.get_embedding(text)
            
            # Generate learned embedding if trained model exists
            learned_emb = None
            if self.contrastive_trainer and self.contrastive_trainer.is_trained():
                learned_emb = self.contrastive_trainer.encode([text])[0]

            memory_entry = MemoryEntry(text=text, embedding=embedding, learned_embedding=learned_emb)
            self.storage_service.add_memory(memory_entry)

            # Check if auto-retrain threshold is reached
            result = {"stored": True, "reason": reason, "message": "Memory stored successfully."}
            self._check_auto_retrain(result)

            return result
        else:
            return {"stored": False, "reason": reason, "message": "Memory not stored."}

    def add_memory_direct(self, text: str) -> dict:
        """
        Directly adds a memory without LLM validation.
        Used for bulk import or trusted sources.
        """
        try:
            # Generate embeddings
            embedding = self.embedding_service.get_embedding(text)
            
            # Generate learned embedding if trained model exists
            learned_emb = None
            if self.contrastive_trainer and self.contrastive_trainer.is_trained():
                learned_emb = self.contrastive_trainer.encode([text])[0]

            memory_entry = MemoryEntry(text=text, embedding=embedding, learned_embedding=learned_emb)
            self.storage_service.add_memory(memory_entry)

            # Check if auto-retrain threshold is reached
            result = {"stored": True, "reason": "Direct add via bulk import.", "message": "Memory stored."}
            self._check_auto_retrain(result)
            return result
        except Exception as e:
            logger.error(f"Error adding memory directly: {e}")
            return {"stored": False, "reason": str(e), "message": "Error adding memory."}

    def _check_auto_retrain(self, result: dict):
        """Triggers background retraining if enough new memories have accumulated."""
        if self._is_training:
            return

        mem_count = self.storage_service.get_memory_count()
        new_since_train = mem_count - self._last_trained_count

        if mem_count >= 5 and new_since_train >= RETRAIN_THRESHOLD:
            logger.info(f"Auto-retrain threshold reached ({new_since_train} new memories). Starting background training...")
            result["training_triggered"] = True
            result["message"] += " Background retraining started."

            thread = threading.Thread(target=self._auto_train, daemon=True)
            thread.start()

    def _auto_train(self):
        """Runs the full training pipeline in a background thread."""
        self._is_training = True
        try:
            trainer = self.contrastive_trainer
            if trainer is None or trainer.model is None:
                logger.warning("Auto-train skipped: sentence-transformers not available.")
                return

            # Step 1: Generate triplets
            from src.services.triplet_service import TripletService
            triplet_service = TripletService(self.llm_service, self.embedding_service)
            memories = self.storage_service.get_all_memories()
            triplets = triplet_service.generate_triplets(memories, num_queries_per_memory=2)

            if len(triplets) < 3:
                logger.warning("Auto-train skipped: not enough triplets generated.")
                return

            # Step 2: Train
            metrics = trainer.train(triplets, epochs=3, batch_size=16, memory_count=len(memories))
            if "error" in metrics:
                logger.error(f"Auto-train failed: {metrics['error']}")
                return

            # Save memory count at training time
            self._save_trained_count(len(memories))

            # Step 3: Re-embed all memories
            count = trainer.reembed_all(self.storage_service)
            logger.info(f"Auto-train complete: model v{metrics.get('model_version')}, re-embedded {count} memories.")

            self._last_trained_count = len(memories)

        except Exception as e:
            logger.error(f"Auto-train error: {e}")
        finally:
            self._is_training = False

    def _save_trained_count(self, count: int):
        """Saves the memory count at training time to the training log."""
        import os, json
        log_path = os.path.join("models", "retriever", "training_log.json")
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    log = json.load(f)
                if log.get("runs"):
                    log["runs"][-1]["memory_count"] = count
                    with open(log_path, 'w') as f:
                        json.dump(log, f, indent=2)
            except Exception:
                pass

    def search_memory(self, query: str, limit: int = 5) -> List[Tuple[MemoryEntry, float]]:
        """
        Searches for memories relevant to the query.
        Uses learned embeddings if a trained model exists, otherwise falls back to Gemini.
        """
        all_memories = self.storage_service.get_all_memories()
        
        if not all_memories:
            return []

        use_learned = (
            self.contrastive_trainer
            and self.contrastive_trainer.is_trained()
            and all_memories[0].learned_embedding is not None
        )

        if use_learned:
            # Use fine-tuned model for retrieval
            query_embedding = self.contrastive_trainer.encode([query])[0]
            scored_memories = []
            for memory in all_memories:
                if memory.learned_embedding:
                    score = self._cosine_similarity(query_embedding, memory.learned_embedding)
                else:
                    # Fallback for memories not yet re-embedded
                    score = 0.0
                scored_memories.append((memory, score))
        else:
            # Cold start: use Gemini embeddings (original behavior)
            query_embedding = self.embedding_service.get_embedding(query)
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
