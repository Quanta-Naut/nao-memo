import os
import json
import logging
from typing import List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Default paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "retriever")
TRAINING_LOG = os.path.join(MODEL_DIR, "training_log.json")


class ContrastiveTrainer:
    """
    Fine-tunes a sentence-transformers model using contrastive (triplet) loss
    for personalized memory retrieval. Supports warm-starting from previous checkpoints.
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads the fine-tuned model if it exists, otherwise loads the base model."""
        try:
            from sentence_transformers import SentenceTransformer

            if self.is_trained():
                logger.info(f"Loading fine-tuned model from {self.model_dir}")
                self.model = SentenceTransformer(self.model_dir)
            else:
                logger.info("Loading base model: all-MiniLM-L6-v2")
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            self.model = None

    def is_trained(self) -> bool:
        """Checks if a fine-tuned checkpoint exists."""
        return os.path.exists(os.path.join(self.model_dir, "config.json"))

    def train(self, triplets: List[dict], epochs: int = 3, batch_size: int = 16, memory_count: int = 0) -> dict:
        """
        Fine-tunes the model using TripletLoss on the provided triplets.
        Warm-starts from previous checkpoint if available.

        Args:
            triplets: List of {"query": str, "positive": str, "negative": str}
            epochs: Number of training epochs
            batch_size: Training batch size
            memory_count: Total number of memories at the time of training

        Returns:
            dict with training metrics: {"duration_seconds", "num_triplets", "model_version", ...}
        """
        if self.model is None:
            return {"error": "sentence-transformers not available"}

        if len(triplets) < 3:
            return {"error": "Need at least 3 triplets to train"}

        from sentence_transformers import InputExample, losses
        from torch.utils.data import DataLoader

        start_time = datetime.now()

        # Convert triplets to InputExamples
        train_examples = []
        for t in triplets:
            train_examples.append(
                InputExample(texts=[t["query"], t["positive"], t["negative"]])
            )

        # Create data loader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

        # Define loss
        train_loss = losses.TripletLoss(model=self.model)

        # Train
        logger.info(f"Starting training: {len(triplets)} triplets, {epochs} epochs")

        warmup_steps = int(len(train_dataloader) * epochs * 0.1)

        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=self.model_dir,
            show_progress_bar=True
        )

        duration = (datetime.now() - start_time).total_seconds()

        # Load training log and update
        learning_rate = 2e-5 # Default for SentenceTransformer
        version = self._update_training_log(
            num_triplets=len(triplets),
            epochs=epochs,
            batch_size=batch_size,
            memory_count=memory_count,
            duration=duration,
            learning_rate=learning_rate
        )

        metrics = {
            "duration_seconds": round(duration, 1),
            "num_triplets": len(triplets),
            "epochs": epochs,
            "batch_size": batch_size,
            "memory_count": memory_count,
            "learning_rate": learning_rate,
            "model_version": version,
            "model_path": self.model_dir
        }

        logger.info(f"Training complete: {metrics}")
        return metrics

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Encodes texts using the fine-tuned model (or base model if not yet trained).

        Returns:
            List of embedding vectors (384-dim for MiniLM)
        """
        if self.model is None:
            logger.warning("Model not available, returning empty embeddings")
            return [[] for _ in texts]

        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

    def reembed_all(self, storage_service) -> int:
        """
        Re-encodes all stored memories with the latest fine-tuned model
        and updates them in the database.

        Returns:
            Number of memories re-embedded
        """
        if self.model is None:
            return 0

        memories = storage_service.get_all_memories()
        if not memories:
            return 0

        # Batch encode all memory texts
        texts = [m.text for m in memories]
        learned_embeddings = self.encode(texts)

        # Update each memory in the database
        count = 0
        for memory, learned_emb in zip(memories, learned_embeddings):
            if memory.id is not None and learned_emb:
                storage_service.update_learned_embedding(memory.id, learned_emb)
                count += 1

        logger.info(f"Re-embedded {count} memories with the fine-tuned model")
        return count

    def _update_training_log(self, num_triplets: int, epochs: int, batch_size: int, memory_count: int, duration: float, learning_rate: float) -> int:
        """Updates the training log with detailed metrics from this run. Returns the new version number."""
        os.makedirs(self.model_dir, exist_ok=True)

        log = {"runs": []}
        if os.path.exists(TRAINING_LOG):
            with open(TRAINING_LOG, 'r') as f:
                log = json.load(f)

        version = len(log["runs"]) + 1
        log["runs"].append({
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "memory_count": memory_count,
            "num_triplets": num_triplets,
            "training_params": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate
            },
            "performance": {
                "duration_seconds": round(duration, 1)
                # "final_loss": "Notcaptured" # Requires callback
            }
        })

        with open(TRAINING_LOG, 'w') as f:
            json.dump(log, f, indent=2)

        return version
