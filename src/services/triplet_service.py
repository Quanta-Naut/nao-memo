import random
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)


class TripletService:
    """
    Generates training triplets (query, positive, negative) from stored memories
    using the LLM. This is the self-supervised data pipeline for contrastive learning.
    """

    def __init__(self, llm_service, embedding_service):
        self.llm_service = llm_service
        self.embedding_service = embedding_service

    def generate_triplets(self, memories, num_queries_per_memory: int = 2) -> List[dict]:
        """
        For each memory, generates natural queries using the LLM that should
        retrieve that memory. Then pairs each query with a positive (the source memory)
        and a negative (a semantically distant memory).

        Args:
            memories: List of MemoryEntry objects with embeddings
            num_queries_per_memory: How many queries to generate per memory

        Returns:
            List of {"query": str, "positive": str, "negative": str}
        """
        if len(memories) < 3:
            logger.warning("Need at least 3 memories to generate triplets.")
            return []

        triplets = []
        # Pre-compute pairwise distances for efficient negative selection
        embeddings = [m.embedding for m in memories]

        for i, memory in enumerate(memories):
            # 1. Generate queries using LLM
            queries = self._generate_queries_for_memory(memory.text, num_queries_per_memory)

            if not queries:
                continue

            # 2. Find hard negatives (memories that are somewhat related but NOT the answer)
            negatives = self._select_negatives(i, embeddings, memories, k=len(queries))

            # 3. Form triplets
            for j, query in enumerate(queries):
                negative = negatives[j % len(negatives)]
                triplets.append({
                    "query": query,
                    "positive": memory.text,
                    "negative": negative.text
                })

            logger.info(f"Generated {len(queries)} triplets for memory #{memory.id}: '{memory.text[:50]}...'")

        logger.info(f"Total triplets generated: {len(triplets)}")
        return triplets

    def _generate_queries_for_memory(self, memory_text: str, count: int) -> List[str]:
        """
        Uses the LLM to generate natural user queries that this memory should answer.
        """
        prompt = f"""
        You are helping build training data for a memory retrieval system.
        
        Given this stored memory about a user:
        "{memory_text}"
        
        Generate exactly {count} natural questions or queries that a user might ask
        where this memory would be the correct answer to retrieve.
        
        Rules:
        1. Questions should be varied (different phrasings, angles).
        2. Questions should be natural (how a real person would ask).
        3. Do NOT include the exact text of the memory in the question.
        4. One question per line, no numbering, no bullets, no extra text.
        
        Return ONLY the questions, one per line:
        """

        try:
            response = self.llm_service.model.generate_content(prompt)
            result = response.text.strip()
            queries = [q.strip() for q in result.split('\n') if q.strip()]
            # Take only the requested count
            return queries[:count]
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            return []

    def _select_negatives(self, anchor_idx: int, embeddings: list, memories: list, k: int) -> list:
        """
        Selects 'hard negatives' — memories that are somewhat similar but not the
        correct answer. These are more informative for training than random negatives.

        Strategy: Pick memories in the middle similarity range (not too similar, not too different).
        """
        anchor_emb = np.array(embeddings[anchor_idx])
        
        similarities = []
        for j, emb in enumerate(embeddings):
            if j == anchor_idx:
                continue
            b = np.array(emb)
            norm_a = np.linalg.norm(anchor_emb)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                sim = 0.0
            else:
                sim = float(np.dot(anchor_emb, b) / (norm_a * norm_b))
            similarities.append((j, sim))

        # Sort by similarity and pick from the middle range (hard negatives)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Skip the top 20% (too similar — might be valid positives)
        # Pick from the 20-60% range (hard negatives)
        n = len(similarities)
        start = max(1, int(n * 0.2))
        end = max(start + 1, int(n * 0.6))
        candidates = similarities[start:end]

        if not candidates:
            candidates = similarities  # fallback

        selected = random.sample(candidates, min(k, len(candidates)))
        return [memories[idx] for idx, _ in selected]
