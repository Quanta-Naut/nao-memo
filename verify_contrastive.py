"""
Verification script for the Contrastive Learning Memory Retrieval system.
Tests: triplet generation, model training, learned embeddings, retrieval comparison, and cold-start fallback.
"""
import sys
import time
import os

# Ensure we use a separate test database
os.environ["DB_PATH"] = "test_contrastive.db"

from src.core.memory_manager import MemoryManager
from src.services.triplet_service import TripletService


def cleanup():
    """Remove test database if it exists."""
    if os.path.exists("test_contrastive.db"):
        os.remove("test_contrastive.db")


def test_cold_start_fallback():
    """Test 1: Without a trained model, system uses Gemini embeddings (no crash)."""
    print("\n[Test 1] Cold Start Fallback")
    manager = MemoryManager()

    # Add a memory — should work without trained model
    result = manager.process_input("I study Computer Science at IISC Bangalore.")
    print(f"  Store result: {result}")
    
    if result['stored']:
        print("  PASSED: Memory stored using Gemini embeddings (cold start).")
    else:
        print(f"  INFO: Not stored (Reason: {result['reason']}). May already exist.")

    # Search — should work without trained model
    results = manager.search_memory("What do I study?")
    if results:
        print(f"  PASSED: Search works in cold start. Top result: '{results[0][0].text}' (score: {results[0][1]:.4f})")
    else:
        print("  WARNING: No search results in cold start.")

    return manager


def seed_memories(manager):
    """Seed the database with diverse memories for training."""
    print("\n[Seeding] Adding test memories...")
    test_memories = [
        "I study Computer Science at IISC Bangalore.",
        "My favorite programming language is Python.",
        "I enjoy playing badminton on weekends.",
        "My dog's name is Bruno and he's a golden retriever.",
        "I'm working on a research paper about AI memory systems.",
        "I like to eat south Indian food, especially dosa.",
        "My birthday is on March 15th.",
        "I use VS Code as my primary code editor.",
        "I'm learning German in my free time.",
        "My favorite movie is Interstellar.",
    ]

    for mem in test_memories:
        result = manager.process_input(mem)
        status = "stored" if result['stored'] else "skipped"
        print(f"  [{status}] {mem[:50]}...")

    count = manager.storage_service.get_memory_count()
    print(f"\n  Total memories in DB: {count}")
    return count


def test_triplet_generation(manager):
    """Test 2: LLM generates valid triplets from stored memories."""
    print("\n[Test 2] Triplet Generation")
    triplet_service = TripletService(manager.llm_service, manager.embedding_service)
    memories = manager.storage_service.get_all_memories()

    print(f"  Generating triplets from {len(memories)} memories...")
    triplets = triplet_service.generate_triplets(memories, num_queries_per_memory=2)

    print(f"  Generated {len(triplets)} triplets.")

    if len(triplets) >= 3:
        print("  PASSED: Sufficient triplets generated.")
        print(f"  Sample triplet:")
        t = triplets[0]
        print(f"    Query:    {t['query']}")
        print(f"    Positive: {t['positive'][:60]}...")
        print(f"    Negative: {t['negative'][:60]}...")
    else:
        print("  FAILED: Not enough triplets generated.")

    return triplets


def test_model_training(manager, triplets):
    """Test 3: Model fine-tunes without errors, saves checkpoint."""
    print("\n[Test 3] Model Training")
    trainer = manager.contrastive_trainer

    if trainer is None or trainer.model is None:
        print("  SKIPPED: sentence-transformers not installed.")
        return False

    print(f"  Training on {len(triplets)} triplets (1 epoch for speed)...")
    metrics = trainer.train(triplets, epochs=1, batch_size=8)

    if "error" in metrics:
        print(f"  FAILED: {metrics['error']}")
        return False

    print(f"  PASSED: Training complete.")
    print(f"    Duration: {metrics['duration_seconds']}s")
    print(f"    Model version: v{metrics['model_version']}")
    print(f"    Saved to: {metrics['model_path']}")

    return True


def test_reembedding(manager):
    """Test 4: All memories get learned embeddings after training."""
    print("\n[Test 4] Re-embedding Memories")
    trainer = manager.contrastive_trainer

    count = trainer.reembed_all(manager.storage_service)
    print(f"  Re-embedded {count} memories.")

    # Verify a memory has learned_embedding
    memories = manager.storage_service.get_all_memories()
    has_learned = sum(1 for m in memories if m.learned_embedding is not None)
    print(f"  Memories with learned embeddings: {has_learned}/{len(memories)}")

    if has_learned == len(memories):
        print("  PASSED: All memories have learned embeddings.")
    else:
        print("  WARNING: Some memories missing learned embeddings.")


def test_retrieval_comparison(manager):
    """Test 5: Compare Gemini vs fine-tuned retrieval on test queries."""
    print("\n[Test 5] Retrieval Comparison (Gemini vs Fine-tuned)")
    
    test_queries = [
        "What do I study?",
        "What's my pet's name?",
        "What sport do I play?",
        "What food do I like?",
        "What's my research about?",
    ]

    trainer = manager.contrastive_trainer
    all_memories = manager.storage_service.get_all_memories()

    print(f"\n  {'Query':<30} {'Gemini Top-1':<40} {'Learned Top-1':<40}")
    print(f"  {'-'*30} {'-'*40} {'-'*40}")

    for query in test_queries:
        # Gemini retrieval
        query_emb_gemini = manager.embedding_service.get_embedding(query)
        gemini_scores = []
        for m in all_memories:
            s = manager._cosine_similarity(query_emb_gemini, m.embedding)
            gemini_scores.append((m, s))
        gemini_scores.sort(key=lambda x: x[1], reverse=True)
        gemini_top = gemini_scores[0] if gemini_scores else (None, 0)

        # Learned retrieval
        query_emb_learned = trainer.encode([query])[0]
        learned_scores = []
        for m in all_memories:
            if m.learned_embedding:
                s = manager._cosine_similarity(query_emb_learned, m.learned_embedding)
                learned_scores.append((m, s))
        learned_scores.sort(key=lambda x: x[1], reverse=True)
        learned_top = learned_scores[0] if learned_scores else (None, 0)

        g_text = f"{gemini_top[0].text[:35]}.. ({gemini_top[1]:.3f})" if gemini_top[0] else "N/A"
        l_text = f"{learned_top[0].text[:35]}.. ({learned_top[1]:.3f})" if learned_top[0] else "N/A"
        print(f"  {query:<30} {g_text:<40} {l_text:<40}")

    print("\n  PASSED: Comparison complete. Check results above.")


if __name__ == "__main__":
    print("=" * 80)
    print("  CONTRASTIVE LEARNING MEMORY RETRIEVAL - VERIFICATION")
    print("=" * 80)

    cleanup()

    try:
        # Test 1: Cold start
        manager = test_cold_start_fallback()

        # Seed memories
        count = seed_memories(manager)
        if count < 5:
            print("\nFATAL: Not enough memories stored. Check LLM connection.")
            sys.exit(1)

        # Test 2: Triplet generation
        triplets = test_triplet_generation(manager)
        if len(triplets) < 3:
            print("\nFATAL: Not enough triplets. Cannot proceed with training.")
            sys.exit(1)

        # Test 3: Model training
        trained = test_model_training(manager, triplets)
        if not trained:
            print("\nSkipping remaining tests (model not trained).")
            sys.exit(0)

        # Test 4: Re-embedding
        test_reembedding(manager)

        # Test 5: Retrieval comparison
        test_retrieval_comparison(manager)

        print("\n" + "=" * 80)
        print("  ALL TESTS COMPLETE")
        print("=" * 80)

    finally:
        cleanup()
        # Clean up model directory if it was created during test
        import shutil
        test_model_dir = os.path.join("models", "retriever")
        if os.path.exists(test_model_dir):
            print(f"\n[Cleanup] Test model saved at: {test_model_dir}")
