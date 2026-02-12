from flask import Flask, request, jsonify
from src.core.memory_manager import MemoryManager
from src.core.chat_service import ChatService
from src.models.memory_entry import MemoryEntry

app = Flask(__name__)

# Initialize services
# Note: In production, consider using a factory pattern or dependency injection
memory_manager = MemoryManager()
chat_service = ChatService(memory_manager=memory_manager)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "Quantum Memory Layer"})

@app.route('/memory', methods=['POST'])
def add_memory():
    """
    Endpoint to manually add a memory (or let LLM decide).
    Payload: {"text": "I like apples"}
    """
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in payload"}), 400
    
    text = data['text']
    result = memory_manager.process_input(text)
    
    return jsonify(result)

@app.route('/search', methods=['GET'])
def search_memory():
    """
    Endpoint to search memories.
    Query param: ?q=food
    """
    query = request.args.get('q')
    if not query:
        return jsonify({"error": "Missing 'q' query parameter"}), 400
    
    limit = int(request.args.get('limit', 5))
    results = memory_manager.search_memory(query, limit=limit)
    
    # Format results
    response_data = []
    for entry, score in results:
        response_data.append({
            "text": entry.text,
            "score": score,
            "created_at": entry.created_at.isoformat(),
            "id": entry.id
        })
        
    return jsonify(response_data)

@app.route('/chat', methods=['POST'])
def chat():
    """
    Endpoint to chat with the LLM.
    Payload: {"text": "What do I like?"}
    """
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in payload"}), 400
    
    user_message = data['text']
    response, store_result = chat_service.chat(user_message)
    
    return jsonify({
        "response": response,
        "memory_action": store_result
    })

@app.route('/train', methods=['POST'])
def train_retriever():
    """
    Endpoint to trigger contrastive learning training.
    Optional payload: {"epochs": 3, "queries_per_memory": 2}
    """
    trainer = memory_manager.contrastive_trainer
    if trainer is None or trainer.model is None:
        return jsonify({"error": "sentence-transformers not installed. Run: pip install sentence-transformers"}), 500

    data = request.json or {}
    epochs = data.get("epochs", 3)
    queries_per_memory = data.get("queries_per_memory", 2)

    # Check memory count
    mem_count = memory_manager.storage_service.get_memory_count()
    if mem_count < 5:
        return jsonify({"error": f"Need at least 5 memories to train. Currently have {mem_count}."}), 400

    # Step 1: Generate triplets
    from src.services.triplet_service import TripletService
    triplet_service = TripletService(memory_manager.llm_service, memory_manager.embedding_service)
    memories = memory_manager.storage_service.get_all_memories()
    triplets = triplet_service.generate_triplets(memories, num_queries_per_memory=queries_per_memory)

    if len(triplets) < 3:
        return jsonify({"error": "Not enough triplets generated. Add more diverse memories."}), 400

    # Step 2: Train
    metrics = trainer.train(triplets, epochs=epochs, batch_size=16)
    if "error" in metrics:
        return jsonify({"error": metrics["error"]}), 500

    # Step 3: Re-embed all memories
    reembedded = trainer.reembed_all(memory_manager.storage_service)
    metrics["memories_reembedded"] = reembedded

    return jsonify(metrics)

@app.route('/train/status', methods=['GET'])
def train_status():
    """
    Endpoint to check the status of the contrastive retriever.
    """
    trainer = memory_manager.contrastive_trainer
    if trainer is None or trainer.model is None:
        return jsonify({"available": False, "reason": "sentence-transformers not installed"})

    import os, json
    status = {
        "available": True,
        "is_trained": trainer.is_trained(),
        "memory_count": memory_manager.storage_service.get_memory_count(),
    }

    # Load training log if it exists
    log_path = os.path.join(trainer.model_dir, "training_log.json")
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log = json.load(f)
        if log.get("runs"):
            latest = log["runs"][-1]
            status["latest_training"] = latest
            status["total_versions"] = len(log["runs"])

    return jsonify(status)

if __name__ == '__main__':
    # Run on port 5000 by default
    print("Starting Quantum Memory Layer API...")
    app.run(debug=True, port=5000)
