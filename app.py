from flask import Flask, request, jsonify
from src.core.memory_manager import MemoryManager
from src.core.chat_service import ChatService
from src.models.memory_entry import MemoryEntry

app = Flask(__name__)

# Initialize services
# Note: In production, consider using a factory pattern or dependency injection
memory_manager = MemoryManager()
chat_service = ChatService()

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
    Payload: {"message": "What do I like?"}
    """
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "Missing 'message' in payload"}), 400
    
    user_message = data['message']
    response, store_result = chat_service.chat(user_message)
    
    return jsonify({
        "response": response,
        "memory_action": store_result
    })

if __name__ == '__main__':
    # Run on port 5000 by default
    print("Starting Quantum Memory Layer API...")
    app.run(debug=True, port=5000)
