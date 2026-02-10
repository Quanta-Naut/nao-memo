# Quantum Memory Layer

A persistent, semantic memory system for AI agents, powered by Google Gemini and SQLite.

## Features
- **Semantic Search**: Retrieve relevant memories using vector embeddings.
- **Conflict Resolution**: Automatically updates contradictory information (e.g., changing favorite food).
- **Implicit Memory**: Learns from conversations without explicit "save" commands.
- **API Support**: Includes a Flask REST API for easy integration.

## Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables:**
   Rename `.env.example` to `.env` and add your Gemini API Key:
   ```bash
   GEMINI_API_KEY=your_api_key_here
   DB_PATH=memory.db
   ```

## Usage

### 1. REST API (Flask App)
Start the API server:
```bash
python app.py
```
The server runs on `http://127.0.0.1:5000`.

#### **API Endpoints**

*   **Chat with Memory** (`POST /chat`)
    *   **Payload**: `{"message": "What is my favorite food?"}`
    *   **Description**: Get an LLM response based on your stored memories.
    *   **Example**:
        ```bash
        curl -X POST http://127.0.0.1:5000/chat -H "Content-Type: application/json" -d "{\"message\": \"What is my favorite food?\"}"
        ```

*   **Search Memories** (`GET /search`)
    *   **Parameters**: `q` (query), `limit` (optional, default 5)
    *   **Description**: Find memories semantically similar to your query.
    *   **Example**:
        ```bash
        curl "http://127.0.0.1:5000/search?q=food"
        ```

*   **Add Memory** (`POST /memory`)
    *   **Payload**: `{"text": "I like coding in Python"}`
    *   **Description**: Manually store a memory (subject to LLM validation logic).
    *   **Example**:
        ```bash
        curl -X POST http://127.0.0.1:5000/memory -H "Content-Type: application/json" -d "{\"text\": \"I like coding\"}"
        ```

---

### 2. Interactive CLI Demo
Run the rich terminal interface:
```bash
python main.py
```
- **Add Memory**: Type new facts.
- **Search**: Find what the AI knows.
- **Chat**: Talk to the AI freely.

### 3. Deep Dive Demo
See the internal vector operations and LLM reasoning:
```bash
python demo.py
```
