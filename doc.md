# Quantum Memory Layer — Complete Technical Documentation

> An AI-powered long-term memory system with LLM-gated storage, semantic retrieval, voice interaction, and contrastive learning-based personalized retrieval.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Project Structure](#3-project-structure)
4. [Configuration & Environment](#4-configuration--environment)
5. [Core System — Memory Pipeline](#5-core-system--memory-pipeline)
   - 5.1 [Memory Entry Data Model](#51-memory-entry-data-model)
   - 5.2 [LLM-Gated Memory Storage](#52-llm-gated-memory-storage)
   - 5.3 [Conflict Detection & Memory Updates](#53-conflict-detection--memory-updates)
   - 5.4 [Embedding Generation](#54-embedding-generation)
   - 5.5 [Storage Layer (SQLite)](#55-storage-layer-sqlite)
   - 5.6 [Semantic Search & Retrieval](#56-semantic-search--retrieval)
6. [Chat System — RAG-Based Responses](#6-chat-system--rag-based-responses)
7. [Voice Interaction](#7-voice-interaction)
   - 7.1 [Wake Word Detection](#71-wake-word-detection)
   - 7.2 [Speech-to-Text](#72-speech-to-text)
   - 7.3 [Voice Chat Flow](#73-voice-chat-flow)
8. [Contrastive Learning — Personalized Retrieval](#8-contrastive-learning--personalized-retrieval)
   - 8.1 [Motivation](#81-motivation)
   - 8.2 [Triplet Generation Service](#82-triplet-generation-service)
   - 8.3 [Contrastive Training Pipeline](#83-contrastive-training-pipeline)
   - 8.4 [Dual-Embedding Retrieval](#84-dual-embedding-retrieval)
   - 8.5 [Auto-Training (Background Thread)](#85-auto-training-background-thread)
   - 8.6 [Continuous Learning Loop](#86-continuous-learning-loop)
9. [CLI Application](#9-cli-application)
10. [Flask REST API](#10-flask-rest-api)
11. [Installation & Setup](#11-installation--setup)
12. [Verification & Testing](#12-verification--testing)
13. [Dependencies](#13-dependencies)

---

## 1. Overview

The Quantum Memory Layer is a system that gives AI assistants **persistent, long-term memory**. Unlike typical chatbots that forget everything between sessions, this system:

- **Stores** facts, preferences, events, and plans about the user
- **Decides** what's worth remembering using an LLM (not everything is stored)
- **Retrieves** relevant memories when the user asks questions
- **Updates** memories when information changes (e.g., "I now like Java" replaces "I like Python")
- **Learns** to retrieve better over time using contrastive learning (ML-based personalization)
- **Listens** via wake word detection and speech-to-text for hands-free interaction

### Key Technologies

| Component | Technology |
|---|---|
| LLM | Google Gemini (`gemini-2.5-flash-lite`) |
| Embeddings | Gemini Embedding API (`models/gemini-embedding-001`, 768-dim) |
| Learned Embeddings | Fine-tuned `all-MiniLM-L6-v2` (384-dim, local) |
| Storage | SQLite (`memory.db`) |
| Wake Word | Picovoice Porcupine |
| Speech-to-Text | Google Speech Recognition |
| ML Training | PyTorch + sentence-transformers |
| API | Flask |
| CLI | Rich (Python) |

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    User Interaction                       │
│            (CLI / Flask API / Voice Chat)                 │
└──────────────┬───────────────────────┬───────────────────┘
               │                       │
        ┌──────▼──────┐         ┌──────▼──────┐
        │   Memory    │         │    Chat     │
        │   Manager   │         │   Service   │
        └──┬───┬───┬──┘         └──────┬──────┘
           │   │   │                   │
    ┌──────▼┐ ┌▼────▼─┐         ┌──────▼──────┐
    │  LLM  │ │Embed- │         │     LLM     │
    │Service│ │ding   │         │  (Generate  │
    │(Gate) │ │Service│         │  Response)  │
    └───────┘ └───┬───┘         └─────────────┘
                  │
           ┌──────▼──────┐
           │   Storage   │
           │  Service    │
           │  (SQLite)   │
           └──────┬──────┘
                  │
        ┌─────────▼─────────┐
        │  Contrastive      │
        │  Training Pipeline│
        │  (Background)     │
        └───────────────────┘
```

### Data Flow — Storing a Memory

```
User Input → LLM decides "worth storing?" → YES/NO
                                              │
                                  ┌───────────▼───────────┐
                                  │ Check for conflicts    │
                                  │ with existing memories │
                                  └───────┬───┬───┬───────┘
                                    ADD  UPDATE  IGNORE
                                          │
                                  ┌───────▼───────┐
                                  │ Generate       │
                                  │ Embeddings     │
                                  │ (Gemini +      │
                                  │  Learned)      │
                                  └───────┬───────┘
                                          │
                                  ┌───────▼───────┐
                                  │ Store in       │
                                  │ SQLite         │
                                  └───────┬───────┘
                                          │
                                  ┌───────▼───────┐
                                  │ Check auto-    │
                                  │ retrain        │
                                  │ threshold      │
                                  └───────────────┘
```

### Data Flow — Searching/Chatting

```
User Query → Generate query embedding
                    │
            ┌───────▼───────────┐
            │ Trained model     │
            │ exists?           │
            └──┬──────────┬─────┘
              YES          NO
               │            │
        ┌──────▼──────┐  ┌──▼──────────┐
        │ Use learned │  │ Use Gemini  │
        │ embeddings  │  │ embeddings  │
        │ (384-dim)   │  │ (768-dim)   │
        └──────┬──────┘  └──┬──────────┘
               └─────┬──────┘
                     │
              ┌──────▼──────┐
              │ Cosine       │
              │ Similarity   │
              │ → Top-K      │
              └──────┬───────┘
                     │
              ┌──────▼──────┐
              │ LLM generates│
              │ response with│
              │ context (RAG)│
              └─────────────┘
```

---

## 3. Project Structure

```
memory-layer/
├── .env                          # API keys (GEMINI_API_KEY, PICOVOICE_ACCESS_KEY)
├── .env.example                  # Template for .env
├── .gitignore
├── requirements.txt              # All Python dependencies
├── main.py                       # CLI application (Rich-based interactive menu)
├── app.py                        # Flask REST API server
├── demo.py                       # Interactive debugging demo
├── verify.py                     # Core system verification tests
├── verify_contrastive.py         # Contrastive learning benchmark tests
├── memory.db                     # SQLite database (auto-created)
├── ARCHITECTURE.md               # Architecture overview with Mermaid diagrams
├── README.md                     # Project README
├── doc.md                        # This file
│
├── models/                       # Auto-created after first training
│   └── retriever/                # Fine-tuned MiniLM model checkpoint
│       ├── config.json
│       ├── model.safetensors
│       └── training_log.json     # Version history of training runs
│
└── src/
    ├── core/
    │   ├── config.py             # Environment config & validation
    │   ├── memory_manager.py     # Central orchestrator for memory operations
    │   └── chat_service.py       # RAG-based chat with memory context
    │
    ├── models/
    │   └── memory_entry.py       # MemoryEntry dataclass (data model)
    │
    ├── services/
    │   ├── llm_service.py        # Gemini LLM interactions (decisions, chat, conflicts)
    │   ├── embedding_service.py  # Gemini embedding generation (768-dim)
    │   ├── storage_service.py    # SQLite database operations
    │   ├── stt_service.py        # Speech-to-Text (Google Speech Recognition)
    │   ├── wakeword_service.py   # Wake word detection (Picovoice Porcupine)
    │   └── triplet_service.py    # LLM-based training triplet generation
    │
    └── ml/
        └── contrastive_trainer.py  # MiniLM fine-tuning with TripletLoss
```

---

## 4. Configuration & Environment

**File:** `src/core/config.py`

All configuration is loaded from environment variables (`.env` file):

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | ✅ Yes | — | Google Gemini API key for LLM and embeddings |
| `PICOVOICE_ACCESS_KEY` | Optional | — | Picovoice API key for wake word detection |
| `DB_PATH` | Optional | `memory.db` | Path to SQLite database file |

**Model constants** (hardcoded in `config.py`):

| Constant | Value | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Gemini embedding model (768 dimensions) |
| `GENERATIVE_MODEL` | `gemini-2.5-flash-lite` | Gemini generative model for all LLM tasks |

### `.env` file format

```env
GEMINI_API_KEY=your_gemini_api_key_here
PICOVOICE_ACCESS_KEY=your_picovoice_key_here
DB_PATH=memory.db
```

---

## 5. Core System — Memory Pipeline

### 5.1 Memory Entry Data Model

**File:** `src/models/memory_entry.py`

Every memory is stored as a `MemoryEntry` dataclass:

```python
@dataclass
class MemoryEntry:
    text: str                                        # The actual memory text
    embedding: List[float]                           # Gemini embedding (768-dim)
    learned_embedding: Optional[List[float]] = None  # Fine-tuned MiniLM embedding (384-dim)
    metadata: dict = field(default_factory=dict)     # Extra metadata (unused, extensible)
    created_at: datetime = field(default_factory=datetime.now)
    id: Optional[int] = None                         # SQLite auto-increment ID
```

**Serialization:**
- `to_db_tuple()` — Converts to a tuple for SQLite insertion. Embeddings are JSON-serialized.
- `from_db_tuple(row)` — Backwards-compatible factory method. Handles both 5-column (legacy, before contrastive learning) and 6-column (current) database rows.

**Backwards Compatibility:** If the database was created before the contrastive learning update, `from_db_tuple` detects the 5-column format and sets `learned_embedding = None`.

---

### 5.2 LLM-Gated Memory Storage

**File:** `src/services/llm_service.py` → `decide_memory_importance()`

Not everything the user says is stored. The LLM acts as a **gate**, deciding what's worth remembering.

**Worth storing:**
- Facts about the user ("I study CS at IISC")
- Preferences ("My favorite food is dosa")
- Events ("I have a meeting tomorrow at 10")
- Relationships ("My dog's name is Bruno")
- Setup details, crucial context

**Not worth storing:**
- Casual greetings ("Hi", "How are you?")
- Fleeting thoughts
- Ephemeral questions ("What is 2+2?")
- Nonsensical input

**How it works:**

```
Input: "I like Python programming"
    ↓
LLM Prompt:
    "You are a memory manager AI. Decide if this is worth storing..."
    ↓
LLM Response:
    "DECISION: YES"
    "REASON: This is a programming language preference worth remembering."
    ↓
Returns: (True, "This is a programming language preference worth remembering.")
```

The LLM is prompted to return a structured response with `DECISION: [YES/NO]` and `REASON: [explanation]`, which is then parsed.

---

### 5.3 Conflict Detection & Memory Updates

**File:** `src/services/llm_service.py` → `check_memory_update()`

When new information is about to be stored, the system checks if it **conflicts** with existing memories.

**Three possible actions:**

| Action | When | Example |
|---|---|---|
| `ADD` | New info, no conflict | "I like badminton" (no existing sport preference) |
| `UPDATE` | New info supersedes old | "I like Java" overrides "I like Python" |
| `IGNORE` | Duplicate information | "I like Python" when already stored |

**How UPDATE works:**
1. The LLM receives the new text and up to 5 existing similar memories
2. It identifies which memories conflict and returns their IDs
3. Those old memories are **deleted** from the database
4. The new memory is then stored

**Example flow:**
```
Existing Memory ID 3: "I like Python"
New Input: "I now only like Java"
    ↓
LLM decides: ACTION: UPDATE, TARGET_IDS: [3]
    ↓
Memory ID 3 deleted
New memory "I now only like Java" stored
```

---

### 5.4 Embedding Generation

**File:** `src/services/embedding_service.py`

Converts text into numerical vectors (embeddings) for similarity comparison.

**Gemini Embeddings (Primary):**
- Model: `models/gemini-embedding-001`
- Dimension: **768**
- Task type: `retrieval_document`
- Requires API call to Google (has latency + cost)

**Learned Embeddings (After Training):**
- Model: Fine-tuned `all-MiniLM-L6-v2`
- Dimension: **384**
- Runs locally (no API call, no cost, faster)
- Only available after contrastive learning training

Both embeddings are stored per memory for dual-retrieval capability.

---

### 5.5 Storage Layer (SQLite)

**File:** `src/services/storage_service.py`

All memories are stored in a single SQLite file (`memory.db` by default).

**Schema:**

```sql
CREATE TABLE IF NOT EXISTS memories (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    text              TEXT NOT NULL,
    embedding         TEXT NOT NULL,       -- JSON-serialized Gemini embedding (768-dim)
    learned_embedding TEXT,                -- JSON-serialized learned embedding (384-dim), nullable
    metadata          TEXT,                -- JSON-serialized metadata dict
    created_at        TEXT NOT NULL        -- ISO 8601 timestamp
);
```

**Schema Migration:** If the database was created before the contrastive learning update (without `learned_embedding` column), the `_migrate_schema()` method automatically adds it via `ALTER TABLE` on startup. No data loss.

**Available Methods:**

| Method | Description |
|---|---|
| `add_memory(entry)` | Inserts a new MemoryEntry |
| `get_all_memories()` | Returns all memories as MemoryEntry objects |
| `get_memory_count()` | Returns total count of stored memories |
| `update_learned_embedding(id, emb)` | Updates the learned embedding for a specific memory |
| `delete_memory(id)` | Deletes a memory by its ID |

---

### 5.6 Semantic Search & Retrieval

**File:** `src/core/memory_manager.py` → `search_memory()`

Retrieval uses **cosine similarity** between the query embedding and all stored memory embeddings.

**Scoring Formula:**
```
score = cosine_similarity(query_embedding, memory_embedding)
```

**Where:**
```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

**Dual-Embedding Retrieval Logic:**

```python
if trained_model_exists and memories_have_learned_embeddings:
    # Use fine-tuned MiniLM (384-dim, local, personalized)
    query_emb = contrastive_trainer.encode(query)
    for memory in all_memories:
        score = cosine_similarity(query_emb, memory.learned_embedding)
else:
    # Cold start fallback: use Gemini (768-dim, API call, generic)
    query_emb = embedding_service.get_embedding(query)
    for memory in all_memories:
        score = cosine_similarity(query_emb, memory.embedding)
```

Results are sorted by score (descending) and the top-K are returned.

---

## 6. Chat System — RAG-Based Responses

**File:** `src/core/chat_service.py`

The chat system implements **Retrieval-Augmented Generation (RAG)**:

1. **Retrieve**: Search for top 3 memories relevant to the user's query
2. **Filter**: Only include memories with similarity score > 0.6 (relevance threshold)
3. **Generate**: Pass the query + filtered memories as context to the LLM
4. **Store**: Simultaneously check if the user's chat message itself should be stored as a new memory

**LLM Chat Prompt Behavior:**
- Memories are presented as facts **about the user** (not the AI)
- The LLM is instructed NOT to randomly recite facts (e.g., don't say "Hi, you like sushi" when user just says "Hi")
- If memories contradict each other, the most recent one is prioritized
- The LLM does not explicitly say "I found this in your memory" unless contextually relevant

**Example:**
```
User: "What food should I order tonight?"
    ↓
Search finds: "I like south Indian food, especially dosa" (score: 0.82)
    ↓
LLM generates: "Based on your love for south Indian food, you might enjoy
                ordering some dosas tonight! You could also try..."
```

---

## 7. Voice Interaction

### 7.1 Wake Word Detection

**File:** `src/services/wakeword_service.py`

Uses **Picovoice Porcupine** for on-device wake word detection. The system listens continuously in a low-power mode until it hears one of the configured wake words.

**Default Wake Words:** `"jarvis"`, `"computer"`

**Available Built-in Wake Words:**
`alexa`, `americano`, `blueberry`, `bumblebee`, `computer`, `grapefruit`, `grasshopper`, `hey google`, `hey siri`, `jarvis`, `ok google`, `picovoice`, `porcupine`, `terminator`

**How it works:**

```python
class WakeWordService:
    def __init__(self, keywords=["jarvis", "computer"]):
        self.porcupine = pvporcupine.create(
            access_key=Config.PICOVOICE_ACCESS_KEY,
            keywords=keywords
        )
        self.recorder = pvrecorder.PvRecorder(
            device_index=-1,  # default microphone
            frame_length=self.porcupine.frame_length
        )

    def listen_for_wake_word(self):
        """Blocks until wake word is detected. Returns True/False."""
        self.recorder.start()
        while True:
            pcm = self.recorder.read()
            keyword_index = self.porcupine.process(pcm)
            if keyword_index >= 0:
                return True  # Wake word detected!
```

**Requirements:**
- `PICOVOICE_ACCESS_KEY` in `.env`
- `pvporcupine` and `pvrecorder` packages
- A working microphone

**Graceful Degradation:** If the Picovoice key is missing or initialization fails, the wake word service is disabled but the rest of the app still works. Voice chat falls back to direct listening mode.

---

### 7.2 Speech-to-Text

**File:** `src/services/stt_service.py`

Converts spoken audio to text using **Google Speech Recognition** (via the `SpeechRecognition` library).

**How it works:**
1. Adjusts for ambient noise (0.5 second calibration)
2. Listens for speech (timeout: 5 seconds, max phrase: 10 seconds)
3. Sends audio to Google STT API for transcription
4. Returns the transcribed text

```python
class STTService:
    def listen_and_transcribe(self) -> str:
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = self.recognizer.recognize_google(audio)
            return text
```

**Error Handling:**
| Error | Behavior |
|---|---|
| `WaitTimeoutError` | "No speech detected. Timed out." → returns `None` |
| `UnknownValueError` | "Could not understand audio." → returns `None` |
| `RequestError` | Google STT API unreachable → returns `None` |
| `Exception` | Generic microphone error → returns `None` |

**Requirements:**
- `SpeechRecognition` and `pyaudio` packages
- A working microphone
- Internet connection (for Google STT API)

---

### 7.3 Voice Chat Flow

**File:** `main.py` → `voice_chat_flow()`

Combines wake word detection + STT + chat into a continuous voice loop:

```
┌──────────────────────────────────────────┐
│           Voice Chat Loop                │
│                                          │
│  1. Wait for wake word ("Jarvis"...)     │
│              ↓                           │
│  2. Listen for speech → STT → text       │
│              ↓                           │
│  3. If "exit"/"quit" → break             │
│              ↓                           │
│  4. Search memories + generate response  │
│              ↓                           │
│  5. Display response                     │
│              ↓                           │
│  6. Auto-store if worth remembering      │
│              ↓                           │
│  7. Go back to step 1                    │
└──────────────────────────────────────────┘
```

If wake word is not available (missing API key), it falls back to direct listening without requiring a wake word first.

---

## 8. Contrastive Learning — Personalized Retrieval

### 8.1 Motivation

The default Gemini embeddings are **generic** — they capture general semantic similarity but don't understand the specific user's retrieval patterns. For example:

- Query: "What do I study?" → Gemini might rank "I use VS Code" high because both are tech-related
- With a fine-tuned model: The model learns that "study" queries should prioritize academic memories, not tool preferences

**Contrastive learning** trains a local model to understand **which memories are relevant for which queries**, specific to this user's data.

**Research Contribution:**
- Personalized retrieval via contrastive fine-tuning on user-specific data
- Self-supervised data pipeline (LLM generates training data, no manual labeling)
- Continuous learning as memories grow
- Benchmarkable: Gemini baseline vs fine-tuned Recall@K comparison

---

### 8.2 Triplet Generation Service

**File:** `src/services/triplet_service.py`

The self-supervised data pipeline. Generates training data without any manual labeling.

**What is a triplet?**
```
(query, positive, negative)
  ↓          ↓           ↓
"What do   "I study    "I like
 I study?"  CS at IISC"  badminton"
```

The model learns: `query` should be closer to `positive` than to `negative` in embedding space.

**How triplets are generated:**

**Step 1 — Generate queries using the LLM:**
For each stored memory, the LLM generates natural questions that this memory should answer.

```
Memory: "I study Computer Science at IISC Bangalore."
    ↓
LLM Prompt: "Generate 2 natural questions that this memory answers..."
    ↓
Queries: ["What do I study?", "Which university do I attend?"]
```

**Step 2 — Select hard negatives:**
For each triplet, a "hard negative" is selected — a memory that is somewhat related but NOT the correct answer. This is more informative for training than random negatives.

Strategy:
1. Compute cosine similarity between the anchor memory and all other memories
2. Sort by similarity
3. Skip the top 20% (too similar — might actually be valid)
4. Pick from the 20–60% range (the "hard negative" zone)

```
Memory: "I study CS at IISC"
    ↓ Similarities:
    0.92 - "I'm working on a research paper about AI" ← too similar, skip
    0.78 - "I use VS Code as my editor"               ← HARD NEGATIVE ✓
    0.45 - "My dog's name is Bruno"                    ← too easy, skip
```

**Step 3 — Form triplets:**
```json
{
    "query": "What do I study?",
    "positive": "I study Computer Science at IISC Bangalore.",
    "negative": "I use VS Code as my primary code editor."
}
```

**Minimum requirement:** 3 memories to generate triplets (need at least 1 positive + negatives).

---

### 8.3 Contrastive Training Pipeline

**File:** `src/ml/contrastive_trainer.py`

The core ML component that fine-tunes a sentence transformer model.

**Base Model:** `sentence-transformers/all-MiniLM-L6-v2`
- 6-layer transformer, 22M parameters
- Output dimension: 384
- ~80MB model size
- Runs entirely on CPU (no GPU needed)

**Training Process:**

```
InputExample(texts=[query, positive, negative])
         ↓
    TripletLoss
    (learns to minimize distance(query, positive)
     while maximizing distance(query, negative))
         ↓
    Fine-tuned model saved to models/retriever/
```

**Implementation details:**

```python
class ContrastiveTrainer:
    def train(self, triplets, epochs=3, batch_size=16):
        # Convert to InputExamples
        train_examples = [
            InputExample(texts=[t["query"], t["positive"], t["negative"]])
            for t in triplets
        ]

        # TripletLoss with default margin
        train_loss = losses.TripletLoss(model=self.model)

        # Warmup: 10% of total training steps
        warmup_steps = int(len(train_dataloader) * epochs * 0.1)

        # Fine-tune
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=self.model_dir
        )
```

**Warm-Starting:** If a previous checkpoint exists at `models/retriever/`, the trainer loads it instead of the base model. This means each retraining continues from where the last one left off — true continuous learning.

**Post-Training Re-Embedding:**
After training, ALL stored memories are re-encoded with the updated model:

```python
def reembed_all(self, storage_service):
    memories = storage_service.get_all_memories()
    texts = [m.text for m in memories]
    learned_embeddings = self.encode(texts)  # batch encode
    for memory, emb in zip(memories, learned_embeddings):
        storage_service.update_learned_embedding(memory.id, emb)
```

**Training Log:** Each training run is logged to `models/retriever/training_log.json`:

```json
{
    "runs": [
        {
            "version": 1,
            "timestamp": "2026-02-11T14:30:00",
            "num_triplets": 40,
            "epochs": 3,
            "duration_seconds": 45.2,
            "memory_count": 20
        },
        {
            "version": 2,
            "timestamp": "2026-02-12T10:15:00",
            "num_triplets": 80,
            "epochs": 3,
            "duration_seconds": 78.6,
            "memory_count": 40
        }
    ]
}
```

---

### 8.4 Dual-Embedding Retrieval

**File:** `src/core/memory_manager.py`

The memory manager transparently switches between Gemini and learned embeddings.

**On Memory Storage:**
```python
# Always generate Gemini embedding
embedding = self.embedding_service.get_embedding(text)  # 768-dim, API call

# Also generate learned embedding if model is trained
learned_emb = None
if self.contrastive_trainer.is_trained():
    learned_emb = self.contrastive_trainer.encode([text])[0]  # 384-dim, local

memory_entry = MemoryEntry(
    text=text,
    embedding=embedding,
    learned_embedding=learned_emb
)
```

**On Search:**
```python
if trained_model_exists and memories_have_learned_embeddings:
    # LEARNED PATH: fast, local, personalized
    query_emb = self.contrastive_trainer.encode([query])[0]
    score = cosine_similarity(query_emb, memory.learned_embedding)
else:
    # GEMINI PATH: original behavior, API call
    query_emb = self.embedding_service.get_embedding(query)
    score = cosine_similarity(query_emb, memory.embedding)
```

**Lazy Loading:** The `ContrastiveTrainer` is loaded lazily (on first access) via a `@property`. This prevents importing PyTorch at startup if it's not needed.

---

### 8.5 Auto-Training (Background Thread)

**File:** `src/core/memory_manager.py`

Training is triggered automatically when enough new memories accumulate.

**Threshold:** `RETRAIN_THRESHOLD = 20` (configurable in `memory_manager.py`)

**How it works:**

```
Memory stored → Check: (current_count - last_trained_count) >= 20?
                    │
                   YES → Start background thread → _auto_train()
                    │
                   NO → Continue normally
```

**Auto-train runs in a daemon thread:**
- The response to the user returns immediately with `"training_triggered": true`
- Training happens in the background (2–5 minutes on CPU)
- A `_is_training` flag prevents concurrent training runs
- Once complete, new searches automatically use the updated model

**Training lifecycle tracking:**
- `_last_trained_count` is initialized from `training_log.json` on startup
- After each training run, the memory count at training time is saved
- The threshold compares current count vs last trained count

**Example timeline:**
```
Memory  1:  _last_trained_count=0, diff=1  → no train
Memory  5:  diff=5  → no train
Memory 19:  diff=19 → no train
Memory 20:  diff=20 → AUTO-TRAIN TRIGGERED (background thread)
            → triplet generation (LLM) → fine-tuning → re-embedding
            → _last_trained_count updated to 20
Memory 21:  diff=1  → no train
...
Memory 40:  diff=20 → AUTO-TRAIN TRIGGERED again
```

---

### 8.6 Continuous Learning Loop

The complete lifecycle as the memory bank grows:

```
Phase 1: Cold Start (0–19 memories)
├── All storage uses Gemini embeddings (768-dim)
├── All retrieval uses Gemini embeddings
└── No training occurs

Phase 2: First Training (memory 20 triggers training)
├── LLM generates ~40 triplets from 20 memories
├── Base MiniLM fine-tuned → model v1
├── All 20 memories re-embedded with model v1
└── Future searches use learned embeddings (384-dim)

Phase 3: Continuous Improvement (every 20 new memories)
├── Model warm-starts from previous checkpoint
├── Triplets generated from ALL memories (not just new ones)
├── Model gets increasingly specialized to this user
└── Retrieval accuracy improves with each cycle

Phase 4: Mature System (100+ memories)
├── Model deeply understands user's memory landscape
├── Retrieval is highly personalized
├── Each retrain cycle adds incremental improvements
└── Can demonstrate measurable improvement over baseline
```

---

## 9. CLI Application

**File:** `main.py`

Interactive terminal application built with the **Rich** library.

### Menu Options

| # | Option | Description |
|---|---|---|
| 1 | **Add Memory** | Enter text → LLM decides if it's worth storing |
| 2 | **Search Memories** | Enter a query → see top matched memories with scores |
| 3 | **Chat with Memory** | Interactive chat loop with RAG-based responses |
| 4 | **Voice Chat** | Wake word → speech → chat → display response (loop) |
| 5 | **Train Retriever** | Manually trigger contrastive learning training |
| 6 | **Exit** | Exit the application |

### Train Retriever Flow (Option 5)

When selected, the CLI walks through 3 steps with Rich status displays:

```
Step 1: Generating training triplets...
  → Asks LLM to generate queries for each memory
  → Shows sample triplets in a table

Step 2: Training contrastive model...
  → Fine-tunes MiniLM with TripletLoss
  → Shows training metrics (version, duration, triplets used)

Step 3: Re-embedding all memories...
  → Encodes all memories with the updated model
  → Shows count of re-embedded memories
```

---

## 10. Flask REST API

**File:** `app.py`

RESTful API running on port 5000 (default).

### Endpoints

#### `GET /health`
Health check endpoint.

**Response:**
```json
{"status": "healthy", "service": "Quantum Memory Layer"}
```

---

#### `POST /memory`
Add a new memory (LLM decides if it's worth storing).

**Request:**
```json
{"text": "I like Python programming"}
```

**Response (stored):**
```json
{
    "stored": true,
    "reason": "This is a programming language preference.",
    "message": "Memory stored successfully."
}
```

**Response (with auto-train):**
```json
{
    "stored": true,
    "reason": "...",
    "message": "Memory stored successfully. Background retraining started.",
    "training_triggered": true
}
```

---

#### `GET /search?q=<query>&limit=<n>`
Search stored memories by semantic similarity.

**Parameters:**
| Param | Type | Default | Description |
|---|---|---|---|
| `q` | string | required | Search query |
| `limit` | int | 5 | Max results to return |

**Response:**
```json
[
    {
        "text": "I like Python programming",
        "score": 0.8723,
        "created_at": "2026-02-11T14:30:00",
        "id": 1
    }
]
```

---

#### `POST /chat`
Chat with the LLM using memory context (RAG).

**Request:**
```json
{"text": "What programming languages do I like?"}
```

**Response:**
```json
{
    "response": "Based on what I know about you, you enjoy Python programming!",
    "memory_action": {
        "stored": false,
        "reason": "This is a question, not a fact to remember.",
        "message": "Memory not stored."
    }
}
```

---

#### `POST /train`
Trigger contrastive learning training.

**Request (optional body):**
```json
{"epochs": 3, "queries_per_memory": 2}
```

**Response:**
```json
{
    "duration_seconds": 45.2,
    "num_triplets": 40,
    "epochs": 3,
    "model_version": 1,
    "model_path": "models/retriever",
    "memories_reembedded": 20
}
```

---

#### `GET /train/status`
Check the status of the contrastive retriever.

**Response (not trained):**
```json
{
    "available": true,
    "is_trained": false,
    "memory_count": 8
}
```

**Response (trained):**
```json
{
    "available": true,
    "is_trained": true,
    "memory_count": 45,
    "total_versions": 2,
    "latest_training": {
        "version": 2,
        "timestamp": "2026-02-11T14:30:00",
        "num_triplets": 80,
        "epochs": 3,
        "duration_seconds": 78.6,
        "memory_count": 40
    }
}
```

---

## 11. Installation & Setup

### Prerequisites
- Python 3.10+
- A Google Gemini API key
- (Optional) Picovoice access key for wake word detection
- (Optional) A working microphone for voice features

### Step-by-step

```bash
# 1. Clone the repository
git clone <repo-url>
cd memory-layer

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
copy .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 5. Run the CLI
python main.py

# OR run the Flask API
python app.py
```

### Installing Contrastive Learning Dependencies

```bash
# This is ~2GB download (PyTorch + sentence-transformers)
pip install sentence-transformers torch
```

The system works without these packages — it just uses Gemini embeddings only.

---

## 12. Verification & Testing

### Core System Tests

**File:** `verify.py`

Tests the fundamental memory operations:
- Adding memories
- Searching memories
- Chat with memory context
- Memory update/conflict resolution

```bash
python verify.py
```

### Contrastive Learning Benchmark

**File:** `verify_contrastive.py`

End-to-end benchmark with 5 tests:

| Test | What It Verifies |
|---|---|
| **Cold Start Fallback** | Without a trained model, system uses Gemini (no crash) |
| **Triplet Generation** | LLM generates valid (query, positive, negative) triplets |
| **Model Training** | MiniLM fine-tunes without errors, checkpoint saved |
| **Re-embedding** | All memories get learned embeddings after training |
| **Retrieval Comparison** | Side-by-side Gemini vs fine-tuned retrieval results |

```bash
python verify_contrastive.py
```

Uses a separate test database (`test_contrastive.db`) to avoid corrupting production data.

### Interactive Demo

**File:** `demo.py`

Interactive debugging tool that shows internal system state:
- Raw embeddings
- Similarity scores
- LLM decisions

```bash
python demo.py
```

---

## 13. Dependencies

### Core Dependencies (`requirements.txt`)

| Package | Version | Purpose |
|---|---|---|
| `google-generativeai` | latest | Gemini LLM and embedding API |
| `python-dotenv` | latest | `.env` file loading |
| `rich` | latest | Terminal UI (colors, tables, panels) |
| `numpy` | latest | Cosine similarity computation |
| `flask` | latest | REST API server |

### Voice Dependencies

| Package | Purpose |
|---|---|
| `SpeechRecognition` | Google Speech-to-Text wrapper |
| `pyaudio` | Microphone audio capture |
| `pvporcupine` | On-device wake word detection |
| `pvrecorder` | Audio recording for Porcupine |

### ML Dependencies (Optional)

| Package | Size | Purpose |
|---|---|---|
| `sentence-transformers` | ~500MB | MiniLM model loading, training utilities |
| `torch` | ~1.5GB | PyTorch deep learning framework |

> **Note:** The ML dependencies are optional. Without them, the system operates using Gemini embeddings only (original behavior). The contrastive learning features are disabled gracefully.
