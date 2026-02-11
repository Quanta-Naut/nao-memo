# System Architecture

The following diagram illustrates the workflow of the **Quantum Memory Layer** for your research paper.

## Component Workflow

1.  **Input Processing**: User speech (via Microphone) triggers Wake Word (`Porcupine`), then captures command (`SpeechRecognition`).
2.  **Memory Management**: Text enters the `MemoryManager`.
3.  **Decision Phase**: `LLMService` (Gemini) decides if the input is worth storing (Importance Check) and checks for conflicts with existing memories (Update Logic).
4.  **Vectorization**: Valid memories are sent to `EmbeddingService` (Gemini Embedding Model) to generate 768-dim vectors.
5.  **Storage**: Vectors and metadata are committed to `StorageService` (SQLite).
6.  **Retrieval (RAG)**: Chat queries are embedded, compared against stored vectors (Cosine Similarity), and top matches are fed to the LLM for context-aware generation.

## Architecture Diagram (Mermaid)

```mermaid
graph TD
    %% User Input Layer
    User((User)) -->|Voice| Mic[Microphone Input]
    Mic -->|Raw Audio| Wake{Wake Word<br/>(Porcupine)}
    Wake -- "Jarvis" --> STT[STT Service<br/>(Google Speech)]
    Wake -- Silence --> Mic
    
    User -->|Text| UI[CLI / API Interface]
    STT -->|Transcribed Text| UI

    subgraph "Core Memory System"
        UI -->|Input Signal| Manager[Memory Manager]
        
        %% Decision Logic
        Manager -->|Analyze Content| Decision_LLM[LLM Service<br/>(Decision Gate)]
        Decision_LLM -->|Is Important?| Check{Store?}
        Check -- No --> Discard[Ignore]
        
        %% Conflict Resolution
        Check -- Yes --> Conflict_Check{Conflict Check?}
        Conflict_Check -- "New Info" --> Embed[Embedding Service]
        Conflict_Check -- "Update" --> Delete_Old[Start Update Process]
        Delete_Old --> Embed

        %% Embedding & Storage layer
        Embed -->|Request Vector| Model_Embed[Gemini Embedding Model]
        Model_Embed -->|Vector Array| Storage[(SQLite Vector DB)]
        
        %% Retrieval Loop
        UI -->|Query| Search[Semantic Search Engine]
        Search -->|Cosine Similarity| Storage
        Storage -->|Top-K Context| Context_Filter[Context Filter]
        Context_Filter -->|Augmented Prompt| Gen_LLM[LLM Service<br/>(Generation)]
        Gen_LLM -->|Contextual Response| UI
    end
    
    classDef plain fill:#fff,stroke:#333,stroke-width:2px;
    classDef db fill:#f9f,stroke:#333,stroke-width:2px;
    classDef llm fill:#ccf,stroke:#333,stroke-width:2px;
    
    class User,Mic,Wake,STT,UI plain;
    class Storage db;
    class Decision_LLM,Gen_LLM,Model_Embed llm;
```

## Image Generation Prompt (for external tools)

If you wish to generate a high-resolution, academic-style diagram using DALL-E 3 or Midjourney, use this prompt:

> **Prompt:** A formal, high-contrast block diagram of a Modular AI Memory Architecture for a research paper on a white background. Black lines, clear hierarchical layout.
> **Components:** Top layer: 'User Interaction' (Voice/Text). Middle layer: 'Core Logic' containing a 'Memory Manager', 'Decision LLM', and 'Embedding Model'. Bottom layer: 'Vector Storage' (Cylindrical Database).
> **Flow:** Arrows show data flow from User to Manager, branching to Decision LLM (Filter), then to Embedding Model (Vectorization), and finally to Storage. A return loop shows Retrieval from Storage to Generation LLM for response. 
> **Style:** IEEE technical diagram style, clean sans-serif font, minimal color coding (blue for logic, green for storage), professional schematic.

---

## Contrastive Learning Pipeline

The system includes a **trainable embedding model** that learns personalized memory retrieval through contrastive learning.

### Training Phase (Offline)

1. **Triplet Generation**: `TripletService` uses the LLM to generate natural queries for each stored memory. Each query is paired with its source memory (positive) and a semantically distant memory (negative).
2. **Fine-Tuning**: `ContrastiveTrainer` fine-tunes `all-MiniLM-L6-v2` using `TripletLoss`. The model learns to place queries closer to their correct memories in embedding space.
3. **Re-Embedding**: After training, all stored memories are re-encoded with the fine-tuned model and the `learned_embedding` column is updated.

### Inference Phase (Online)

- If a fine-tuned model exists → queries and memories are encoded with the **learned model** (384-dim)
- If no model exists → falls back to **Gemini embeddings** (768-dim, original behavior)

```mermaid
graph TD
    subgraph "Training Pipeline (Offline)"
        Memories[(Stored Memories)] -->|All texts| TripletGen[Triplet Service]
        TripletGen -->|LLM generates queries| Triplets[("(query, positive, negative)")]
        Triplets --> Trainer[Contrastive Trainer<br/>(TripletLoss)]
        Trainer -->|Fine-tuned| Model[MiniLM Model<br/>(384-dim)]
        Model -->|Re-encode all| Memories
    end

    subgraph "Retrieval (Online)"
        Query[User Query] --> Check{Model<br/>Exists?}
        Check -- Yes --> Learned[Learned Embedding<br/>(384-dim, local)]
        Check -- No --> Gemini[Gemini Embedding<br/>(768-dim, API)]
        Learned --> CosSim[Cosine Similarity]
        Gemini --> CosSim
        Memories -->|Stored vectors| CosSim
        CosSim -->|Top-K| LLM_Gen[LLM Generation]
    end

    classDef ml fill:#fcf,stroke:#333,stroke-width:2px;
    class Trainer,Model ml;
```

### Retrieval Scoring Formula

```
score = cosine_similarity(query_embedding, memory_embedding)
```

Where the embedding model is either:
- **Cold start**: Gemini `models/gemini-embedding-001` (768-dim, API call)
- **After training**: Fine-tuned `all-MiniLM-L6-v2` (384-dim, local inference)

