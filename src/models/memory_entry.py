from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import json

@dataclass
class MemoryEntry:
    text: str
    embedding: List[float]
    learned_embedding: Optional[List[float]] = None  # Fine-tuned MiniLM embedding (384-dim)
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    id: Optional[int] = None

    def to_db_tuple(self):
        return (
            self.text,
            json.dumps(self.embedding),
            json.dumps(self.learned_embedding) if self.learned_embedding else None,
            json.dumps(self.metadata),
            self.created_at.isoformat()
        )

    @classmethod
    def from_db_tuple(cls, row):
        """
        Backwards compatible: handles both 5-column (legacy) and 6-column (new) rows.
        """
        if len(row) == 5:
            id, text, embedding_json, metadata_json, created_at_str = row
            return cls(
                id=id,
                text=text,
                embedding=json.loads(embedding_json),
                metadata=json.loads(metadata_json),
                created_at=datetime.fromisoformat(created_at_str)
            )
        else:
            # New format: id, text, embedding, learned_embedding, metadata, created_at
            id, text, embedding_json, learned_emb_json, metadata_json, created_at_str = row
            return cls(
                id=id,
                text=text,
                embedding=json.loads(embedding_json),
                learned_embedding=json.loads(learned_emb_json) if learned_emb_json else None,
                metadata=json.loads(metadata_json),
                created_at=datetime.fromisoformat(created_at_str)
            )
