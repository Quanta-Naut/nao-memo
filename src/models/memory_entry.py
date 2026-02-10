from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import json

@dataclass
class MemoryEntry:
    text: str
    embedding: List[float]
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    id: Optional[int] = None

    def to_db_tuple(self):
        return (
            self.text,
            json.dumps(self.embedding),
            json.dumps(self.metadata),
            self.created_at.isoformat()
        )

    @classmethod
    def from_db_tuple(cls, row):
        id, text, embedding_json, metadata_json, created_at_str = row
        return cls(
            id=id,
            text=text,
            embedding=json.loads(embedding_json),
            metadata=json.loads(metadata_json),
            created_at=datetime.fromisoformat(created_at_str)
        )
