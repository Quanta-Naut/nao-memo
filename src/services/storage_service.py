import sqlite3
import json
from typing import List
from src.core.config import Config
from src.models.memory_entry import MemoryEntry

class StorageService:
    def __init__(self):
        self.db_path = Config.DB_PATH
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initializes the database table if it doesn't exist, and migrates schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                learned_embedding TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        conn.commit()
        # Migrate legacy tables
        self._migrate_schema(cursor, conn)
        conn.close()

    def _migrate_schema(self, cursor, conn):
        """Adds learned_embedding column to legacy tables if it doesn't exist."""
        cursor.execute("PRAGMA table_info(memories)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        if "learned_embedding" not in existing_columns:
            cursor.execute("ALTER TABLE memories ADD COLUMN learned_embedding TEXT")
            conn.commit()

    def add_memory(self, entry: MemoryEntry):
        """Adds a new memory entry to the database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO memories (text, embedding, learned_embedding, metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', entry.to_db_tuple())
        conn.commit()
        conn.close()

    def get_all_memories(self) -> List[MemoryEntry]:
        """Retrieves all memories."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM memories')
        rows = cursor.fetchall()
        conn.close()
        return [MemoryEntry.from_db_tuple(row) for row in rows]

    def get_memory_count(self) -> int:
        """Returns the total number of stored memories."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM memories')
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def update_learned_embedding(self, memory_id: int, learned_embedding: List[float]):
        """Updates the learned embedding for a specific memory after retraining."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE memories SET learned_embedding = ? WHERE id = ?',
            (json.dumps(learned_embedding), memory_id)
        )
        conn.commit()
        conn.close()

    def delete_memory(self, memory_id: int):
        """Deletes a memory by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM memories WHERE id = ?', (memory_id,))
        conn.commit()
        conn.close()
