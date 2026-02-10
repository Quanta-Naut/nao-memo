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
        """Initializes the database table if it doesn't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    def add_memory(self, entry: MemoryEntry):
        """Adds a new memory entry to the database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO memories (text, embedding, metadata, created_at)
            VALUES (?, ?, ?, ?)
        ''', entry.to_db_tuple())
        conn.commit()
        conn.close()

    def get_all_memories(self) -> List[MemoryEntry]:
        """Retrieves all memories. In a production system, you'd use a vector DB."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM memories')
        rows = cursor.fetchall()
        conn.close()
        return [MemoryEntry.from_db_tuple(row) for row in rows]

    def delete_memory(self, memory_id: int):
        """Deletes a memory by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM memories WHERE id = ?', (memory_id,))
        conn.commit()
        conn.close()
