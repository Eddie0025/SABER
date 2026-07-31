import sqlite3
import json
import logging

logger = logging.getLogger("SABER_SharedMemory")

class SharedMemory:
    """
    SQLite-backed shared memory state that allows different Coding Specialists 
    (e.g., Python, JS, SQL, Architecture) to hand off tasks, share context, 
    and maintain persistence across inference calls.
    """
    def __init__(self, db_path="saber_memory.db"):
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS context_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            
    def set(self, key: str, value: dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO context_state (key, value, updated_at) 
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET 
                value=excluded.value, updated_at=CURRENT_TIMESTAMP
            ''', (key, json.dumps(value)))
            conn.commit()
            
    def get(self, key: str) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT value FROM context_state WHERE key=?', (key,))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return {}
            
    def clear(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('DELETE FROM context_state')
            conn.commit()
