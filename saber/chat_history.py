# -*- coding: utf-8 -*-
"""saber.chat_history

Lightweight SQLite-backed chat history for the SABER UI.
Stores a single conversation with all user/system messages
and their metadata (confidence, domains, flags, etc.).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional


class ChatHistory:
    """Persistent chat history backed by SQLite.

    Stores messages for a single active conversation.
    Each message includes role, content, and optional SABER metadata
    (confidence, domains activated, verification cycles, etc.).
    """

    def __init__(self, db_path: str = "data/chat_history.db") -> None:
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create the messages table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    timestamp REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp
                ON messages(timestamp)
            """)
            conn.commit()

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a message to the conversation.

        Parameters
        ----------
        role : str
            'user' or 'system'
        content : str
            The message text.
        metadata : dict or None
            SABER-specific data (confidence, domains, flags, etc.).

        Returns
        -------
        dict
            The stored message record.
        """
        msg_id = f"msg-{uuid.uuid4().hex[:12]}"
        ts = time.time()
        meta_json = json.dumps(metadata or {}, default=str)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO messages (id, role, content, metadata, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (msg_id, role, content, meta_json, ts),
            )
            conn.commit()

        return {
            "id": msg_id,
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": ts,
        }

    def get_messages(self) -> List[Dict[str, Any]]:
        """Return all messages in chronological order."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, role, content, metadata, timestamp "
                "FROM messages ORDER BY timestamp ASC"
            ).fetchall()

        messages = []
        for row in rows:
            messages.append({
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"]),
                "timestamp": row["timestamp"],
            })
        return messages

    def clear(self) -> None:
        """Delete all messages (start a fresh conversation)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages")
            conn.commit()

    def message_count(self) -> int:
        """Return the number of stored messages."""
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        return count
