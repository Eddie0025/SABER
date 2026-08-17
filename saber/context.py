import logging
from typing import List, Dict, Optional

logger = logging.getLogger("SABER_Context")


class SessionContext:
    """
    Multi-turn conversation history manager.
    Stores chat messages and provides them for prompt construction.
    Enforces a max_turns limit to prevent unbounded context growth.
    """

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []
        self.metadata: Dict[str, str] = {}  # session-level metadata (e.g., active domain)

    def add_message(self, role: str, content: str):
        """Append a message to the conversation history."""
        self.history.append({"role": role, "content": content})
        # Trim oldest messages if we exceed max_turns (keep system prompt if present)
        if len(self.history) > self.max_turns * 2:
            # Preserve the first message if it's a system prompt
            if self.history[0]["role"] == "system":
                self.history = [self.history[0]] + self.history[-(self.max_turns * 2 - 1):]
            else:
                self.history = self.history[-(self.max_turns * 2):]

    def get_history(self) -> List[Dict[str, str]]:
        """Return the full message history for prompt construction."""
        return list(self.history)

    def get_last_user_message(self) -> Optional[str]:
        """Return the most recent user message."""
        for msg in reversed(self.history):
            if msg["role"] == "user":
                return msg["content"]
        return None

    def set_metadata(self, key: str, value: str):
        """Store session-level metadata (e.g., active specialist, routing info)."""
        self.metadata[key] = value

    def get_metadata(self, key: str) -> Optional[str]:
        """Retrieve session-level metadata."""
        return self.metadata.get(key)

    def clear(self):
        """Reset the conversation for a new session."""
        self.history.clear()
        self.metadata.clear()
        logger.info("Session context cleared.")

    def __len__(self) -> int:
        return len(self.history)
