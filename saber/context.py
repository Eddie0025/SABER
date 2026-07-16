# -*- coding: utf-8 -*-
"""saber.context

Session Context Memory — maintains per-session conversation history
for multi-turn interactions, similar to ChatGPT and Claude.

Memory Strategy — Summary + Recent Messages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Instead of naively truncating old messages, this system uses a
**rolling summary** approach:

1. The last ``keep_recent`` message pairs are kept verbatim.
2. All older messages are compressed into a **running summary** that
   preserves only the important parts:
   - Explicit user instructions, requests, and preferences
   - Key specs, numbers, calculations, and technical details
   - Important decisions and conclusions
   - Domain-specific facts and references
3. Redundant text, pleasantries, boilerplate, and filler are discarded.

The summary is updated incrementally each time the conversation
exceeds the token budget, so context is never silently lost.

Usage
~~~~~
::

    from saber.context import SessionMemory

    memory = SessionMemory(max_tokens=2048, keep_recent=4)

    sid = memory.create_session(system_prompt="You are helpful.")
    memory.add_message(sid, "user", "Set temperature to 0.7")
    memory.add_message(sid, "assistant", "Temperature set to 0.7.")
    # ... many more turns ...

    # Returns: system prompt + rolling summary + last 4 messages
    history = memory.get_history(sid)

"""

from __future__ import annotations

import re
import time
import uuid
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ------------------------------------------------------------------
# Data Classes
# ------------------------------------------------------------------

@dataclass
class Message:
    """A single message in a conversation session."""
    role: str          # "system", "user", or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class Session:
    """A single conversation session with its message history."""
    session_id: str
    messages: List[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    system_prompt: Optional[str] = None

    # Rolling summary of older conversation turns
    running_summary: str = ""

    # How many messages have already been summarized (index pointer)
    _summarized_up_to: int = 0

    @property
    def turn_count(self) -> int:
        """Number of user turns in the session."""
        return sum(1 for m in self.messages if m.role == "user")


# ------------------------------------------------------------------
# Heuristic Summarizer (no LLM needed)
# ------------------------------------------------------------------

class _HeuristicSummarizer:
    """Extracts important information from conversation messages
    without loading an LLM. Focuses on:
    - Explicit user instructions and preferences
    - Numbers, specs, calculations, and technical details
    - Key decisions and conclusions
    - Named entities and domain terms
    """

    # Patterns that indicate important content worth keeping
    _IMPORTANT_PATTERNS = [
        # Numbers, measurements, calculations
        re.compile(r'\d+\.?\d*\s*(?:kg|m/s|N|Hz|GB|MB|tokens?|epochs?|steps?|%|degrees?)', re.I),
        # Key instruction verbs from the user
        re.compile(r'\b(?:set|use|configure|change|update|make|ensure|always|never|must|should|remember|note)\b', re.I),
        # Technical specs and parameters
        re.compile(r'\b(?:temperature|learning[_ ]rate|batch[_ ]size|model|version|port|path|file|directory|domain|mode)\b', re.I),
        # Explicit values / assignments
        re.compile(r'(?:=|is set to|set to|changed to|updated to|equals)\s*.+', re.I),
        # Questions (usually important)
        re.compile(r'\?$'),
        # Code / formulas
        re.compile(r'[A-Z_]{2,}\s*=|def |class |import |F\s*=\s*m'),
        # Lists / enumerations
        re.compile(r'^\s*[\d\-\*•]\s+', re.M),
    ]

    # Patterns indicating boilerplate / filler to skip
    _FILLER_PATTERNS = [
        re.compile(r'^(?:ok(?:ay)?|sure|thanks?|thank you|got it|alright|great|yes|no|hi|hello|hey)[\.\!\,]?\s*$', re.I),
        re.compile(r'^(?:let me know|feel free|happy to help|you\'re welcome|no problem)', re.I),
    ]

    @classmethod
    def summarize_messages(
        cls,
        messages: List[Message],
        existing_summary: str = "",
    ) -> str:
        """Produce a condensed summary of messages.

        Extracts key user instructions, specs, numbers, decisions,
        and important assistant conclusions. Drops filler and boilerplate.

        Parameters
        ----------
        messages : list[Message]
            The messages to summarize.
        existing_summary : str
            Any prior summary to build upon.

        Returns
        -------
        str
            Updated rolling summary.
        """
        extracted: List[str] = []

        for msg in messages:
            lines = msg.content.strip().splitlines()
            important_lines: List[str] = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Skip pure filler
                if any(p.match(line) for p in cls._FILLER_PATTERNS):
                    continue

                # Keep lines matching important patterns
                if any(p.search(line) for p in cls._IMPORTANT_PATTERNS):
                    important_lines.append(line)
                    continue

                # For user messages: keep everything that isn't filler
                # (users usually say important things)
                if msg.role == "user":
                    important_lines.append(line)
                    continue

                # For assistant messages: keep shorter factual lines,
                # skip long explanatory paragraphs
                if msg.role == "assistant" and len(line) < 150:
                    important_lines.append(line)

            if important_lines:
                role_label = "User" if msg.role == "user" else "Assistant"
                condensed = "; ".join(important_lines[:6])  # cap per message
                # Truncate very long condensed lines
                if len(condensed) > 300:
                    condensed = condensed[:297] + "..."
                extracted.append(f"[{role_label}] {condensed}")

        # Build the new summary
        new_parts = "\n".join(extracted)

        if existing_summary:
            combined = f"{existing_summary}\n{new_parts}"
        else:
            combined = new_parts

        # Final trim: if summary itself is getting too long, keep the
        # most recent portion (last ~1500 chars ≈ 375 tokens)
        if len(combined) > 1500:
            # Keep the first line (oldest context anchor) + recent lines
            lines = combined.strip().splitlines()
            anchor = lines[0]
            # Take the most recent lines that fit
            recent_lines = []
            char_count = len(anchor) + 1
            for line in reversed(lines[1:]):
                if char_count + len(line) + 1 > 1400:
                    break
                recent_lines.insert(0, line)
                char_count += len(line) + 1
            combined = anchor + "\n...\n" + "\n".join(recent_lines)

        return combined.strip()


# ------------------------------------------------------------------
# LLM-Powered Summarizer
# ------------------------------------------------------------------

class _LLMSummarizer:
    """Uses a loaded LLMEngine to produce intelligent summaries."""

    SUMMARY_PROMPT = (
        "You are a context memory manager. Summarize the following conversation "
        "excerpt into a compact context block. Rules:\n"
        "1. KEEP: All explicit user instructions, preferences, and requests.\n"
        "2. KEEP: All numbers, specs, calculations, formulas, and technical details.\n"
        "3. KEEP: Key decisions, conclusions, and important facts.\n"
        "4. KEEP: Any named entities, file paths, model names, or domain terms.\n"
        "5. DROP: Pleasantries, greetings, filler, boilerplate, and redundant explanations.\n"
        "6. DROP: Verbose reasoning that can be condensed into a single line.\n"
        "7. Format as bullet points. Be extremely concise.\n\n"
        "Existing context summary (if any):\n{existing_summary}\n\n"
        "New conversation to summarize:\n{conversation}\n\n"
        "Output ONLY the updated context summary as bullet points:"
    )

    @classmethod
    def summarize_with_engine(
        cls,
        engine,
        messages: List[Message],
        existing_summary: str = "",
    ) -> str:
        """Produce an LLM-generated summary of conversation messages.

        Parameters
        ----------
        engine : LLMEngine
            An already-loaded LLM engine instance.
        messages : list[Message]
            Messages to summarize.
        existing_summary : str
            Prior summary to incorporate.

        Returns
        -------
        str
            The LLM-generated summary.
        """
        # Format the conversation for the prompt
        conv_parts = []
        for msg in messages:
            label = "User" if msg.role == "user" else "Assistant"
            conv_parts.append(f"{label}: {msg.content[:500]}")

        conversation = "\n".join(conv_parts)

        prompt = cls.SUMMARY_PROMPT.format(
            existing_summary=existing_summary or "(none)",
            conversation=conversation,
        )

        try:
            summary = engine.generate(
                prompt,
                system_prompt="You are a memory compression engine. Output only bullet points."
            )
            return summary.strip()
        except Exception as e:
            # Fallback to heuristic if LLM fails
            print(f"[context] LLM summarization failed ({e}), using heuristic fallback")
            return _HeuristicSummarizer.summarize_messages(messages, existing_summary)


# ------------------------------------------------------------------
# Session Memory Manager
# ------------------------------------------------------------------

class SessionMemory:
    """Manages conversation sessions with summary-based context compression.

    Instead of simply dropping old messages, this system:
    1. Keeps the last ``keep_recent`` messages verbatim.
    2. Compresses all older messages into a rolling summary that
       preserves important details, specs, and explicit user instructions.

    Parameters
    ----------
    max_tokens : int
        Approximate token budget for the full context window.
    keep_recent : int
        Number of recent messages to keep in full (not summarized).
        Counted as individual messages (not pairs).
    max_sessions : int
        Maximum concurrent sessions before oldest is evicted.
    use_llm_summary : bool
        If True, use the LLM for summarization when an engine is
        available. If False, always use the fast heuristic summarizer.
    """

    CHARS_PER_TOKEN = 4

    def __init__(
        self,
        max_tokens: int = 2048,
        keep_recent: int = 6,
        max_sessions: int = 100,
        use_llm_summary: bool = False,
    ) -> None:
        self._lock = threading.Lock()
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent
        self.max_sessions = max_sessions
        self.use_llm_summary = use_llm_summary
        self._sessions: Dict[str, Session] = {}

    # ------------------------------------------------------------------
    # Session Lifecycle
    # ------------------------------------------------------------------

    def create_session(
        self,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Create a new conversation session and return its ID."""
        sid = session_id or f"sess-{uuid.uuid4().hex[:12]}"
        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                self._evict_oldest()
            self._sessions[sid] = Session(
                session_id=sid,
                system_prompt=system_prompt,
            )
        return sid

    def get_or_create_session(
        self,
        session_id: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Return existing session or create a new one."""
        with self._lock:
            if session_id not in self._sessions:
                if len(self._sessions) >= self.max_sessions:
                    self._evict_oldest()
                self._sessions[session_id] = Session(
                    session_id=session_id,
                    system_prompt=system_prompt,
                )
            return session_id

    def clear_session(self, session_id: str) -> None:
        """Delete a session and all its history."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def list_sessions(self) -> List[str]:
        return list(self._sessions.keys())

    # ------------------------------------------------------------------
    # Message Operations
    # ------------------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """Append a message and trigger summarization if needed.

        Parameters
        ----------
        session_id : str
            Must already exist.
        role : str
            One of "system", "user", or "assistant".
        content : str
            The message text.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(f"Session '{session_id}' does not exist. "
                               f"Call create_session() first.")
            session.messages.append(Message(role=role, content=content))
            session.last_active = time.time()

            # Check if we need to compress older messages
            self._maybe_compress(session)

    def get_history(
        self,
        session_id: str,
        include_system: bool = True,
    ) -> List[Dict[str, str]]:
        """Return context-aware history: system + summary + recent messages.

        The returned list contains:
        1. The system prompt (if any and ``include_system`` is True).
        2. A summary message containing the compressed history of
           older turns (if any summarization has occurred).
        3. The last ``keep_recent`` messages verbatim.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return []

            result: List[Dict[str, str]] = []

            # 1. System prompt
            if include_system and session.system_prompt:
                result.append({"role": "system", "content": session.system_prompt})

            # 2. Rolling summary of older conversation
            if session.running_summary:
                summary_content = (
                    f"[Previous conversation summary]\n"
                    f"{session.running_summary}\n"
                    f"[End of summary — recent messages follow]"
                )
                result.append({"role": "system", "content": summary_content})

            # 3. Recent messages (kept verbatim)
            recent_start = max(0, len(session.messages) - self.keep_recent)
            for msg in session.messages[recent_start:]:
                result.append(msg.to_dict())

            return result

    def get_summary(self, session_id: str) -> str:
        """Return the current rolling summary for a session."""
        session = self._sessions.get(session_id)
        if session is None:
            return ""
        return session.running_summary

    def get_formatted_prompt(
        self,
        session_id: str,
        include_system: bool = True,
    ) -> str:
        """Return conversation history formatted as a single prompt string."""
        history = self.get_history(session_id, include_system=include_system)
        parts: List[str] = []
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|{role}|>\n{content}")
        parts.append("<|assistant|>")
        return "\n".join(parts) + "\n"

    def get_turn_count(self, session_id: str) -> int:
        """Return the number of user turns in a session."""
        session = self._sessions.get(session_id)
        if session is None:
            return 0
        return session.turn_count

    # ------------------------------------------------------------------
    # Summarization & Compression
    # ------------------------------------------------------------------

    def _maybe_compress(self, session: Session, engine=None) -> None:
        """Check if older messages need to be summarized.

        Called after every add_message. If the number of unsummarized
        messages exceeds ``keep_recent``, the older ones are compressed
        into the running summary.

        Parameters
        ----------
        session : Session
            The session to potentially compress.
        engine : LLMEngine or None
            If provided and ``use_llm_summary`` is True, uses the LLM
            for summarization. Otherwise uses the heuristic.
        """
        total_messages = len(session.messages)

        # Only compress when we have more messages than keep_recent
        if total_messages <= self.keep_recent:
            return

        # Messages that should be summarized: everything before the
        # most recent `keep_recent` messages
        summarize_end = total_messages - self.keep_recent

        # Only summarize new messages (from _summarized_up_to to summarize_end)
        if summarize_end <= session._summarized_up_to:
            return

        messages_to_summarize = session.messages[session._summarized_up_to:summarize_end]

        if not messages_to_summarize:
            return

        # Generate the summary
        if self.use_llm_summary and engine is not None:
            session.running_summary = _LLMSummarizer.summarize_with_engine(
                engine, messages_to_summarize, session.running_summary
            )
        else:
            session.running_summary = _HeuristicSummarizer.summarize_messages(
                messages_to_summarize, session.running_summary
            )

        # Update the pointer
        session._summarized_up_to = summarize_end

    def force_summarize(
        self,
        session_id: str,
        engine=None,
    ) -> str:
        """Force an immediate summarization of older messages.

        Useful when you want to trigger LLM-based summarization
        while the engine is already loaded.

        Parameters
        ----------
        session_id : str
            The session to summarize.
        engine : LLMEngine or None
            If provided, uses LLM summarization.

        Returns
        -------
        str
            The updated running summary.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return ""
            self._maybe_compress(session, engine=engine)
            return session.running_summary

    # ------------------------------------------------------------------
    # Token Estimation
    # ------------------------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from character length."""
        return max(1, len(text) // self.CHARS_PER_TOKEN)

    def _total_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Estimate total token count for a list of messages."""
        return sum(self._estimate_tokens(m["content"]) for m in messages)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _evict_oldest(self) -> None:
        """Remove the least recently active session (caller holds lock)."""
        if not self._sessions:
            return
        oldest_sid = min(
            self._sessions,
            key=lambda sid: self._sessions[sid].last_active,
        )
        del self._sessions[oldest_sid]
