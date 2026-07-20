# -*- coding: utf-8 -*-
"""saber.specialist

Base class for domain specialists in the SABER architecture.
Specialists now communicate strictly using the Signal Schema lifecycle.
"""

from __future__ import annotations

import uuid
from enum import Enum
import importlib
import pkgutil
import inspect
from typing import Optional

# pyrefly: ignore [missing-import]
from pydantic import BaseModel

from saber.signal import Signal, SignalType, Claim, ClaimStatus
from saber.cot_maintainer import CoTMaintainer

class HealthStatus(str, Enum):
    ONLINE = "ONLINE"
    BUSY = "BUSY"
    DEGRADED = "DEGRADED"
    OFFLINE = "OFFLINE"


class SpecialistMeta(BaseModel):
    """Metadata describing a specialist's capabilities."""
    domain: str
    specialist_id: str
    version: str = "1.0.0"
    model_path: str = ""
    capabilities: list[str] = []
    keywords: list[str] = []
    authority_score: float = 0.5
    health: HealthStatus = HealthStatus.ONLINE


class Specialist:
    """Base class for all SABER domain specialists.

    Lifecycle:
    1. Meta-Reasoning Layer sends a TASK_SIGNAL.
    2. Specialist acknowledges with CONFIRMATION_SIGNAL.
    3. Specialist processes task, returning OUTPUT_SIGNAL (with Claims).
    4. Meta-Reasoning Layer sends VERIFICATION_SIGNAL (compiled output).
    5. Specialist checks meaning, returns VERIFICATION_SIGNAL (GREEN_CHIT) or FLAG_SIGNAL.

    Session Memory:
    Each specialist maintains a SessionMemory instance for multi-turn
    conversations. Use ``chat()`` for conversational interactions that
    remember prior context (like ChatGPT/Claude).
    """

    def __init__(self) -> None:
        from saber.context import SessionMemory
        self.meta = SpecialistMeta(
            domain=self.domain,
            specialist_id=f"SPEC-{self.domain.upper()}-{uuid.uuid4().hex[:6].upper()}",
            keywords=self.keywords,
        )
        # Session context memory for multi-turn conversations
        self._session_memory = SessionMemory(max_tokens=2048, max_sessions=50)
        self.cot = CoTMaintainer()
        self._cached_response = None

    @property
    def domain(self) -> str:
        raise NotImplementedError

    @property
    def keywords(self) -> list[str]:
        """Domain keywords used by the Orchestrator for routing.

        Override this in subclasses to declare the keywords that should
        trigger this specialist's activation.  The Orchestrator builds
        its routing table dynamically from all registered specialists,
        so adding a new specialist with keywords is all that's needed
        to make it routable — no hardcoded tables to update.
        """
        return []

    def load_model(self, model_path: str) -> None:
        """Register the model path to be used by the LLMEngine."""
        self.meta.model_path = model_path

    # ------------------------------------------------------------------
    # Session Context Memory (Multi-Turn Chat)
    # ------------------------------------------------------------------

    def start_session(
        self,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Start a new conversation session for this specialist.

        Parameters
        ----------
        session_id : str or None
            If provided, use this as the session ID. Otherwise auto-generate.
        system_prompt : str or None
            The system prompt to use for this session. If None, a default
            domain-specific prompt is used.

        Returns
        -------
        str
            The session ID.
        """
        if system_prompt is None:
            system_prompt = (
                f"You are an expert {self.domain} AI specialist. "
                f"Provide thorough, evidence-based answers. "
                f"Remember the full conversation context."
            )
        return self._session_memory.create_session(
            session_id=session_id,
            system_prompt=system_prompt,
        )

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
    ) -> str:
        """Send a message in a multi-turn conversation.

        If no ``session_id`` is provided, a default session is created
        (or reused) automatically for this specialist instance.

        Parameters
        ----------
        message : str
            The user's message.
        session_id : str or None
            The session to continue. If None, uses a default session.

        Returns
        -------
        str
            The specialist's response, with full conversation context.
        """
        from saber.llm_engine import LLMEngine

        # Use default session if none specified
        if session_id is None:
            session_id = f"default-{self.meta.specialist_id}"

        # Create session if it doesn't exist yet
        if not self._session_memory.session_exists(session_id):
            self.start_session(session_id=session_id)

        model_path = self.meta.model_path or "Qwen/Qwen2.5-7B"

        with LLMEngine(model_path) as engine:
            response = engine.generate_from_session(
                session_memory=self._session_memory,
                session_id=session_id,
                user_message=message,
            )

        return response

    def get_session_history(self, session_id: Optional[str] = None) -> list:
        """Retrieve the conversation history for a session.

        Parameters
        ----------
        session_id : str or None
            If None, returns the default session history.

        Returns
        -------
        list[dict]
            List of {"role": ..., "content": ...} message dicts.
        """
        if session_id is None:
            session_id = f"default-{self.meta.specialist_id}"
        return self._session_memory.get_history(session_id)

    def get_turn_count(self, session_id: Optional[str] = None) -> int:
        """Return the number of user turns in a session."""
        if session_id is None:
            session_id = f"default-{self.meta.specialist_id}"
        return self._session_memory.get_turn_count(session_id)

    def clear_session(self, session_id: Optional[str] = None) -> None:
        """Clear a conversation session's history."""
        if session_id is None:
            session_id = f"default-{self.meta.specialist_id}"
        self._session_memory.clear_session(session_id)

    def clear_all_sessions(self) -> None:
        """Clear all conversation sessions for this specialist."""
        for sid in self._session_memory.list_sessions():
            self._session_memory.clear_session(sid)

    # ------------------------------------------------------------------
    # The Signal Interface
    # ------------------------------------------------------------------

    def handle_signal(self, signal: Signal) -> Signal:
        """Universal entry point for all incoming signals."""
        if signal.signal_type == SignalType.TASK_SIGNAL:
            return self._handle_task(signal)
        elif signal.signal_type == SignalType.VERIFICATION_SIGNAL:
            return self._handle_verification(signal)
        else:
            # Return an audit/error signal if unsupported
            return Signal(
                signal_type=SignalType.AUDIT_SIGNAL,
                query_id=signal.query_id,
                source_id=self.meta.specialist_id,
                target_id=signal.source_id,
                payload={"error": f"Unsupported signal type: {signal.signal_type}"}
            ).freeze_and_hash()

    def confirm_task(self, task_signal: Signal) -> Signal:
        """Query Confirmation Loop: Acknowledge task understanding."""
        # By default, confirm. Can be overridden for strict checking.
        return Signal(
            signal_type=SignalType.CONFIRMATION_SIGNAL,
            query_id=task_signal.query_id,
            source_id=self.meta.specialist_id,
            target_id=task_signal.source_id,
            payload={"status": "CONFIRMED"}
        ).freeze_and_hash()

    def _handle_task(self, task_signal: Signal) -> Signal:
        """Process the task and return a COT_SIGNAL."""
        task_objective = task_signal.payload.get("objective", "")
        query_id = task_signal.query_id
        
        self.cot.begin_chain(self.domain, query_id)
        claims = self.process_task(task_objective)
        
        # Populate CoT chain reasoning steps generically if in benchmark mode
        import os
        if os.getenv("SABER_BENCHMARK_MODE") == "1":
            raw_output = getattr(self, "_last_raw_response", None)
            if raw_output and "REASONING:" in raw_output:
                try:
                    reasoning_part = raw_output.split("REASONING:")[1].split("CLAIMS:")[0].strip()
                    import re
                    sentences = re.split(r'(?<=[.!?])\s+', reasoning_part)
                    for s in sentences:
                        s = s.strip()
                        if s:
                            self.cot.add_step(action="ANALYZE", content=s, confidence=0.9)
                except Exception:
                    pass
        
        if not self.cot._current_chain.is_complete:
            conclusion = claims[0].statement if claims else task_objective
            confidence = claims[0].confidence if claims else 0.5
            self.cot.conclude(conclusion, confidence)
            
        self.cot.cleanup()
        
        self._cached_response = {
            "claims": [c.model_dump() for c in claims],
            "cot_chain": self.cot.export_for_signal(),
            "raw_response": getattr(self, "_last_raw_response", None),
        }
        
        return Signal(
            signal_type=SignalType.COT_SIGNAL,
            query_id=query_id,
            source_id=self.meta.specialist_id,
            target_id=task_signal.source_id,
            payload=self._cached_response
        ).freeze_and_hash()

    def get_cached_response(self) -> dict:
        return self._cached_response or {}

    def clear_cache(self) -> None:
        self._cached_response = None

    def _handle_verification(self, verification_signal: Signal) -> Signal:
        """Check mode: Verify the Meta-Reasoning Layer's compiled text."""
        # Default verification returns GREEN_CHIT
        # Override this or use SENTINEL's LLM engine to perform strict verification
        return Signal(
            signal_type=SignalType.VERIFICATION_SIGNAL,
            query_id=verification_signal.query_id,
            source_id=self.meta.specialist_id,
            target_id=verification_signal.source_id,
            payload={"status": "GREEN_CHIT"}
        ).freeze_and_hash()

    # ------------------------------------------------------------------
    # Abstract Methods for Subclasses
    # ------------------------------------------------------------------

    def process_task(self, objective: str) -> list[Claim]:
        """Perform domain reasoning and return a list of Pydantic Claim objects.
        Must be implemented by subclasses.
        """
        raise NotImplementedError


class SpecialistLoader:
    """Utility class to dynamically discover and load Specialists."""

    @staticmethod
    def discover(package_name: str) -> list[Specialist]:
        """Discover all Specialist subclasses in a given package."""
        specialists = []
        try:
            package = importlib.import_module(package_name)
        except ImportError:
            return specialists

        if not hasattr(package, "__path__"):
            return specialists

        for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
            full_module_name = f"{package_name}.{module_name}"
            try:
                module = importlib.import_module(full_module_name)
            except ImportError:
                continue

            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Specialist) and obj is not Specialist:
                    try:
                        specialists.append(obj())
                    except Exception:
                        pass
        return specialists
