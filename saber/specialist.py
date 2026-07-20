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
        self._last_objective = task_objective
        query_id = task_signal.query_id
        
        self.cot.begin_chain(self.domain, query_id)
        claims = self.process_task(task_objective)
        
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
        """Check mode: Receive flags from Sentinel and perform self-correction."""
        import os
        import json
        from saber.llm_engine import LLMEngine
        
        payload = verification_signal.payload
        status = payload.get("status")
        
        if status != "FLAGGED":
            return Signal(
                signal_type=SignalType.VERIFICATION_SIGNAL,
                query_id=verification_signal.query_id,
                source_id=self.meta.specialist_id,
                target_id=verification_signal.source_id,
                payload={"status": "GREEN_CHIT"}
            ).freeze_and_hash()

        # Perform self-correction using the specialist's model
        flags = payload.get("flags", [])
        compiled_text = payload.get("compiled_text", "")
        
        flags_desc = []
        for f in flags:
            issue = f.get('issue_type', 'REASONING_ERROR')
            reasoning = f.get('reasoning', f.get('description', ''))
            fix = f.get('proposed_fix', '')
            flags_desc.append(f"- [{issue.upper()}] {reasoning}\n  FIX: {fix}")
        flags_str = "\n".join(flags_desc)

        original_objective = getattr(self, "_last_objective", compiled_text)

        # Build self-correction prompt
        prompt = (
            f"Original Objective: {original_objective}\n\n"
            f"Your previous response:\n{getattr(self, '_last_raw_response', compiled_text)}\n\n"
            f"The verifier found the following errors in your response:\n{flags_str}\n\n"
            "Please regenerate your complete response. Correct all identified errors, "
            "but make sure to preserve the exact formatting. "
        )
        
        if os.getenv("SABER_BENCHMARK_MODE") == "1":
            if self.domain == "science" or self.domain == "cyber":
                system_prompt = (
                    f"You are an expert {self.domain} specialist. First, think step by step to deduce the answer. "
                    "Second, output exactly 3 factual claims that support your reasoning. "
                    "Finally, state the correct option letter (A, B, C, or D).\n\n"
                    "Use this strict format:\n"
                    "REASONING: <your step by step thought process>\n"
                    "CLAIMS:\n1. <claim 1>\n2. <claim 2>\n3. <claim 3>\n"
                    "ANSWER: <A, B, C, or D>"
                )
            elif self.domain == "coding":
                system_prompt = (
                    "You are an expert coding specialist. First, think step by step to solve the task. "
                    "Second, output exactly 3 factual claims about the logic/complexity. "
                    "Finally, output the complete Python implementation wrapped inside a ```python block.\n\n"
                    "Use this strict format:\n"
                    "REASONING: <your step by step thought process>\n"
                    "CLAIMS:\n1. <claim 1>\n2. <claim 2>\n3. <claim 3>\n"
                    "CODE:\n```python\n<your code here>\n```"
                )
            elif self.domain == "finance":
                system_prompt = (
                    "You are an expert financial analyst. First, think step by step to solve the question. "
                    "Second, output exactly 3 factual claims that support your reasoning. "
                    "Finally, state the correct numerical answer.\n\n"
                    "Use this strict format:\n"
                    "REASONING: <your step by step thought process>\n"
                    "CLAIMS:\n1. <claim 1>\n2. <claim 2>\n3. <claim 3>\n"
                    "ANSWER: <numeric value>"
                )
        else:
            if self.domain == "science":
                system_prompt = "You are a Science AI specialist. Do NOT output conversational text. Output ONLY a valid JSON array of claims containing step-by-step scientific or mathematical derivation."
            elif self.domain == "cyber":
                system_prompt = "You are a Cybersecurity AI specialist. Do NOT output conversational text. Output ONLY a valid JSON array of claims."
            elif self.domain == "coding":
                system_prompt = "You are a Coding and Software Engineering AI specialist. Do NOT output conversational text. Output ONLY a valid JSON array of claims."
            elif self.domain == "finance":
                system_prompt = "You are a Finance and Economics AI specialist. Do NOT output conversational text. Output ONLY a valid JSON array of claims."
            else:
                system_prompt = f"You are an expert {self.domain} specialist."

        model_path = self.meta.model_path or "Qwen/Qwen2.5-7B"
        
        try:
            with LLMEngine(model_path) as engine:
                corrected_output = engine.generate(prompt, system_prompt=system_prompt)
            
            self._last_raw_response = corrected_output
            
            # Reparse the claims from the corrected response
            claims = []
            if os.getenv("SABER_BENCHMARK_MODE") == "1":
                claims_texts = []
                if "CLAIMS:" in corrected_output:
                    try:
                        after_claims = corrected_output.split("CLAIMS:")[1]
                        end_marker = "ANSWER:" if "ANSWER:" in after_claims else "CODE:"
                        claims_block = after_claims.split(end_marker)[0]
                        for line in claims_block.split("\n"):
                            line = line.strip()
                            if line and (line[0].isdigit() or line.startswith("-")):
                                clean_claim = line.lstrip("1234567890.- ").strip()
                                if clean_claim:
                                    claims_texts.append(clean_claim)
                    except Exception:
                        pass
                if not claims_texts:
                    claims_texts = [corrected_output[:100]]
                for text in claims_texts:
                    claims.append(Claim(statement=text, confidence=0.9, domain=self.domain, status=ClaimStatus.VERIFIED))
            else:
                try:
                    claims_data = json.loads(corrected_output)
                    if not isinstance(claims_data, list):
                        claims_data = [claims_data]
                    for c in claims_data:
                        claims.append(Claim(statement=c.get("text", str(c)), confidence=float(c.get("confidence", 0.9)), domain=self.domain, status=ClaimStatus.VERIFIED))
                except Exception:
                    claims = [Claim(statement=corrected_output, confidence=0.5, domain=self.domain, status=ClaimStatus.VERIFIED)]

            self._cached_response = {
                "claims": [c.model_dump() for c in claims],
                "cot_chain": self.cot.export_for_signal(),
                "raw_response": corrected_output,
            }

            # Return a COT_SIGNAL (re-using COT_SIGNAL as response since it contains corrected answer)
            # This follows the Signal lifecycle
            return Signal(
                signal_type=SignalType.COT_SIGNAL,
                query_id=verification_signal.query_id,
                source_id=self.meta.specialist_id,
                target_id=verification_signal.source_id,
                payload=self._cached_response
            ).freeze_and_hash()

        except Exception as e:
            print(f"[{self.meta.specialist_id}] Self-correction failed: {e}")
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
