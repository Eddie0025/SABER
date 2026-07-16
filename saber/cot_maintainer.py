# -*- coding: utf-8 -*-
"""saber.cot_maintainer

Bidirectional working memory module that specialists read from and write to during processing.
Enables step-by-step reasoning chains (Chain of Thought).
"""

from __future__ import annotations

import difflib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ReasoningStep(BaseModel):
    step_number: int
    action: str        # IDENTIFY | ANALYZE | HYPOTHESIZE | EVIDENCE | EVALUATE | CONCLUDE
    content: str
    confidence: float
    evidence_refs: List[str] = Field(default_factory=list)
    depends_on: List[int] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CoTChain(BaseModel):
    domain: str
    query_id: str
    steps: List[ReasoningStep] = Field(default_factory=list)
    final_conclusion: str = ""
    chain_confidence: float = 0.0
    is_complete: bool = False
    cleanup_applied: bool = False


class CoTMaintainer:
    def __init__(self):
        self._current_chain: Optional[CoTChain] = None
        self._completed_chains: List[CoTChain] = []

    def begin_chain(self, domain: str, query_id: str) -> None:
        """Start a new reasoning chain for a task."""
        self._current_chain = CoTChain(domain=domain, query_id=query_id)

    def add_step(
        self,
        action: str,
        content: str,
        confidence: float,
        evidence_refs: List[str] = None,
        depends_on: List[int] = None
    ) -> int:
        """Add a step to the current reasoning chain. Returns the step number."""
        if not self._current_chain:
            raise ValueError("No active CoT chain. Call begin_chain() first.")
        
        step_num = len(self._current_chain.steps) + 1
        step = ReasoningStep(
            step_number=step_num,
            action=action,
            content=content.strip(),
            confidence=confidence,
            evidence_refs=evidence_refs or [],
            depends_on=depends_on or []
        )
        self._current_chain.steps.append(step)
        return step_num

    def read_steps(self) -> List[ReasoningStep]:
        """Get the steps of the current chain."""
        if not self._current_chain:
            return []
        return self._current_chain.steps

    def read_summary(self) -> str:
        """Get a formatted text summary for LLM prompt injection."""
        if not self._current_chain or not self._current_chain.steps:
            return ""
            
        summary = "Previous reasoning steps:\n"
        for step in self._current_chain.steps:
            summary += f"  Step {step.step_number} [{step.action}]: {step.content} (confidence: {step.confidence:.2f})\n"
            
        summary += "\nBased on the above, determine the next reasoning step."
        return summary

    def conclude(self, conclusion: str, confidence: float) -> None:
        """Mark the chain as complete and record final conclusion."""
        if not self._current_chain:
            return
        self._current_chain.final_conclusion = conclusion.strip()
        self._current_chain.chain_confidence = confidence
        self._current_chain.is_complete = True
        
        # Add a CONCLUDE step if one isn't already the last step
        if not self._current_chain.steps or self._current_chain.steps[-1].action != "CONCLUDE":
            self.add_step(
                action="CONCLUDE",
                content=self._current_chain.final_conclusion,
                confidence=confidence,
                depends_on=[s.step_number for s in self._current_chain.steps]
            )

    def cleanup(self) -> None:
        """Cleanup redundancies in the current chain (dedup, loop detection, consolidation)."""
        if not self._current_chain or not self._current_chain.steps:
            return
            
        cleaned_steps = []
        for i, step in enumerate(self._current_chain.steps):
            if len(step.content) < 20 and step.action != "CONCLUDE":
                continue  # skip too short
                
            is_redundant = False
            for prev_step in cleaned_steps:
                # Text dedup and loop detection
                similarity = difflib.SequenceMatcher(None, step.content, prev_step.content).ratio()
                if similarity > 0.85:
                    is_redundant = True
                    # Keep higher confidence
                    if step.confidence > prev_step.confidence:
                        prev_step.content = step.content
                        prev_step.confidence = step.confidence
                        prev_step.action = step.action
                    break
                    
            if not is_redundant:
                # Merge consecutive same-action
                if cleaned_steps and cleaned_steps[-1].action == step.action and step.action not in ("IDENTIFY", "CONCLUDE"):
                    cleaned_steps[-1].content += f" {step.content}"
                    cleaned_steps[-1].confidence = (cleaned_steps[-1].confidence + step.confidence) / 2
                else:
                    # Update step number since we might have removed some
                    step.step_number = len(cleaned_steps) + 1
                    cleaned_steps.append(step)
                    
        self._current_chain.steps = cleaned_steps
        self._current_chain.cleanup_applied = True

    def get_chain(self) -> Optional[CoTChain]:
        return self._current_chain

    def reset(self) -> None:
        """Archive the current chain and reset for next task."""
        if self._current_chain:
            self._completed_chains.append(self._current_chain)
            self._current_chain = None

    def get_all_chains(self) -> List[CoTChain]:
        return self._completed_chains
        
    def export_for_signal(self) -> Dict[str, Any]:
        """Export current chain as a dictionary payload for signals."""
        if not self._current_chain:
            return {}
        return self._current_chain.model_dump()
