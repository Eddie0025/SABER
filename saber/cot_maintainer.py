import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
import numpy as np

class ReasoningStep(BaseModel):
    step_number: int
    action: str  # IDENTIFY | ANALYZE | HYPOTHESIZE | EVIDENCE | EVALUATE | CONCLUDE
    content: str
    confidence: float
    evidence_refs: List[str] = Field(default_factory=list)
    depends_on: List[int] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

class CoTChain(BaseModel):
    domain: str
    query_id: str
    steps: List[ReasoningStep] = Field(default_factory=list)
    is_concluded: bool = False

class CoTMaintainer:
    """
    Passive scratchpad and buffer for multi-step reasoning.
    Stores model output and re-injects it on subsequent steps without LLM intervention.
    """
    def __init__(self):
        self.active_chains: Dict[str, CoTChain] = {}
        
    def begin_chain(self, domain: str, query_id: str) -> str:
        """Starts a new reasoning chain for a task."""
        chain = CoTChain(domain=domain, query_id=query_id)
        chain_id = f"cot_{uuid.uuid4().hex[:8]}"
        self.active_chains[chain_id] = chain
        return chain_id
        
    def add_step(self, chain_id: str, action: str, content: str, confidence: float = 1.0, 
                 evidence_refs: List[str] = None, depends_on: List[int] = None) -> int:
        """Stores one reasoning step written by the model."""
        if chain_id not in self.active_chains:
            raise ValueError(f"Chain {chain_id} not found.")
            
        chain = self.active_chains[chain_id]
        if chain.is_concluded:
            raise RuntimeError(f"Chain {chain_id} is already concluded.")
            
        step_number = len(chain.steps) + 1
        step = ReasoningStep(
            step_number=step_number,
            action=action,
            content=content,
            confidence=confidence,
            evidence_refs=evidence_refs or [],
            depends_on=depends_on or []
        )
        chain.steps.append(step)
        return step_number
        
    def read_summary(self, chain_id: str) -> str:
        """Formats all stored steps as text for re-injection into the next prompt."""
        if chain_id not in self.active_chains:
            return ""
            
        chain = self.active_chains[chain_id]
        if not chain.steps:
            return ""
            
        summary = "PRIOR REASONING STEPS:\n"
        for step in chain.steps:
            summary += f"[{step.step_number}] {step.action}: {step.content}\n"
        return summary
        
    def conclude(self, chain_id: str, conclusion: str, confidence: float = 1.0):
        """Marks the chain complete and adds a CONCLUDE step."""
        self.add_step(chain_id, "CONCLUDE", conclusion, confidence)
        self.active_chains[chain_id].is_concluded = True
        
    def cleanup(self, chain_id: str):
        """
        Deduplicates and merges redundant steps (similarity > 0.85).
        For this implementation, it's a structural placeholder for actual semantic dedup.
        """
        # Semantic dedup logic goes here using sentence transformers
        pass
        
    def export_for_signal(self, chain_id: str) -> Dict[str, Any]:
        """Exports chain as a dictionary payload for Sentinel verification."""
        if chain_id not in self.active_chains:
            raise ValueError(f"Chain {chain_id} not found.")
        return self.active_chains[chain_id].model_dump()
        
    def reset(self, chain_id: str):
        """Archives the current chain and resets for the next task."""
        if chain_id in self.active_chains:
            # Here we would archive to disk
            del self.active_chains[chain_id]
