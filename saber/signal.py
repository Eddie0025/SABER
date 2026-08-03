import hashlib
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class BaseSignal(BaseModel):
    """
    Base class for all SABER signals. 
    Provides cryptographic hashing to ensure signal integrity across the pipeline.
    """
    signal_id: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    integrity_hash: Optional[str] = None

    def freeze_and_hash(self) -> str:
        """
        Computes a SHA-256 hash of the signal's payload (excluding the hash field itself)
        and freezes it to prevent undetected tampering between orchestrator/sentinel bounds.
        """
        payload = self.model_dump(exclude={"integrity_hash"})
        # Sort keys to ensure deterministic hashing
        payload_str = json.dumps(payload, sort_keys=True, default=str)
        self.integrity_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
        return self.integrity_hash

class QUERY_SIGNAL(BaseSignal):
    user_input: str
    is_casual_chat: bool = False
    ambiguity_score: float = 0.0

class TASK_SIGNAL(BaseSignal):
    query_id: str
    domain: str
    requires_coding: bool = False
    context: str

class CONFIRMATION_SIGNAL(BaseSignal):
    task_id: str
    specialist_id: str
    status: str = "RECEIVED"

class COT_SIGNAL(BaseSignal):
    task_id: str
    specialist_id: str
    reasoning_chain: List[Dict[str, Any]]
    extracted_claims: List[str]
    raw_output: str

class VERIFICATION_SIGNAL(BaseSignal):
    cot_id: str
    status: str = "GREEN_CHIT"
    kb_passages_used: int

class FLAG_SIGNAL(BaseSignal):
    cot_id: str
    issue_type: str  # FACTUAL_ERROR | REASONING_ERROR
    reasoning: str
    proposed_fix: str

class OUTPUT_SIGNAL(BaseSignal):
    query_id: str
    final_response: str
    footer: str

class AUDIT_SIGNAL(BaseSignal):
    event_type: str
    details: Dict[str, Any]

# --- Coding Sector Signals ---

class CODE_PLAN_SIGNAL(BaseSignal):
    task_id: str
    plan_details: Dict[str, Any]
    subtasks: List[Dict[str, Any]]

class CODE_DISPATCH_SIGNAL(BaseSignal):
    plan_id: str
    subtask_id: str
    language: str

class CODE_BLOCK_SIGNAL(BaseSignal):
    subtask_id: str
    specialist: str
    code: str
    tests: str

class CODE_SENTINEL_VERDICT(BaseSignal):
    block_id: str
    verdict: str  # CONFIRMED | FLAG
    tests_passed: int
    tests_total: int
    failure_context: Optional[str] = None

class CODE_REWRITE_SIGNAL(BaseSignal):
    block_id: str
    failure_context: str
    retry_count: int
