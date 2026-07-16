# -*- coding: utf-8 -*-
"""saber.signal

Strict Signal Schema definitions for the SABER architecture.
Uses Pydantic for validation and serialization.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SignalType(str, Enum):
    """The types of signals exchanged in the SABER lifecycle."""
    QUERY_SIGNAL = "QUERY_SIGNAL"
    TASK_SIGNAL = "TASK_SIGNAL"
    CONFIRMATION_SIGNAL = "CONFIRMATION_SIGNAL"
    OUTPUT_SIGNAL = "OUTPUT_SIGNAL"
    VERIFICATION_SIGNAL = "VERIFICATION_SIGNAL"
    FLAG_SIGNAL = "FLAG_SIGNAL"
    PATCH_SIGNAL = "PATCH_SIGNAL"
    HEALTH_SIGNAL = "HEALTH_SIGNAL"
    AUDIT_SIGNAL = "AUDIT_SIGNAL"
    COT_SIGNAL = "COT_SIGNAL"


class ClaimStatus(str, Enum):
    """The lifecycle status of a specific claim."""
    UNVERIFIED = "UNVERIFIED"
    VERIFIED = "VERIFIED"
    FLAGGED = "FLAGGED"
    REJECTED = "REJECTED"


class Claim(BaseModel):
    """The fundamental unit of knowledge in SABER."""
    claim_id: str = Field(default_factory=lambda: f"C-{uuid.uuid4().hex[:8].upper()}")
    statement: str
    confidence: float
    evidence_refs: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    domain: str
    status: ClaimStatus = ClaimStatus.UNVERIFIED


class Signal(BaseModel):
    """The immutable structured message used for all inter-component communication."""
    signal_id: str = Field(default_factory=lambda: f"S-{uuid.uuid4().hex[:8].upper()}")
    signal_type: SignalType
    query_id: str
    source_id: str
    target_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = "2.0.0"
    priority: int = 1
    confidence: float = 0.0
    integrity_hash: str = ""
    payload: Dict[str, Any] = Field(default_factory=dict)

    def freeze_and_hash(self) -> "Signal":
        """Compute the SHA-256 hash over the payload and set it."""
        # We sort keys to ensure deterministic hashing
        raw = json.dumps(self.payload, sort_keys=True, default=str).encode("utf-8")
        self.integrity_hash = hashlib.sha256(raw).hexdigest()
        return self

    def verify_integrity(self) -> bool:
        """Return True if the stored hash matches a freshly computed one."""
        raw = json.dumps(self.payload, sort_keys=True, default=str).encode("utf-8")
        return self.integrity_hash == hashlib.sha256(raw).hexdigest()
