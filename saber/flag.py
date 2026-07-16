# -*- coding: utf-8 -*-
"""saber.flag

Defines the Flag dataclass — a structured correction object generated
by SENTINEL when a verification check detects an issue.

Flags are machine-readable, never free-form NLP.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class Severity(str, Enum):
    """Flag severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IssueType(str, Enum):
    """Categorisation of detected issues."""
    FACTUAL_ERROR = "factual_error"
    REASONING_ERROR = "reasoning_error"
    LOGIC_GAP = "logic_gap"
    MISSING_EVIDENCE = "missing_evidence"
    DOMAIN_CONFLICT = "domain_conflict"
    CALCULATION_ERROR = "calculation_error"
    SECURITY_ASSUMPTION_ERROR = "security_assumption_error"
    DIAGNOSTIC_INCONSISTENCY = "diagnostic_inconsistency"
    FINANCIAL_ANALYSIS_ERROR = "financial_analysis_error"
    INTEGRITY_FAILURE = "integrity_failure"


@dataclass
class Flag:
    """A structured correction / warning object.

    Attributes
    ----------
    flag_id : str
        Unique identifier.
    originating_specialist : str
        The specialist (or component) that raised the flag.
    severity : Severity
        How urgent the issue is.
    issue_type : IssueType
        The category of the detected issue.
    confidence : float
        Confidence in the flag's correctness (0.0 to 1.0).
    evidence : str
        Evidence supporting the flag (e.g. quote from text or external knowledge).
    reasoning : str
        The chain of thought explaining why the flag was raised.
    target_claim : str
        The claim ID (or signal ID) the flag refers to.
    proposed_fix : str
        Machine-readable description of the proposed correction.
    description : str
        Human-readable explanation of the issue.
    resolved : bool
        Whether the flag has been addressed by a patch cycle.
    resolution_notes : str
        Notes on how the flag was resolved.
    timestamp : str
        ISO-8601 creation timestamp.
    """

    flag_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    originating_specialist: str = ""
    severity: Severity = Severity.MEDIUM
    issue_type: IssueType = IssueType.REASONING_ERROR
    confidence: float = 0.0
    evidence: str = ""
    reasoning: str = ""
    target_claim: str = ""
    proposed_fix: str = ""
    description: str = ""
    resolved: bool = False
    resolution_notes: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def resolve(self, notes: str = "") -> None:
        """Mark this flag as resolved."""
        self.resolved = True
        self.resolution_notes = notes

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        d["issue_type"] = self.issue_type.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Flag":
        return cls(
            flag_id=data.get("flag_id", str(uuid.uuid4())),
            originating_specialist=data.get("originating_specialist", ""),
            severity=Severity(data.get("severity", "medium")),
            issue_type=IssueType(data.get("issue_type", "reasoning_error")),
            confidence=float(data.get("confidence", 0.0)),
            evidence=data.get("evidence", ""),
            reasoning=data.get("reasoning", ""),
            target_claim=data.get("target_claim", ""),
            proposed_fix=data.get("proposed_fix", ""),
            description=data.get("description", ""),
            resolved=data.get("resolved", False),
            resolution_notes=data.get("resolution_notes", ""),
            timestamp=data.get("timestamp", ""),
        )
