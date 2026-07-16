# -*- coding: utf-8 -*-
"""saber.errors

Failure classification taxonomy for SABER.
Used to track and categorize systemic failures for metric analysis.
"""

from enum import Enum


class FailureCategory(str, Enum):
    """Categorisation of system failures."""
    ROUTING_FAILURE = "routing_failure"
    SPECIALIST_FAILURE = "specialist_failure"
    VERIFICATION_FAILURE = "verification_failure"
    CONSENSUS_FAILURE = "consensus_failure"
    KNOWLEDGE_FAILURE = "knowledge_failure"
    SYNTHESIS_FAILURE = "synthesis_failure"
    SYSTEM_FAILURE = "system_failure"


class SaberError(Exception):
    """Base exception for SABER system errors."""
    def __init__(self, message: str, category: FailureCategory):
        super().__init__(message)
        self.category = category
