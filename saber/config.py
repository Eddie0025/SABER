# -*- coding: utf-8 -*-
"""saber.config

Global configuration for the SABER system.

All settings have sensible defaults and can be overridden via
environment variables or by passing a dict to ``SaberConfig.from_dict()``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Any, List


class VerificationTier(IntEnum):
    """User-selectable verification depth.

    Tier 0 — no verification (maximum speed)
    Tier 1 — standard verification (2 cycles)
    """
    TIER_0 = 0
    TIER_1 = 1

    @property
    def max_cycles(self) -> int:
        return {0: 0, 1: 2}[self.value]


# Specialist activation threshold (relevance score)
DEFAULT_ACTIVATION_THRESHOLD: float = 0.30

# Default ambiguity threshold — queries above this score trigger
# a clarification request before entering the reasoning pipeline.
DEFAULT_AMBIGUITY_THRESHOLD: float = 0.70


@dataclass
class SaberConfig:
    """Central configuration object.

    Attributes
    ----------
    verification_tier : VerificationTier
        Default verification depth for new queries.
    activation_threshold : float
        Minimum relevance score for a specialist to participate.
    ambiguity_threshold : float
        Ambiguity score above which the system asks for clarification.
    specialist_dirs : list[str]
        Directories to scan for specialist plug-in modules.
    model_dir : str
        Path to the directory storing fine-tuned model checkpoints.
    data_dir : str
        Path to the directory storing processed training data.
    db_path : str
        Path to the SQLite database used by the Specialist Registry.
    audit_log_path : str
        Path to the audit log JSONL file.
    max_compilation_retries : int
        Maximum number of patch-recompile cycles before giving up.
    concurrency : str
        Concurrency model: "asyncio" | "threading" | "sequential".
    """

    verification_tier: VerificationTier = VerificationTier.TIER_1
    activation_threshold: float = DEFAULT_ACTIVATION_THRESHOLD
    ambiguity_threshold: float = DEFAULT_AMBIGUITY_THRESHOLD
    specialist_dirs: List[str] = field(
        default_factory=lambda: ["saber/specialists"]
    )
    base_model: str = "Qwen/Qwen2.5-7B"
    model_dir: str = "models"
    data_dir: str = "data/processed"
    db_path: str = "saber_registry.db"
    audit_log_path: str = "logs/audit.jsonl"
    max_compilation_retries: int = 6
    concurrency: str = "sequential"

    # ------------------------------------------------------------------
    # Overrides from environment / dict
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "SaberConfig":
        """Create a config populated from environment variables."""
        tier = int(os.getenv("SABER_TIER", "2"))
        return cls(
            verification_tier=VerificationTier(tier),
            activation_threshold=float(
                os.getenv("SABER_ACTIVATION_THRESHOLD", str(DEFAULT_ACTIVATION_THRESHOLD))
            ),
            ambiguity_threshold=float(
                os.getenv("SABER_AMBIGUITY_THRESHOLD", str(DEFAULT_AMBIGUITY_THRESHOLD))
            ),
            base_model=os.getenv("SABER_BASE_MODEL", "Qwen/Qwen2.5-7B"),
            model_dir=os.getenv("SABER_MODEL_DIR", "models"),
            data_dir=os.getenv("SABER_DATA_DIR", "data/processed"),
            db_path=os.getenv("SABER_DB_PATH", "saber_registry.db"),
            audit_log_path=os.getenv("SABER_AUDIT_LOG", "logs/audit.jsonl"),
            concurrency=os.getenv("SABER_CONCURRENCY", "sequential"),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SaberConfig":
        tier = data.get("verification_tier", 2)
        if isinstance(tier, int):
            tier = VerificationTier(tier)
        return cls(
            verification_tier=tier,
            activation_threshold=data.get(
                "activation_threshold", DEFAULT_ACTIVATION_THRESHOLD
            ),
            ambiguity_threshold=data.get(
                "ambiguity_threshold", DEFAULT_AMBIGUITY_THRESHOLD
            ),
            specialist_dirs=data.get("specialist_dirs", ["saber/specialists"]),
            base_model=data.get("base_model", "Qwen/Qwen2.5-7B"),
            model_dir=data.get("model_dir", "models"),
            data_dir=data.get("data_dir", "data/processed"),
            db_path=data.get("db_path", "saber_registry.db"),
            audit_log_path=data.get("audit_log_path", "logs/audit.jsonl"),
            max_compilation_retries=data.get("max_compilation_retries", 6),
            concurrency=data.get("concurrency", "sequential"),
        )
