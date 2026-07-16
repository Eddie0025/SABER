# -*- coding: utf-8 -*-
"""saber.metrics

Selective Activation Tracking and Research Metrics for SABER v2.0.

Tracks:
- Active specialists per query
- Active parameters (estimated)
- Tokens consumed (estimated from character count)
- Inference cost (relative)
- Memory usage
- Latency breakdown (per-component)

This module provides the data needed to support the
"flagship performance with fewer active parameters" hypothesis.
"""

from __future__ import annotations

import time
import os
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Parameter estimates per model (for activation tracking)
# ---------------------------------------------------------------------------

_MODEL_PARAMS: Dict[str, int] = {
    "Qwen/Qwen2.5-3B": 3_000_000_000,
    "Qwen/Qwen2.5-7B": 7_000_000_000,
    "meta-llama/Llama-3.2-3B": 3_000_000_000,
    "mistralai/Mistral-7B-v0.3": 7_000_000_000,
}


@dataclass
class ActivationRecord:
    """Tracks resource usage for a single query through the SABER pipeline."""

    query_id: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Specialists
    specialists_activated: List[str] = field(default_factory=list)
    total_specialists_available: int = 0

    # Parameters
    models_used: List[str] = field(default_factory=list)
    active_parameters: int = 0          # sum of unique model params
    total_possible_parameters: int = 0  # if ALL specialists were active

    # Tokens (estimated from character counts)
    input_chars: int = 0
    output_chars: int = 0
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0

    # Latency
    total_latency_seconds: float = 0.0
    specialist_latency: Dict[str, float] = field(default_factory=dict)
    verification_latency: float = 0.0
    compilation_latency: float = 0.0

    # Memory (if available)
    peak_memory_mb: float = 0.0

    def compute_derived(self) -> None:
        """Compute derived metrics from raw counters."""
        # Rough token estimate: 1 token ≈ 4 characters
        self.estimated_input_tokens = self.input_chars // 4
        self.estimated_output_tokens = self.output_chars // 4

        # Active parameters from unique models
        unique_models = set(self.models_used)
        self.active_parameters = sum(
            _MODEL_PARAMS.get(m, 3_000_000_000) for m in unique_models
        )

    def efficiency_ratio(self) -> float:
        """Return active_params / total_possible_params.

        A lower ratio means SABER is being more selective.
        """
        if self.total_possible_parameters == 0:
            return 1.0
        return self.active_parameters / self.total_possible_parameters

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ActivationTracker:
    """Accumulates ActivationRecords and writes them for analysis."""

    def __init__(self, log_path: str = "logs/activation.jsonl") -> None:
        self._log_path = log_path
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    def record(self, activation: ActivationRecord) -> None:
        """Persist a single activation record."""
        activation.compute_derived()
        with open(self._log_path, "a", encoding="utf-8") as f:
            json.dump(activation.to_dict(), f, ensure_ascii=False, default=str)
            f.write("\n")

    def get_all_records(self) -> List[Dict[str, Any]]:
        """Load all activation records."""
        records = []
        if not os.path.isfile(self._log_path):
            return records
        with open(self._log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def compute_summary(self) -> Dict[str, Any]:
        """Compute aggregate activation metrics across all queries.

        Returns
        -------
        dict with:
            total_queries, avg_specialists_activated, avg_active_parameters,
            avg_efficiency_ratio, total_estimated_tokens, avg_latency_seconds.
        """
        records = self.get_all_records()
        if not records:
            return {"total_queries": 0}

        n = len(records)
        total_specs = sum(len(r.get("specialists_activated", [])) for r in records)
        total_params = sum(r.get("active_parameters", 0) for r in records)
        total_possible = sum(r.get("total_possible_parameters", 0) for r in records)
        total_tokens = sum(
            r.get("estimated_input_tokens", 0) + r.get("estimated_output_tokens", 0)
            for r in records
        )
        total_latency = sum(r.get("total_latency_seconds", 0) for r in records)

        avg_eff = (total_params / total_possible) if total_possible > 0 else 1.0

        return {
            "total_queries": n,
            "avg_specialists_activated": round(total_specs / n, 2),
            "avg_active_parameters": total_params // n,
            "avg_active_parameters_human": f"{total_params // n / 1e9:.1f}B",
            "avg_efficiency_ratio": round(avg_eff, 4),
            "total_estimated_tokens": total_tokens,
            "avg_latency_seconds": round(total_latency / n, 2),
        }
