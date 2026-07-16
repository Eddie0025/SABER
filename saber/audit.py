# -*- coding: utf-8 -*-
"""saber.audit

Centralised audit logging for the SABER system.

SABER v2.0 — Now supports:
- Structured Decision Ledger entries (one per query)
- Verification effectiveness metrics
- Failure classification logging
- Complete reasoning audit trails

Every action — query receipt, signal exchange, flag generation,
verification cycles, and final output — is logged as a JSON-Lines
record keyed by ``query_id`` for replay and forensic analysis.
"""

from __future__ import annotations

import json
import os
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class AuditLogger:
    """Thread-safe, append-only audit log writer.

    Parameters
    ----------
    log_path : str
        File path for the JSONL audit log.
    """

    def __init__(self, log_path: str = "logs/audit.jsonl") -> None:
        self._log_path = log_path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    # ------------------------------------------------------------------
    # Core log writer
    # ------------------------------------------------------------------

    def log(
        self,
        event_type: str,
        query_id: str,
        data: Optional[Dict[str, Any]] = None,
        *,
        component: str = "",
    ) -> None:
        """Append an audit record.

        Parameters
        ----------
        event_type : str
            E.g. ``"query_received"``, ``"signal_sent"``, ``"flag_raised"``,
            ``"verification_pass"``, ``"compilation"``, ``"output_sent"``,
            ``"failure"``, ``"ledger_entry"``, ``"disagreement_detected"``.
        query_id : str
            Links all records belonging to the same user query.
        data : dict, optional
            Arbitrary payload.
        component : str
            The SABER component that produced the event.
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "query_id": query_id,
            "component": component,
            "data": data or {},
        }
        with self._lock:
            with open(self._log_path, "a", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, default=str)
                f.write("\n")

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def log_query(self, query_id: str, query_text: str) -> None:
        self.log("query_received", query_id, {"query": query_text}, component="orchestrator")

    def log_signal(self, query_id: str, signal_dict: Dict[str, Any]) -> None:
        self.log("signal_sent", query_id, signal_dict, component="specialist")

    def log_flag(self, query_id: str, flag_dict: Dict[str, Any]) -> None:
        self.log("flag_raised", query_id, flag_dict, component="sentinel")

    def log_verification(self, query_id: str, cycle: int, passed: bool) -> None:
        self.log(
            "verification_pass",
            query_id,
            {"cycle": cycle, "passed": passed},
            component="sentinel",
        )

    def log_compilation(self, query_id: str, compiled_text: str) -> None:
        self.log("compilation", query_id, {"text": compiled_text}, component="meta_reasoner")

    def log_output(self, query_id: str, output: str) -> None:
        self.log("output_sent", query_id, {"output": output}, component="orchestrator")

    # ------------------------------------------------------------------
    # Decision Ledger v2
    # ------------------------------------------------------------------

    def log_ledger(self, query_id: str, ledger: Dict[str, Any]) -> None:
        """Write a complete Decision Ledger entry for a query.

        The ledger contains:
        - Query text and ID
        - Selected specialists
        - Initial responses from each specialist
        - All flags raised during verification
        - All corrections applied
        - Full verification history (per-cycle)
        - Specialist disagreements
        - Final resolution text
        - Final confidence score
        """
        self.log("ledger_entry", query_id, ledger, component="meta_reasoner")

    # ------------------------------------------------------------------
    # Replay / query
    # ------------------------------------------------------------------

    def get_records(self, query_id: Optional[str] = None) -> list:
        """Read back all audit records, optionally filtered by query_id."""
        records = []
        if not os.path.isfile(self._log_path):
            return records
        with open(self._log_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if query_id is None or rec.get("query_id") == query_id:
                    records.append(rec)
        return records

    def get_ledger(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the structured Decision Ledger for a specific query."""
        records = self.get_records(query_id)
        for r in records:
            if r.get("event_type") == "ledger_entry":
                return r.get("data", {})
        return None

    # ------------------------------------------------------------------
    # Verification Effectiveness Metrics
    # ------------------------------------------------------------------

    def compute_verification_metrics(self) -> Dict[str, Any]:
        """Compute aggregate verification effectiveness metrics.

        Returns
        -------
        dict with keys:
            total_queries, total_flags_raised, total_flags_resolved,
            false_flags (flags that didn't lead to corrections),
            correction_success_rate, flags_by_category, flags_by_specialist,
            failure_distribution.
        """
        records = self.get_records()

        total_queries = 0
        total_flags = 0
        flags_by_category: Counter = Counter()
        flags_by_specialist: Counter = Counter()
        failure_distribution: Counter = Counter()
        verification_passes = 0
        verification_failures = 0

        for r in records:
            evt = r.get("event_type", "")
            data = r.get("data", {})

            if evt == "query_received":
                total_queries += 1

            elif evt == "flag_raised":
                total_flags += 1
                cat = data.get("issue_type", "unknown")
                flags_by_category[cat] += 1
                src = data.get("originating_specialist", r.get("component", "unknown"))
                flags_by_specialist[src] += 1

            elif evt == "verification_pass":
                if data.get("passed"):
                    verification_passes += 1
                else:
                    verification_failures += 1

            elif evt == "failure":
                cat = data.get("category", "unknown")
                failure_distribution[cat] += 1

        # Ledger-based metrics
        total_flags_resolved = 0
        total_corrections = 0
        ledger_entries = [r for r in records if r.get("event_type") == "ledger_entry"]
        for le in ledger_entries:
            ld = le.get("data", {})
            total_flags_resolved += sum(
                c.get("flags_addressed", 0) for c in ld.get("corrections", [])
            )
            total_corrections += len(ld.get("corrections", []))

        correction_rate = (total_flags_resolved / total_flags * 100) if total_flags > 0 else 0.0

        return {
            "total_queries": total_queries,
            "total_flags_raised": total_flags,
            "total_flags_resolved": total_flags_resolved,
            "false_flags": max(0, total_flags - total_flags_resolved),
            "correction_success_rate": round(correction_rate, 2),
            "verification_passes": verification_passes,
            "verification_failures": verification_failures,
            "flags_by_category": dict(flags_by_category),
            "flags_by_specialist": dict(flags_by_specialist),
            "failure_distribution": dict(failure_distribution),
            "total_corrections": total_corrections,
        }

    # ------------------------------------------------------------------
    # Failure Analysis
    # ------------------------------------------------------------------

    def get_failure_distribution(self) -> Dict[str, int]:
        """Return a breakdown of failure categories across all queries."""
        records = self.get_records()
        dist: Counter = Counter()
        for r in records:
            if r.get("event_type") == "failure":
                cat = r.get("data", {}).get("category", "unknown")
                dist[cat] += 1
        return dict(dist)
