# -*- coding: utf-8 -*-
"""saber.registry

Specialist Registry — a persistent hash-table of all registered
specialists, their metadata, and health states.

Specialists are **updated**, never retired.  The registry tracks
versions, authority scores, and health so the Orchestrator can make
informed routing decisions.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Dict, List, Optional

from saber.specialist import HealthStatus, Specialist, SpecialistLoader, SpecialistMeta


class SpecialistRegistry:
    """In-memory registry backed by an optional JSON file.

    Parameters
    ----------
    persist_path : str or None
        If set, the registry state is loaded from / saved to this file.
    """

    def __init__(self, persist_path: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self._persist_path = persist_path
        # domain → Specialist instance
        self._specialists: Dict[str, Specialist] = {}
        # domain → SpecialistMeta (for serialisation)
        self._meta: Dict[str, dict] = {}

        if persist_path and os.path.isfile(persist_path):
            self._load()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, specialist: Specialist) -> None:
        """Register (or update) a specialist."""
        with self._lock:
            self._specialists[specialist.domain] = specialist
            self._meta[specialist.domain] = specialist.meta.__dict__.copy()
            self._save()

    def unregister(self, domain: str) -> None:
        """Remove a specialist by domain."""
        with self._lock:
            self._specialists.pop(domain, None)
            self._meta.pop(domain, None)
            self._save()

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    def auto_discover(self, package_name: str = "saber.specialists") -> int:
        """Discover specialist subclasses and register them.

        Returns the number of specialists registered.
        """
        specialists = SpecialistLoader.discover(package_name)
        for s in specialists:
            self.register(s)
        return len(specialists)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, domain: str) -> Optional[Specialist]:
        """Return the specialist for *domain*, or None."""
        return self._specialists.get(domain)

    def get_online(self) -> List[Specialist]:
        """Return all specialists with health ONLINE or BUSY."""
        return [
            s for s in self._specialists.values()
            if s.meta.health in (HealthStatus.ONLINE, HealthStatus.BUSY)
        ]

    def list_domains(self) -> List[str]:
        return list(self._specialists.keys())

    def all(self) -> Dict[str, Specialist]:
        return dict(self._specialists)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def update_health(self, domain: str, status: HealthStatus) -> None:
        spec = self._specialists.get(domain)
        if spec:
            spec.meta.health = status
            self._meta[domain] = spec.meta.__dict__.copy()
            self._save()

    # ------------------------------------------------------------------
    # Persistence (simple JSON)
    # ------------------------------------------------------------------

    def _save(self) -> None:
        if not self._persist_path:
            return
        os.makedirs(os.path.dirname(self._persist_path) or ".", exist_ok=True)
        with open(self._persist_path, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, indent=2, default=str)

    def _load(self) -> None:
        if not self._persist_path or not os.path.isfile(self._persist_path):
            return
        with open(self._persist_path, "r", encoding="utf-8") as f:
            self._meta = json.load(f)

    def to_dict(self) -> Dict[str, dict]:
        return {domain: meta.copy() for domain, meta in self._meta.items()}
