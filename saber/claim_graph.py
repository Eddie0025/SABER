# -*- coding: utf-8 -*-
"""saber.claim_graph

Implements the Claim Graph — SABER's internal knowledge representation.

Nodes are individual claims; edges encode relationships:
    * ``supports``   — one claim reinforces another
    * ``contradicts`` — one claim conflicts with another
    * ``depends_on``  — one claim requires another to hold

The graph is the Meta-Reasoning Layer's working memory during compilation and is
used by the cross-domain consistency pass before final delivery.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class EdgeType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DEPENDS_ON = "depends_on"


@dataclass
class Claim:
    """A single claim node in the graph."""

    claim_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    domain: str = ""
    source_signal_id: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.claim_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Claim):
            return NotImplemented
        return self.claim_id == other.claim_id


@dataclass
class Edge:
    """A directed relationship between two claims."""

    source_id: str = ""
    target_id: str = ""
    edge_type: EdgeType = EdgeType.SUPPORTS
    weight: float = 1.0


class ClaimGraph:
    """In-memory directed graph of claims.

    The implementation is intentionally simple — it uses plain dicts
    rather than pulling in ``networkx`` so that the system has zero
    heavyweight dependencies by default.
    """

    def __init__(self) -> None:
        self._claims: Dict[str, Claim] = {}
        self._edges: List[Edge] = []
        # Adjacency maps for quick lookup
        self._outgoing: Dict[str, List[Edge]] = {}
        self._incoming: Dict[str, List[Edge]] = {}

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add_claim(self, claim: Claim) -> None:
        """Add a claim node to the graph."""
        self._claims[claim.claim_id] = claim
        self._outgoing.setdefault(claim.claim_id, [])
        self._incoming.setdefault(claim.claim_id, [])

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
    ) -> None:
        """Add a directed edge between two claims."""
        edge = Edge(source_id=source_id, target_id=target_id, edge_type=edge_type, weight=weight)
        self._edges.append(edge)
        self._outgoing.setdefault(source_id, []).append(edge)
        self._incoming.setdefault(target_id, []).append(edge)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        return self._claims.get(claim_id)

    def all_claims(self) -> List[Claim]:
        return list(self._claims.values())

    def get_supporters(self, claim_id: str) -> List[Claim]:
        """Return claims that *support* the given claim."""
        return [
            self._claims[e.source_id]
            for e in self._incoming.get(claim_id, [])
            if e.edge_type == EdgeType.SUPPORTS and e.source_id in self._claims
        ]

    def get_contradictions(self, claim_id: str) -> List[Claim]:
        """Return claims that *contradict* the given claim."""
        related: List[Claim] = []
        for e in self._incoming.get(claim_id, []):
            if e.edge_type == EdgeType.CONTRADICTS and e.source_id in self._claims:
                related.append(self._claims[e.source_id])
        for e in self._outgoing.get(claim_id, []):
            if e.edge_type == EdgeType.CONTRADICTS and e.target_id in self._claims:
                related.append(self._claims[e.target_id])
        return related

    def get_dependencies(self, claim_id: str) -> List[Claim]:
        """Return claims that the given claim *depends on*."""
        return [
            self._claims[e.target_id]
            for e in self._outgoing.get(claim_id, [])
            if e.edge_type == EdgeType.DEPENDS_ON and e.target_id in self._claims
        ]

    # ------------------------------------------------------------------
    # Consistency checking
    # ------------------------------------------------------------------

    def find_all_contradictions(self) -> List[Tuple[Claim, Claim]]:
        """Return every pair of contradicting claims in the graph."""
        pairs: List[Tuple[Claim, Claim]] = []
        seen: Set[Tuple[str, str]] = set()
        for edge in self._edges:
            if edge.edge_type != EdgeType.CONTRADICTS:
                continue
            key = tuple(sorted([edge.source_id, edge.target_id]))
            if key in seen:
                continue
            seen.add(key)
            src = self._claims.get(edge.source_id)
            tgt = self._claims.get(edge.target_id)
            if src and tgt:
                pairs.append((src, tgt))
        return pairs

    def cross_domain_consistency(self) -> Dict[str, List[Tuple[Claim, Claim]]]:
        """Group contradictions by domain pair.

        Returns a dict keyed by ``"domainA↔domainB"`` with a list of
        contradicting claim pairs.
        """
        result: Dict[str, List[Tuple[Claim, Claim]]] = {}
        for a, b in self.find_all_contradictions():
            key = "↔".join(sorted([a.domain, b.domain]))
            result.setdefault(key, []).append((a, b))
        return result

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claims": [
                {
                    "claim_id": c.claim_id,
                    "text": c.text,
                    "domain": c.domain,
                    "source_signal_id": c.source_signal_id,
                    "confidence": c.confidence,
                    "metadata": c.metadata,
                }
                for c in self._claims.values()
            ],
            "edges": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "edge_type": e.edge_type.value,
                    "weight": e.weight,
                }
                for e in self._edges
            ],
        }
