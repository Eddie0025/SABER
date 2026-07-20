# -*- coding: utf-8 -*-
"""saber.specialists.architecture

Architecture domain specialist for secure system design.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from saber.signal import Claim, ClaimStatus
from saber.specialist import Specialist


class ArchitectureSpecialist(Specialist):
    """Specialist for system architecture, secure design, and cloud infrastructure."""

    @property
    def domain(self) -> str:
        return "architecture"

    @property
    def keywords(self) -> list[str]:
        return [
            "architecture", "design", "microservice", "monolith",
            "scalability", "load", "balancer", "database", "cache", "redis",
            "kubernetes", "docker", "cloud", "aws", "azure", "gcp",
            "gateway", "queue", "serverless", "infrastructure", "deployment",
            "devops", "pipeline", "container", "orchestration", "distributed",
            "network", "protocol", "latency", "throughput", "availability",
            "consistency", "partition", "replication", "sharding", "cdn",
        ]

    def __init__(self) -> None:
        super().__init__()
        self.meta.capabilities = [
            "system_design",
            "cloud_infrastructure",
            "secure_architecture",
            "threat_modeling",
            "scalability_analysis"
        ]
        self.meta.authority_score = 0.94

    def process_task(self, objective: str) -> List[Claim]:
        if self.meta.model_path:
            raw_output = self._infer(objective)
            self._last_raw_response = raw_output
            return self.parse_raw_output_to_claims(raw_output)
        else:
            return [Claim(
                statement=f"[Architecture Placeholder] Analysis of: {objective}",
                confidence=0.9,
                domain=self.domain,
                status=ClaimStatus.UNVERIFIED
            )]

    def _infer(self, query: str) -> str:
        from saber.llm_engine import LLMEngine
        try:
            with LLMEngine(self.meta.model_path) as engine:
                system_prompt = (
                    "You are a Software Architecture and Systems Design AI specialist. Do NOT output conversational text. "
                    "Output ONLY a valid JSON array of claims. "
                    "Example: [{\"text\": \"Deploying an API Gateway mitigates DDoS risks and centralizes auth.\", \"confidence\": 0.95}]"
                )
                return engine.generate(query, system_prompt=system_prompt)
        except Exception as e:
            print(f"[ArchitectureSpecialist] Inference failed: {e}")
            return f"[Architecture Error] {e}"
