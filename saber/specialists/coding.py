# -*- coding: utf-8 -*-
"""saber.specialists.coding

Coding domain specialist.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from saber.signal import Claim, ClaimStatus
from saber.specialist import Specialist


class CodingSpecialist(Specialist):
    """Specialist for coding and software engineering domain reasoning."""

    @property
    def domain(self) -> str:
        return "coding"

    @property
    def keywords(self) -> list[str]:
        return [
            "code", "coding", "program", "programming", "function", "algorithm",
            "python", "javascript", "java", "debug", "error", "exception",
            "api", "git", "compile", "runtime", "syntax", "loop", "recursion",
            "array", "sort", "software", "developer", "engineering", "refactor",
            "class", "object", "method", "variable", "library", "framework",
            "network", "tcp", "udp", "http", "socket", "protocol", "server",
            "client", "request", "response", "database", "sql", "regex", "test", "deploy",
        ]

    def __init__(self) -> None:
        super().__init__()
        self.meta.capabilities = [
            "code_generation",
            "code_review",
            "debugging",
            "software_architecture",
        ]
        self.meta.authority_score = 0.95

    def process_task(self, objective: str) -> List[Claim]:
        if self.meta.model_path:
            raw_output = self._infer(objective)
            return self.parse_raw_output_to_claims(raw_output)
        else:
            return [Claim(
                statement=f"[Coding Placeholder] Analysis of: {objective}",
                confidence=0.9,
                domain=self.domain,
                status=ClaimStatus.UNVERIFIED
            )]

    def _infer(self, query: str) -> str:
        from saber.llm_engine import LLMEngine
        try:
            with LLMEngine(self.meta.model_path) as engine:
                system_prompt = (
                    "You are a Coding and Software Engineering AI specialist. Do NOT output conversational text. "
                    "Output ONLY a valid JSON array of claims. "
                    "Example: [{\"text\": \"A null pointer exception is likely in line 42\", \"confidence\": 0.95}]"
                )
                return engine.generate(query, system_prompt=system_prompt)
        except Exception as e:
            print(f"[CodingSpecialist] Inference failed: {e}")
            return f"[Coding Error] {e}"
