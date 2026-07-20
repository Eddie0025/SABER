# -*- coding: utf-8 -*-
"""saber.specialists.science

Science domain specialist (math, physics, chemistry).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from saber.signal import Claim, ClaimStatus
from saber.specialist import Specialist


class ScienceSpecialist(Specialist):
    """Specialist for scientific calculations and reasoning."""

    @property
    def domain(self) -> str:
        return "science"

    @property
    def keywords(self) -> list[str]:
        return [
            "science", "physics", "chemistry", "mathematics", "math", "calculation",
            "calculate", "formula", "equation", "velocity", "gravity", "mass",
            "molecule", "reaction", "solve", "atomic", "compound", "thermodynamics",
            "organic", "algebra", "calculus", "integral", "derivative", "acceleration",
            "density", "electron", "proton", "quantum", "entropy", "kinetic",
            "potential", "wavelength", "energy", "force", "momentum", "frequency",
            "projectile", "temperature", "pressure", "volume",
        ]

    def __init__(self) -> None:
        super().__init__()
        self.meta.capabilities = [
            "mathematical_reasoning",
            "physics_calculations",
            "chemical_reactions",
            "formula_derivation",
        ]
        self.meta.authority_score = 0.90

    def process_task(self, objective: str) -> List[Claim]:
        """Perform domain reasoning and return a list of Claims."""
        if self.meta.model_path:
            raw_output = self._infer(objective)
            try:
                # Attempt to parse the LLM's JSON output
                claims_data = json.loads(raw_output)
                if not isinstance(claims_data, list):
                    claims_data = [claims_data]
                    
                claims = []
                for c in claims_data:
                    claims.append(Claim(
                        statement=c.get("text", str(c)),
                        confidence=float(c.get("confidence", 0.9)),
                        domain=self.domain,
                        status=ClaimStatus.UNVERIFIED
                    ))
                return claims
            except json.JSONDecodeError:
                # Fallback if LLM fails to output strict JSON
                return [Claim(
                    statement=raw_output,
                    confidence=0.5,
                    domain=self.domain,
                    status=ClaimStatus.UNVERIFIED
                )]
        else:
            return [Claim(
                statement=f"[Science Placeholder] Analytical calculation of: {objective}",
                confidence=0.9,
                domain=self.domain,
                status=ClaimStatus.UNVERIFIED
            )]

    def _infer(self, query: str) -> str:
        """Run model inference dynamically using LLMEngine."""
        from saber.llm_engine import LLMEngine
        try:
            with LLMEngine(self.meta.model_path) as engine:
                system_prompt = (
                    "You are a Science AI specialist. Do NOT output conversational text. "
                    "Output ONLY a valid JSON array of claims containing step-by-step scientific or mathematical derivation. "
                    "Example: [{\"text\": \"Force F = m*a = 10kg * 9.8m/s^2 = 98 N\", \"confidence\": 0.98}]"
                )
                return engine.generate(query, system_prompt=system_prompt)
        except Exception as e:
            print(f"[ScienceSpecialist] Inference failed: {e}")
            return f"[Science Error] {e}"
