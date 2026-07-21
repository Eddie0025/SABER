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
        import os
        if self.meta.model_path:
            raw_output = self._infer(objective)
            self._last_raw_response = raw_output
            
            if os.getenv("SABER_BENCHMARK_MODE") == "1":
                # Parse claims from sequential benchmark format
                claims_texts = []
                if "CLAIMS:" in raw_output:
                    try:
                        after_claims = raw_output.split("CLAIMS:")[1]
                        end_marker = "ANSWER:" if "ANSWER:" in after_claims else "CODE:"
                        claims_block = after_claims.split(end_marker)[0]
                        for line in claims_block.split("\n"):
                            line = line.strip()
                            if line and (line[0].isdigit() or line.startswith("-")):
                                clean_claim = line.lstrip("1234567890.- ").strip()
                                if clean_claim:
                                    claims_texts.append(clean_claim)
                    except Exception:
                        pass
                if not claims_texts:
                    claims_texts = [raw_output[:100]]
                    
                claims = []
                for text in claims_texts:
                    claims.append(Claim(
                        statement=text,
                        confidence=0.9,
                        domain=self.domain,
                        status=ClaimStatus.UNVERIFIED
                    ))
                return claims
            
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
        import os
        try:
            with LLMEngine(self.meta.model_path) as engine:
                if os.getenv("SABER_BENCHMARK_MODE") == "1":
                    system_prompt = (
                        "You are a science specialist with expertise in physics, chemistry, "
                        "biology, and mathematical reasoning. Show all work and explain "
                        "each step clearly. Think through your reasoning step by step "
                        "before providing your final answer."
                    )
                else:
                    system_prompt = (
                        "You are a Science AI specialist. Do NOT output conversational text. "
                        "Output ONLY a valid JSON array of claims containing step-by-step scientific or mathematical derivation. "
                        "Example: [{\"text\": \"Force F = m*a = 10kg * 9.8m/s^2 = 98 N\", \"confidence\": 0.98}]"
                    )
                return engine.generate(query, system_prompt=system_prompt)
        except Exception as e:
            print(f"[ScienceSpecialist] Inference failed: {e}")
            return f"[Science Error] {e}"
