# -*- coding: utf-8 -*-
"""saber.specialists.medical

Medical domain specialist.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from saber.signal import Claim, ClaimStatus
from saber.specialist import Specialist


class MedicalSpecialist(Specialist):
    """Specialist for medical domain reasoning."""

    @property
    def domain(self) -> str:
        return "medical"

    @property
    def keywords(self) -> list[str]:
        return [
            "medical", "drug", "patient", "clinical", "diagnosis", "treatment",
            "hospital", "disease", "symptom", "therapy", "pharma", "medicine",
            "health", "doctor", "surgery", "dosage", "prescription", "vaccine",
            "pathology", "anatomy", "cardiology", "oncology", "radiology",
            "medication", "antibiotic", "side effect", "chronic", "acute",
            "prognosis", "biopsy",
        ]

    def __init__(self) -> None:
        super().__init__()
        self.meta.capabilities = [
            "clinical_guidelines",
            "drug_interactions",
            "diagnostics",
            "patient_safety",
        ]
        self.meta.authority_score = 0.95

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
                statement=f"[Medical Placeholder] Analysis of: {objective}",
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
                    "You are a Medical AI specialist. Do NOT output conversational text. "
                    "Output ONLY a valid JSON array of claims. "
                    "Example: [{\"text\": \"Patient requires X\", \"confidence\": 0.95}]"
                )
                return engine.generate(query, system_prompt=system_prompt)
        except Exception as e:
            print(f"[MedicalSpecialist] Inference failed: {e}")
            return f"[Medical Error] {e}"
