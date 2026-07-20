# -*- coding: utf-8 -*-
"""saber.specialists.finance

Finance and economics domain specialist.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from saber.signal import Claim, ClaimStatus
from saber.specialist import Specialist


class FinanceSpecialist(Specialist):
    """Specialist for finance, economics, and investment domain reasoning."""

    @property
    def domain(self) -> str:
        return "finance"

    @property
    def keywords(self) -> list[str]:
        return [
            "finance", "financial", "stock", "market", "investment", "trading",
            "portfolio", "risk", "bond", "equity", "financial derivative", "financial option",
            "hedge", "dividend", "interest", "inflation", "gdp", "economics",
            "banking", "loan", "credit", "asset", "liability", "revenue",
            "profit", "loss", "accounting", "budget", "tax", "valuation",
            "monetary", "fiscal", "capital", "yield", "leverage",
        ]

    def __init__(self) -> None:
        super().__init__()
        self.meta.capabilities = [
            "financial_analysis",
            "risk_assessment",
            "economic_reasoning",
            "investment_strategy",
            "accounting_principles",
        ]
        self.meta.authority_score = 0.93

    def process_task(self, objective: str) -> List[Claim]:
        if self.meta.model_path:
            raw_output = self._infer(objective)
            return self.parse_raw_output_to_claims(raw_output)
        else:
            return [Claim(
                statement=f"[Finance Placeholder] Analysis of: {objective}",
                confidence=0.9,
                domain=self.domain,
                status=ClaimStatus.UNVERIFIED
            )]

    def _infer(self, query: str) -> str:
        from saber.llm_engine import LLMEngine
        try:
            with LLMEngine(self.meta.model_path) as engine:
                system_prompt = (
                    "You are a Finance and Economics AI specialist. Do NOT output conversational text. "
                    "Output ONLY a valid JSON array of claims. "
                    "Example: [{\"text\": \"The DCF valuation suggests a fair value of $42.50\", \"confidence\": 0.88}]"
                )
                return engine.generate(query, system_prompt=system_prompt)
        except Exception as e:
            print(f"[FinanceSpecialist] Inference failed: {e}")
            return f"[Finance Error] {e}"
