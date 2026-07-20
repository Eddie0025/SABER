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
            except Exception:
                return [Claim(
                    statement=raw_output,
                    confidence=0.5,
                    domain=self.domain,
                    status=ClaimStatus.UNVERIFIED
                )]
        else:
            return [Claim(
                statement=f"[Finance Placeholder] Analysis of: {objective}",
                confidence=0.9,
                domain=self.domain,
                status=ClaimStatus.UNVERIFIED
            )]

    def _infer(self, query: str) -> str:
        from saber.llm_engine import LLMEngine
        import os
        try:
            with LLMEngine(self.meta.model_path) as engine:
                if os.getenv("SABER_BENCHMARK_MODE") == "1":
                    system_prompt = (
                        "You are an expert financial analyst. First, think step by step to solve the question. "
                        "Second, output exactly 3 factual claims that support your reasoning. "
                        "Finally, state the correct numerical answer.\n\n"
                        "Use this strict format:\n"
                        "REASONING: <your step by step thought process>\n"
                        "CLAIMS:\n1. <claim 1>\n2. <claim 2>\n3. <claim 3>\n"
                        "ANSWER: <numeric value>"
                    )
                else:
                    system_prompt = (
                        "You are a Finance and Economics AI specialist. Do NOT output conversational text. "
                        "Output ONLY a valid JSON array of claims. "
                        "Example: [{\"text\": \"The DCF valuation suggests a fair value of $42.50\", \"confidence\": 0.88}]"
                    )
                return engine.generate(query, system_prompt=system_prompt)
        except Exception as e:
            print(f"[FinanceSpecialist] Inference failed: {e}")
            return f"[Finance Error] {e}"
