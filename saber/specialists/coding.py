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
                statement=f"[Coding Placeholder] Analysis of: {objective}",
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
                        "You are an expert coding specialist. First, think step by step to solve the task. "
                        "Second, output exactly 3 factual claims about the logic/complexity. "
                        "Finally, output the complete Python implementation wrapped inside a ```python block.\n\n"
                        "Use this strict format:\n"
                        "REASONING: <your step by step thought process>\n"
                        "CLAIMS:\n1. <claim 1>\n2. <claim 2>\n3. <claim 3>\n"
                        "CODE:\n```python\n<your code here>\n```"
                    )
                else:
                    system_prompt = (
                        "You are a Coding and Software Engineering AI specialist. Do NOT output conversational text. "
                        "Output ONLY a valid JSON array of claims. "
                        "Example: [{\"text\": \"A null pointer exception is likely in line 42\", \"confidence\": 0.95}]"
                    )
                return engine.generate(query, system_prompt=system_prompt)
        except Exception as e:
            print(f"[CodingSpecialist] Inference failed: {e}")
            return f"[Coding Error] {e}"
