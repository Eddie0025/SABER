# -*- coding: utf-8 -*-
"""saber.specialists.cybersecurity

Cybersecurity domain specialist.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from saber.signal import Claim, ClaimStatus
from saber.specialist import Specialist


class CyberSpecialist(Specialist):
    """Specialist for cybersecurity domain reasoning."""

    @property
    def domain(self) -> str:
        return "cyber"

    @property
    def keywords(self) -> list[str]:
        return [
            "cyber", "security", "vulnerability", "exploit", "malware",
            "phishing", "firewall", "encryption", "breach", "ransomware",
            "intrusion", "threat", "attack", "hacker", "incident", "cve",
            "mitre", "apt", "penetration", "nmap", "payload", "trojan",
            "botnet", "ddos", "authentication", "authorization",
            "network", "protocol", "ssl", "tls", "certificate", "token", "session",
            "xss", "csrf",
        ]

    def __init__(self) -> None:
        super().__init__()
        self.meta.capabilities = [
            "threat_intelligence",
            "vulnerability_analysis",
            "incident_response",
            "network_security",
        ]
        self.meta.authority_score = 0.92

    def process_task(self, objective: str) -> List[Claim]:
        if self.meta.model_path:
            raw_output = self._infer(objective)
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
                statement=f"[Cyber Placeholder] Analysis of: {objective}",
                confidence=0.9,
                domain=self.domain,
                status=ClaimStatus.UNVERIFIED
            )]

    def _infer(self, query: str) -> str:
        from saber.llm_engine import LLMEngine
        try:
            with LLMEngine(self.meta.model_path) as engine:
                system_prompt = (
                    "You are a Cybersecurity AI specialist. Do NOT output conversational text. "
                    "Output ONLY a valid JSON array of claims. "
                    "Example: [{\"text\": \"CVE-2024-X detected\", \"confidence\": 0.95}]"
                )
                return engine.generate(query, system_prompt=system_prompt)
        except Exception as e:
            print(f"[CyberSpecialist] Inference failed: {e}")
            return f"[Cyber Error] {e}"
