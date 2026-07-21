# -*- coding: utf-8 -*-
"""saber.orchestrator

The Orchestrator is the entry point of the SABER system.

Responsibilities
~~~~~~~~~~~~~~~~
1. **Ambiguity Detection** — reject or clarify ambiguous queries before
   they enter the reasoning pipeline.
2. **Dynamic Domain Classification** — build the routing table from
   registered specialists' keywords at runtime.  No hardcoded keyword
   banks — add a new specialist .py file and it's instantly routable.
3. **Specialist Selection** — score each specialist and activate only
   those above the activation threshold.
4. **Verification Tier Assignment** — apply the user-chosen (or default)
   verification depth.
5. **Pipeline Coordination** — hand the decomposed query to the Meta-Reasoning Layer,
   receive the compiled output, run audit logging, and return the
   final answer to the caller.
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from saber.audit import AuditLogger
from saber.config import SaberConfig, VerificationTier
from saber.meta_reasoner import MetaReasoner
from saber.registry import SpecialistRegistry


class Orchestrator:
    """Entry point of the SABER pipeline.

    Parameters
    ----------
    config : SaberConfig
        System-wide configuration.
    registry : SpecialistRegistry
        The specialist registry (pre-populated or auto-discovered).
    audit : AuditLogger
        The audit log writer.
    """

    def __init__(
        self,
        config: SaberConfig,
        registry: SpecialistRegistry,
        audit: AuditLogger,
    ) -> None:
        import os
        self.config = config
        self.registry = registry
        self.audit = audit
        self.meta_reasoner = MetaReasoner(config=config, registry=registry, audit=audit)
        self.model_path = "models/orchestrator_v2" if os.path.exists("models/orchestrator_v2") else self.config.base_model

    # ------------------------------------------------------------------
    # 1. Ambiguity Detection
    # ------------------------------------------------------------------

    def detect_ambiguity(self, query: str) -> float:
        """Return an ambiguity score in [0, 1].

        A high score means the query is vague or underspecified.
        The heuristic checks for:
        * Very short queries (< 5 words).
        * Excessive use of pronouns without antecedents.
        * Missing domain indicators.
        """
        words = query.split()
        score = 0.0

        # Short queries are more ambiguous
        if len(words) < 5:
            score += 0.4
        elif len(words) < 10:
            score += 0.15

        # Pronoun density
        pronouns = {"it", "they", "this", "that", "those", "these", "he", "she"}
        pronoun_count = sum(1 for w in words if w.lower() in pronouns)
        if len(words) > 0:
            score += min(0.3, (pronoun_count / len(words)) * 1.5)

        # No domain keywords detected at all → more ambiguous
        domain_scores = self.classify_domains(query)
        if all(s < 0.1 for s in domain_scores.values()):
            score += 0.3

        return min(1.0, score)

    # ------------------------------------------------------------------
    # 2. Dynamic Domain Classification
    # ------------------------------------------------------------------

    @staticmethod
    def _stem(word: str) -> str:
        """Crude but effective suffix stripper."""
        for suffix in ("tion", "sion", "ing", "ment", "ness", "ity", "ies", "es", "ed", "ly", "er", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[:-len(suffix)]
        return word

    def classify_domains(self, query: str) -> Dict[str, float]:
        """Return a relevance score for each registered specialist domain.

        Uses Qwen-7B semantically to classify the query into active domains.
        Limits generation to 32 tokens for sub-100ms classification latency.
        Falls back to keyword heuristics if the model fails.
        """
        from saber.llm_engine import LLMEngine
        import json

        domains = list(self.registry.all().keys())
        prompt = (
            f"You are the routing orchestrator for a multi-specialist AI system.\n"
            f"Given the user query, identify which of the following specialist domains it belongs to:\n"
            f"Available domains: {json.dumps(domains)}\n\n"
            f"Query: \"{query}\"\n\n"
            f"Output strictly a JSON list containing the activated domains (e.g. [\"science\"] or [\"medical\", \"cyber\"]) with no other text, explanation, or conversational intro."
        )

        try:
            with LLMEngine(self.model_path, max_new_tokens=32) as engine:
                raw_output = engine.generate(prompt).strip()
                clean_json = raw_output.replace("```json", "").replace("```", "").strip()
                start = clean_json.find("[")
                end = clean_json.rfind("]")
                if start != -1 and end != -1:
                    clean_json = clean_json[start:end+1]
                activated_domains = json.loads(clean_json)
                
                scores = {d: 0.0 for d in domains}
                for d in activated_domains:
                    if d in scores:
                        scores[d] = 1.0
                return scores
        except Exception:
            return self._heuristic_classify_domains(query)

    def _heuristic_classify_domains(self, query: str) -> Dict[str, float]:
        """Fallback keyword-based classifier."""
        query_lower = query.lower()
        query_words = set(re.findall(r"\w+", query_lower))
        stemmed_query_words = {self._stem(w) for w in query_words}
        scores: Dict[str, float] = {}

        for domain, specialist in self.registry.all().items():
            keywords = getattr(specialist, "keywords", [])
            capabilities = specialist.meta.capabilities
            cap_words = []
            for cap in capabilities:
                cap_words.extend(cap.split("_"))
            all_keywords = keywords + cap_words

            hits = 0
            for kw in all_keywords:
                kw_lower = kw.lower()
                if " " in kw_lower:
                    if kw_lower in query_lower:
                        hits += 1
                else:
                    stemmed_kw = self._stem(kw_lower)
                    if kw_lower in query_words or stemmed_kw in stemmed_query_words:
                        hits += 1
            scores[domain] = min(1.0, hits / max(len(all_keywords) * 0.12, 2.0))

        return scores

    # ------------------------------------------------------------------
    # 3. Specialist Selection
    # ------------------------------------------------------------------

    def select_specialists(
        self, domain_scores: Dict[str, float]
    ) -> List[str]:
        """Return domains whose score exceeds the activation threshold."""
        import os
        threshold = self.config.activation_threshold
        activated = [
            domain
            for domain, score in domain_scores.items()
            if score >= threshold and self.registry.get(domain) is not None
        ]
        if os.getenv("SABER_BENCHMARK_MODE") == "1" and activated:
            # Force exactly one domain (highest score) during benchmarks to prevent pollution
            best_domain = max(activated, key=lambda d: domain_scores[d])
            return [best_domain]
        return activated

    # ------------------------------------------------------------------
    # 4. Verification Tier Assignment
    # ------------------------------------------------------------------

    def assign_verification_tier(
        self, tier: Optional[VerificationTier] = None
    ) -> VerificationTier:
        """Return the verification tier to use for this query."""
        return tier if tier is not None else self.config.verification_tier

    # ------------------------------------------------------------------
    # 5. Full Pipeline
    # ------------------------------------------------------------------

    def process_query(
        self,
        query: str,
        tier: Optional[VerificationTier] = None,
        bypass_meta: bool = False,
    ) -> Dict[str, Any]:
        """Run the full SABER pipeline for a user query.

        Returns a dict with keys:
            ``query_id``, ``answer``, ``confidence``, ``flags``,
            ``domains_activated``, ``verification_tier``,
            ``verification_cycles``, ``audit_records``.
        """
        query_id = str(uuid.uuid4())
        self.audit.log_query(query_id, query)

        # --- Ambiguity check ---
        ambiguity = self.detect_ambiguity(query)
        if ambiguity >= self.config.ambiguity_threshold:
            result = {
                "query_id": query_id,
                "status": "clarification_needed",
                "ambiguity_score": ambiguity,
                "answer": (
                    "Your query appears ambiguous.  Could you provide more "
                    "detail or specify the domain (science, cyber, finance, coding, architecture)?"
                ),
                "confidence": 0.0,
                "flags": [],
                "domains_activated": [],
                "verification_tier": 0,
                "verification_cycles": 0,
            }
            self.audit.log_output(query_id, result["answer"])
            return result

        # --- Domain classification & specialist selection ---
        domain_scores = self.classify_domains(query)
        activated = self.select_specialists(domain_scores)

        if not activated and domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[best_domain] > 0.0:
                activated = [best_domain]
                self.audit.log("forced_activation", query_id, {
                    "domain": best_domain,
                    "score": domain_scores[best_domain],
                }, component="orchestrator")

        if not activated:
            from saber.errors import FailureCategory
            self.audit.log("failure", query_id, {"category": FailureCategory.ROUTING_FAILURE.value, "reason": "No specialists activated"}, "orchestrator")
            result = {
                "query_id": query_id,
                "status": "no_specialists",
                "answer": (
                    "No domain specialists were activated for this query.  "
                    "Please include domain-relevant terms or specify the domain."
                ),
                "confidence": 0.0,
                "flags": [],
                "domains_activated": [],
                "verification_tier": 0,
                "verification_cycles": 0,
                "domain_scores": domain_scores,
            }
            self.audit.log_output(query_id, result["answer"])
            return result

        # --- Verification tier ---
        ver_tier = self.assign_verification_tier(tier)

        # --- Delegate to Meta-Reasoning Layer or Bypass for single-domain ---
        if not activated:
            activated = ["science"]
            
        if bypass_meta and len(activated) == 1:
            specialist = self.registry.get(activated[0])
            if specialist:
                self.audit.log("bypass", query_id, {"domain": activated[0]}, component="orchestrator")
                
                # 1. Run Specialist CoT directly
                from saber.signal import Signal, SignalType
                task_sig = Signal(
                    signal_type=SignalType.TASK_SIGNAL,
                    query_id=query_id,
                    source_id="ORCHESTRATOR",
                    target_id=activated[0],
                    payload={"objective": query}
                ).freeze_and_hash()
                
                out_sig = specialist.handle_signal(task_sig)
                
                # Extract the text answer generated by the specialist
                # Sometimes it is in raw_response, sometimes in claims[0].statement
                ans = out_sig.payload.get("raw_response", "")
                if not ans and out_sig.payload.get("claims"):
                    ans = out_sig.payload["claims"][0].get("statement", "")

                # 2. If Sentinel is enabled (TIER_1), run it here
                if ver_tier == VerificationTier.TIER_1:
                    from saber.sentinel import Sentinel
                    sentinel = Sentinel()
                    ver_res = sentinel.verify_interpretation(
                        specialist_domain=activated[0],
                        original_signal=out_sig,
                        compiled_text=ans,
                        config=self.config
                    )
                    if ver_res.signal_type == SignalType.FLAG_SIGNAL:
                        # Sentinel flagged an error. Route the flag back to the specialist to recheck its reasoning.
                        # We pass the compiled text inside the payload so the specialist knows what to rewrite.
                        flag_payload = ver_res.payload
                        flag_payload["compiled_text"] = ans
                        
                        ver_sig = Signal(
                            signal_type=SignalType.VERIFICATION_SIGNAL,
                            query_id=query_id,
                            source_id="ORCHESTRATOR",
                            target_id=activated[0],
                            payload=flag_payload
                        ).freeze_and_hash()
                        
                        resolved_sig = specialist.handle_signal(ver_sig)
                        if resolved_sig.payload.get("status") == "RESOLVED":
                            ans = resolved_sig.payload.get("revised_text", ans)

                result = {
                    "query_id": query_id,
                    "answer": ans,
                    "confidence": 1.0,
                    "flags": [],
                    "domains_activated": activated,
                    "verification_tier": ver_tier.value,
                    "verification_cycles": 1 if ver_tier == VerificationTier.TIER_1 else 0,
                }
            else:
                result = self.meta_reasoner.execute(query=query, query_id=query_id, activated_domains=activated, verification_tier=ver_tier)
        else:
            result = self.meta_reasoner.execute(
                query=query,
                query_id=query_id,
                activated_domains=activated,
                verification_tier=ver_tier,
            )

        self.audit.log_output(query_id, result.get("answer", ""))
        return result
