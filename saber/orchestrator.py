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

        Conversational intents (greetings, chitchat, general questions)
        are detected first and bypass the ambiguity gate entirely.
        """
        query_lower = query.strip().lower()
        words = query.split()

        # ----------------------------------------------------------
        # Conversational Intent Detection: Greetings and general
        # queries should never be flagged as ambiguous.
        # ----------------------------------------------------------
        greeting_patterns = {
            "hi", "hello", "hey", "howdy", "sup", "yo", "hola",
            "good morning", "good afternoon", "good evening", "good night",
            "how are you", "what's up", "whats up", "how's it going",
            "how do you do", "nice to meet you", "pleased to meet you",
            "thanks", "thank you", "bye", "goodbye", "see you",
            "who are you", "what are you", "what can you do",
            "help", "help me", "what is saber", "tell me about yourself",
        }
        # Check exact match or prefix match for greetings
        if query_lower in greeting_patterns:
            return 0.0
        for pattern in greeting_patterns:
            if query_lower.startswith(pattern):
                return 0.0

        # General questions (starts with interrogative words) are not ambiguous
        question_starters = ("what", "how", "why", "when", "where", "who", "which",
                             "can you", "could you", "would you", "tell me", "explain",
                             "describe", "show me", "give me")
        if any(query_lower.startswith(qs) for qs in question_starters):
            return 0.0

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
        query_clean = query.split("Options:")[0].split("options:")[0]
        query_lower = query_clean.lower()
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
        threshold = self.config.activation_threshold
        activated = [
            domain
            for domain, score in domain_scores.items()
            if score >= threshold and self.registry.get(domain) is not None
        ]
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

    def _generate_primer(self, query: str) -> str:
        """Generate a quick conversational acknowledgment before deep processing.

        This makes SABER feel responsive — like a real expert who says
        'Great question! Let me analyze that...' before diving in.
        """
        from saber.llm_engine import LLMEngine
        try:
            with LLMEngine(self.config.base_model, max_new_tokens=60) as engine:
                primer = engine.generate(
                    query,
                    system_prompt=(
                        "You are SABER, a friendly AI assistant. "
                        "Generate ONLY a brief 1-sentence acknowledgment of the user's message. "
                        "If it's a greeting, respond warmly. "
                        "If it's a question, briefly acknowledge it and say you'll analyze it. "
                        "Do NOT answer the question itself. Keep it under 20 words. "
                        "Examples: 'Great question! Let me analyze this for you.' "
                        "'Hey there! How can I help you today?' "
                        "'Interesting problem — let me dig into this.'"
                    ),
                )
            return primer.strip()
        except Exception:
            return ""

    def process_query(
        self,
        query: str,
        tier: Optional[VerificationTier] = None,
        activated_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run the full SABER pipeline for a user query.

        Returns a dict with keys:
            ``query_id``, ``answer``, ``confidence``, ``flags``,
            ``domains_activated``, ``verification_tier``,
            ``verification_cycles``, ``audit_records``.
        """
        query_id = str(uuid.uuid4())
        self.audit.log_query(query_id, query)

        # --- Conversational Primer: Disabled for now to avoid latency ---
        primer = ""

        # --- Ambiguity & Domain Selection ---
        if activated_domains is not None:
            activated = activated_domains
            domain_scores = {}
        else:
            # --- Ambiguity check ---
            ambiguity = self.detect_ambiguity(query)
            if ambiguity >= self.config.ambiguity_threshold:
                result = {
                    "query_id": query_id,
                    "status": "clarification_needed",
                    "ambiguity_score": ambiguity,
                    "answer": (
                        f"{primer}\n\n" if primer else ""
                    ) + (
                        "Your query appears ambiguous.  Could you provide more "
                        "detail or specify the domain (medical, legal, cyber, finance)?"
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
            # ----------------------------------------------------------
            # General Conversation Fallback: Use base Qwen for greetings,
            # chitchat, and queries outside specialist domains.
            # The primer IS the response for simple greetings.
            # For longer general queries, generate a full response.
            # ----------------------------------------------------------
            from saber.llm_engine import LLMEngine
            query_lower = query.strip().lower()
            greeting_words = {"hi", "hello", "hey", "howdy", "sup", "yo", "hola",
                              "good morning", "good afternoon", "good evening"}

            # For simple greetings, the primer alone is sufficient
            if query_lower in greeting_words or len(query.split()) <= 3:
                general_answer = primer if primer else "Hey there! How can I help you today?"
            else:
                # For longer general queries, generate a full conversational response
                try:
                    with LLMEngine(self.config.base_model, max_new_tokens=512) as engine:
                        general_answer = engine.generate(
                            query,
                            system_prompt=(
                                "You are SABER, a helpful, knowledgeable AI assistant. "
                                "Respond naturally and conversationally. Be friendly, "
                                "concise, and helpful."
                            ),
                        )
                    if primer:
                        general_answer = f"{primer}\n\n{general_answer}"
                except Exception as e:
                    general_answer = f"I'm sorry, I encountered an issue: {e}"

            result = {
                "query_id": query_id,
                "status": "general_conversation",
                "answer": general_answer,
                "confidence": 0.7,
                "flags": [],
                "domains_activated": ["general"],
                "verification_tier": 0,
                "verification_cycles": 0,
                "domain_scores": domain_scores,
            }
            self.audit.log_output(query_id, result["answer"])
            return result

        # --- Verification tier ---
        ver_tier = self.assign_verification_tier(tier)

        # --- Delegate to Meta-Reasoning Layer ---
        result = self.meta_reasoner.execute(
            query=query,
            query_id=query_id,
            activated_domains=activated,
            verification_tier=ver_tier,
        )

        # Prepend the conversational primer to the specialist's deep answer
        if primer and result.get("answer"):
            result["answer"] = f"{primer}\n\n{result['answer']}"

        self.audit.log_output(query_id, result.get("answer", ""))
        return result

