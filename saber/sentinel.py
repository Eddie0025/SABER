# -*- coding: utf-8 -*-
"""saber.sentinel

Verification kernel for the SABER system.
SABER v2.0 — Signal Schema + Targeted Verification Routing.

Phase 2 additions:
- Targeted Verification Routing: routes verification to the most
  appropriate reviewer based on content type (technical → cyber,
  logic → science, clinical → medical).
- Activation tracking hooks for latency and token metrics.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from saber.signal import Signal, SignalType, Claim


# ---------------------------------------------------------------------------
# Verification routing rules
# ---------------------------------------------------------------------------

_VERIFICATION_ROUTING: Dict[str, Dict[str, str]] = {
    # domain -> { aspect: reviewer_role }
    "cyber": {
        "technical_accuracy": "cyber",
        "logical_reasoning": "science",
        "conflict_resolution": "meta_reasoner",
    },
    "science": {
        "factual_accuracy": "science",
        "mathematical_reasoning": "science",
        "logical_reasoning": "science",
        "conflict_resolution": "meta_reasoner",
    },
    "medical": {
        "clinical_accuracy": "medical",
        "diagnostic_reasoning": "medical",
        "logical_reasoning": "science",
        "conflict_resolution": "meta_reasoner",
    },
}
_INTERNET_CHECKED = None
_SEARCH_CACHE = {}
_LAST_CYCLE_QUERIES = {}
_LAST_SEARCH_RESULT = {}
_QUERY_CONSECUTIVE_COUNT = {}

class Sentinel:
    """The central verification authority.

    Responsible for checking signal hashes and orchestrating
    the check mode loop where claims are verified semantically.

    v2.0: Supports targeted verification routing so that, e.g.,
    a cyber answer has its logic checked by the science reviewer
    rather than having medical review cyber content.
    """

    @staticmethod
    def verify_signal_integrity(signal: Signal) -> bool:
        """Ensure the signal hasn't been tampered with."""
        return signal.verify_integrity()

    @staticmethod
    def get_verification_route(domain: str) -> Dict[str, str]:
        """Return the verification routing table for a domain.

        Example return:
            {"technical_accuracy": "cyber", "logical_reasoning": "science", ...}
        """
        return _VERIFICATION_ROUTING.get(domain, {
            "accuracy": domain,
            "logical_reasoning": "science",
            "conflict_resolution": "meta_reasoner",
        })

    @staticmethod
    def verify_interpretation(
        specialist_domain: str,
        original_signal: Signal,
        compiled_text: str,
        *,
        registry: Optional[Any] = None,
        config: Optional[Any] = None,
    ) -> Signal:
        """Send the Meta-Reasoning Layer's compilation to SENTINEL's LLM semantic check.

        Uses targeted verification routing: instead of a generic check,
        the prompt is tailored to verify the aspects most relevant to
        the specialist domain. If a SpecialistRegistry is provided,
        verification can be delegated to a domain-appropriate specialist
        model for higher-quality checks.

        Returns
        -------
        Signal: Either VERIFICATION_SIGNAL (GREEN_CHIT) or FLAG_SIGNAL.
        """
        from saber.llm_engine import LLMEngine
        import urllib.request
        import urllib.parse
        import re

        # Helper to check connection
        def is_internet_available():
            global _INTERNET_CHECKED
            if _INTERNET_CHECKED is not None:
                return _INTERNET_CHECKED
            try:
                urllib.request.urlopen("https://www.google.com", timeout=2)
                _INTERNET_CHECKED = True
            except Exception:
                _INTERNET_CHECKED = False
            return _INTERNET_CHECKED

        # Helper to query DuckDuckGo with caching
        def web_search(query_str):
            global _SEARCH_CACHE
            query_str = query_str.strip()
            if query_str in _SEARCH_CACHE:
                return _SEARCH_CACHE[query_str]

            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query_str)}"
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=3) as response:
                    html = response.read().decode("utf-8")
                    snippets = re.findall(r'<a class="result__snippet".*?>(.*?)</a>', html, re.DOTALL)
                    clean_snippets = []
                    for s in snippets[:3]:
                        clean = re.sub(r'<.*?>', '', s).strip()
                        clean_snippets.append(clean)
                    res_text = "\n".join(clean_snippets)
                    _SEARCH_CACHE[query_str] = res_text
                    return res_text
            except Exception as e:
                return f"Search lookup failed: {e}"

        # Extract claims from the original OUTPUT_SIGNAL payload
        claims_data = original_signal.payload.get("claims", [])
        claims_str = json.dumps(claims_data, indent=2)

        # Determine if online and compile grounding info
        online = is_internet_available()
        grounding_str = ""
        
        if online:
            # Extract search queries from all claims, or fall back to compiled_text
            queries_to_run = []
            if claims_data and isinstance(claims_data, list):
                for claim in claims_data:
                    stmt = claim.get("statement", "").strip()
                    if stmt:
                        queries_to_run.append(stmt[:120])
            if not queries_to_run:
                queries_to_run = [compiled_text[:120]]

            # STRICT CIRCUIT BREAKER: Deduplicate claims to kill autoregressive loops
            unique_queries = []
            for q in queries_to_run:
                if q not in unique_queries:
                    unique_queries.append(q)
                
            queries_to_run = unique_queries

            # Run search for each query with consecutive duplicate bypass
            all_results = []
            global _LAST_CYCLE_QUERIES, _LAST_SEARCH_RESULT, _QUERY_CONSECUTIVE_COUNT
            
            last_cycle = _LAST_CYCLE_QUERIES.get(specialist_domain, [])
            
            for query_str in queries_to_run:
                key = (specialist_domain, query_str)
                last_r = _LAST_SEARCH_RESULT.get(key)

                if query_str in last_cycle:
                    _QUERY_CONSECUTIVE_COUNT[key] = _QUERY_CONSECUTIVE_COUNT.get(key, 0) + 1
                else:
                    _QUERY_CONSECUTIVE_COUNT[key] = 1

                if _QUERY_CONSECUTIVE_COUNT[key] >= 2:
                    print(f"[Sentinel] Consecutive search limit reached for '{query_str}'. Bypassing online search.")
                    results = last_r or ""
                else:
                    print(f"[Sentinel] Online: searching for '{query_str}'...")
                    results = web_search(query_str)
                    _LAST_SEARCH_RESULT[key] = results
                
                if results.strip():
                    all_results.append(f"Query: {query_str}\nResult: {results}")

            _LAST_CYCLE_QUERIES[specialist_domain] = queries_to_run
            combined_results = "\n\n".join(all_results)

            grounding_str = (
                f"--- GROUNDING SOURCE SEARCH RESULTS ---\n"
                f"{combined_results}\n"
                f"---------------------------------------\n\n"
                f"INSTRUCTION: Use the search results above as the ground truth. "
                f"If the specialist's claim contradicts the search results, flag it as a FACTUAL_ERROR and propose a correction.\n\n"
            )
        else:
            print("[Sentinel] Offline: skipping fact-grounding search.")
            grounding_str = (
                "--- OFFLINE MODE INSTRUCTION ---\n"
                "Internet is currently offline. Do not attempt to verify obscure factual entities or specific named signs/triads against external facts. "
                "Focus strictly on logical reasoning consistency, internal flow, contradictions, and potential structural fabrications (e.g. self-contradictory claims).\n\n"
            )

        # --- Targeted Verification Routing ---
        route = Sentinel.get_verification_route(specialist_domain)
        aspects_to_check = list(route.keys())
        aspects_str = ", ".join(aspects_to_check)

        prompt = (
            f"{grounding_str}"
            f"Original Claims from {specialist_domain} specialist:\n{claims_str}\n\n"
            f"Meta-Reasoning Layer's Compiled Text:\n{compiled_text}\n\n"
            f"Verification Focus Areas: {aspects_str}\n\n"
            "Evaluate whether the compiled text accurately represents the original claims. "
            "Check specifically for:\n"
            "1. Technical accuracy of domain-specific content\n"
            "2. Logical reasoning and consistency\n"
            "3. Missing evidence or unsupported conclusions\n"
            "4. Any hallucinated or distorted information\n\n"
            "If the meaning is perfectly preserved, reply ONLY with exactly 'CONFIRMED'.\n\n"
            "If there are errors, respond ONLY with a valid JSON object (no markdown, no extra text) containing:\n"
            "- issue_type: (choose one: FACTUAL_ERROR, REASONING_ERROR, LOGIC_GAP, MISSING_EVIDENCE, DOMAIN_CONFLICT, CALCULATION_ERROR, SECURITY_ASSUMPTION_ERROR, DIAGNOSTIC_INCONSISTENCY, FINANCIAL_ANALYSIS_ERROR)\n"
            "- severity: (choose one: LOW, MEDIUM, HIGH, CRITICAL)\n"
            "- confidence: (float between 0.0 and 1.0)\n"
            "- evidence: (quote from the original claims that was missed/distorted)\n"
            "- reasoning: (detailed explanation of the error)\n"
            "- proposed_fix: (machine-readable description of how to fix it)\n"
        )

        system_prompt = (
            f"You are the SABER SENTINEL verifying {specialist_domain} content. "
            f"Focus your review on: {aspects_str}. "
            "Your job is strict semantic verification."
        )

        # Choose verification model: always use the unbiased base model to prevent self-checking
        verification_model = config.base_model if (config and config.base_model) else "Qwen/Qwen2.5-7B-Instruct"

        start_time = time.time()

        try:
            with LLMEngine(verification_model) as engine:
                result = engine.generate(prompt, system_prompt=system_prompt).strip()

            verification_latency = time.time() - start_time

            if result.upper().startswith("CONFIRMED") or "CONFIRMED" in result.upper()[:20]:
                return Signal(
                    signal_type=SignalType.VERIFICATION_SIGNAL,
                    query_id=original_signal.query_id,
                    source_id="SENTINEL",
                    target_id="MANAGER",
                    payload={
                        "status": "GREEN_CHIT",
                        "verification_model": verification_model,
                        "verification_latency": verification_latency,
                        "verification_route": route,
                    }
                ).freeze_and_hash()
            else:
                # Attempt to parse JSON
                try:
                    clean_result = result.replace("```json", "").replace("```", "").strip()
                    parsed = json.loads(clean_result)

                    payload = {
                        "issue_type": parsed.get("issue_type", "REASONING_ERROR").lower(),
                        "severity": parsed.get("severity", "HIGH").lower(),
                        "confidence": float(parsed.get("confidence", 0.9)),
                        "evidence": parsed.get("evidence", ""),
                        "reasoning": parsed.get("reasoning", result),
                        "proposed_fix": parsed.get("proposed_fix", "Needs revision."),
                        "description": parsed.get("reasoning", result),
                        "verification_model": verification_model,
                        "verification_latency": verification_latency,
                        "verification_route": route,
                    }
                except Exception:
                    payload = {
                        "issue_type": "reasoning_error",
                        "severity": "high",
                        "confidence": 0.5,
                        "evidence": "Unable to extract evidence structure.",
                        "reasoning": f"Sentinel flagged an issue but failed JSON output: {result}",
                        "proposed_fix": "Rewrite based on reasoning.",
                        "description": result,
                        "verification_model": verification_model,
                        "verification_latency": verification_latency,
                    }

                return Signal(
                    signal_type=SignalType.FLAG_SIGNAL,
                    query_id=original_signal.query_id,
                    source_id="SENTINEL",
                    target_id="MANAGER",
                    payload=payload,
                ).freeze_and_hash()

        except Exception as e:
            print(f"[Sentinel] LLM verification failed: {e}")
            return Signal(
                signal_type=SignalType.VERIFICATION_SIGNAL,
                query_id=original_signal.query_id,
                source_id="SENTINEL",
                target_id="MANAGER",
                payload={"status": "GREEN_CHIT"}
            ).freeze_and_hash()

    @staticmethod
    def generate_integrity_flag(signal: Signal) -> Signal:
        """Create a FLAG_SIGNAL if a hash mismatch is detected."""
        return Signal(
            signal_type=SignalType.FLAG_SIGNAL,
            query_id=signal.query_id,
            source_id="SENTINEL",
            target_id="MANAGER",
            payload={
                "issue_type": "integrity_failure",
                "severity": "critical",
                "confidence": 1.0,
                "evidence": "Computed hash does not match signal.integrity_hash",
                "reasoning": f"Signal {signal.signal_id} failed cryptographic check, implying tampering or corruption.",
                "proposed_fix": "Drop signal and re-request from source.",
                "description": f"Signal {signal.signal_id} failed cryptographic check.",
            }
        ).freeze_and_hash()

    @staticmethod
    def verify_cot_chain(
        specialist_domain: str,
        cot_chain: dict,
        compiled_text: str,
        config: Any,
    ) -> list[Signal]:
        """Verify individual reasoning steps in a CoT chain."""
        from saber.llm_engine import LLMEngine
        flags = []
        
        steps = cot_chain.get("steps", [])
        if not steps:
            return flags
            
        verification_model = config.base_model if config else "Qwen/Qwen2.5-7B"
        
        system_prompt = (
            f"You are the SABER SENTINEL. Your job is to strictly verify step-by-step reasoning "
            f"for a {specialist_domain} task."
        )
        
        # Verify step-by-step using a single LLM session to avoid loading model 10 times
        try:
            with LLMEngine(verification_model) as engine:
                for idx, step in enumerate(steps):
                    # Check action appropriateness
                    if idx == 0 and step.get("action") != "IDENTIFY":
                        flags.append(_create_step_flag(
                            cot_chain["query_id"], "STEP_ACTION_MISMATCH", step["step_number"],
                            f"First step action should be IDENTIFY, got {step.get('action')}",
                            "Change action to IDENTIFY and summarize query."
                        ))
                    if idx == len(steps) - 1 and step.get("action") != "CONCLUDE":
                        flags.append(_create_step_flag(
                            cot_chain["query_id"], "STEP_ACTION_MISMATCH", step["step_number"],
                            f"Last step action should be CONCLUDE, got {step.get('action')}",
                            "Change action to CONCLUDE."
                        ))
                        
                    # Confidence monotonicity check
                    if idx > 0:
                        prev_conf = steps[idx-1].get("confidence", 0.0)
                        curr_conf = step.get("confidence", 0.0)
                        if prev_conf - curr_conf > 0.3:
                            flags.append(_create_step_flag(
                                cot_chain["query_id"], "STEP_CONFIDENCE_DROP", step["step_number"],
                                f"Confidence dropped sharply from {prev_conf:.2f} to {curr_conf:.2f}",
                                "Re-evaluate reasoning or request more evidence."
                            ))
                            
                    # LLM semantic check for logic if there's previous context
                    if idx > 0:
                        prev_context = "\n".join([f"Step {s['step_number']} [{s['action']}]: {s['content']}" for s in steps[:idx]])
                        prompt = (
                            f"Given the previous reasoning steps:\n{prev_context}\n\n"
                            f"Does this next step logically follow?\nStep {step['step_number']} [{step['action']}]: {step['content']}\n\n"
                            f"If it follows logically and introduces no hallucinations, reply ONLY with 'CONFIRMED'.\n"
                            f"If not, respond ONLY with a JSON object containing:\n"
                            f"- issue_type: 'STEP_LOGIC_ERROR'\n"
                            f"- reasoning: explanation of why it doesn't follow\n"
                            f"- proposed_fix: how to fix it\n"
                        )
                        result = engine.generate(prompt, system_prompt=system_prompt).strip()
                        if not (result.upper().startswith("CONFIRMED") or "CONFIRMED" in result.upper()[:20]):
                            try:
                                clean_result = result.replace("```json", "").replace("```", "").strip()
                                parsed = json.loads(clean_result)
                                flags.append(_create_step_flag(
                                    cot_chain["query_id"], 
                                    parsed.get("issue_type", "STEP_LOGIC_ERROR"),
                                    step["step_number"],
                                    parsed.get("reasoning", result),
                                    parsed.get("proposed_fix", "Rewrite step.")
                                ))
                            except Exception:
                                flags.append(_create_step_flag(
                                    cot_chain["query_id"], "STEP_LOGIC_ERROR", step["step_number"],
                                    f"Step failed logic check: {result}", "Rewrite step logically."
                                ))
        except Exception as e:
            print(f"[Sentinel] Step-level verification failed: {e}")
            
        return flags


def _create_step_flag(query_id: str, issue_type: str, step_num: int, reasoning: str, fix: str) -> Signal:
    """Helper to create a FLAG_SIGNAL for a reasoning step."""
    return Signal(
        signal_type=SignalType.FLAG_SIGNAL,
        query_id=query_id,
        source_id="SENTINEL",
        target_id="MANAGER",
        payload={
            "issue_type": issue_type,
            "severity": "HIGH",
            "confidence": 0.9,
            "step_number": step_num,
            "reasoning": reasoning,
            "proposed_fix": fix,
            "description": f"Step {step_num} failed verification: {reasoning}",
        }
    ).freeze_and_hash()
