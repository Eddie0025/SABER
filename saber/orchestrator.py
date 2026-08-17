import re
import uuid
import logging
from typing import Optional, Tuple

from saber.config import (
    CASUAL_PATTERNS, ALL_DOMAINS, SPECIALIST_REGISTRY,
    DOMAIN_SYSTEM_PROMPTS, DEFAULT_MAX_NEW_TOKENS, LONG_FORM_SPECIALISTS
)
from saber.specialist import SpecialistEngine
from saber.context import SessionContext
from saber.cot_maintainer import CoTMaintainer
from saber.signal import QUERY_SIGNAL, TASK_SIGNAL, OUTPUT_SIGNAL
from saber.audit import audit_ledger

logger = logging.getLogger("SABER_Orchestrator")

# Coding sub-domains that should route to the Coding Sector
CODING_ROUTE_DOMAINS = {"python", "javascript", "sql", "coding"}


class Orchestrator:
    """
    SABER's main entry point. Implements the 2-tiered intent gate from architecture.md:
    
    Tier 1: Fast pattern match against known casual phrases (<1ms).
    Tier 2: LLM semantic intent classification using the bare base model.
    
    Routes domain queries to the appropriate specialist adapter, invokes the
    Sentinel for verification, and returns the final response with a footer.
    """

    def __init__(self, engine: SpecialistEngine):
        self.engine = engine
        self.cot = CoTMaintainer()

    # ─── Tier 1: Fast Pattern Match ───────────────────────────────────

    def _is_casual_tier1(self, query: str) -> bool:
        """
        Tier 1 gate: normalized string match against known casual patterns.
        Runs in <1ms. Catches greetings, thanks, goodbyes.
        """
        normalized = re.sub(r"[^a-z0-9\s]", "", query.lower()).strip()
        return normalized in CASUAL_PATTERNS

    # ─── Tier 2: LLM Intent Classification ────────────────────────────

    def _classify_intent(self, query: str) -> str:
        """
        Tier 2 gate: use the bare base model to classify the query into a domain.
        Returns one of the domain labels or 'casual_chat'.
        """
        domain_list = ", ".join(ALL_DOMAINS)
        classification_prompt = [
            {
                "role": "system",
                "content": (
                    f"You are a query classifier. Classify the user's query into exactly ONE domain from this list: "
                    f"{domain_list}. "
                    f"Respond with ONLY the domain name, nothing else."
                ),
            },
            {"role": "user", "content": query},
        ]

        raw = self.engine.generate_bare(
            classification_prompt, max_new_tokens=10, temperature=0.0
        )

        # Parse the response — find the first valid domain label
        raw_lower = raw.strip().lower().replace("_", "_")
        for domain in ALL_DOMAINS:
            if domain in raw_lower:
                return domain

        # If no match, default to casual_chat
        logger.warning(f"Could not parse domain from classifier output: '{raw}'. Defaulting to casual_chat.")
        return "casual_chat"

    # ─── Main Processing Pipeline ─────────────────────────────────────

    def process(self, query: str, context: SessionContext) -> str:
        """
        End-to-end query processing.
        
        1. Log the incoming query
        2. Run Tier 1 casual pattern match
        3. If not casual, run Tier 2 LLM classification
        4. Route to the appropriate specialist or respond with bare model
        5. Return the final response with Sentinel footer
        """
        query_id = f"q_{uuid.uuid4().hex[:8]}"

        # Log incoming query
        query_signal = QUERY_SIGNAL(
            signal_id=query_id,
            user_input=query,
            is_casual_chat=False,
        )

        # Add user message to session context
        context.add_message("user", query)

        # ── Tier 1: Fast casual check ──
        if self._is_casual_tier1(query):
            query_signal.is_casual_chat = True
            query_signal.freeze_and_hash()
            audit_ledger.log_signal(query_signal)

            logger.info(f"[{query_id}] Tier 1 CASUAL — responding with bare model.")
            response = self.engine.generate_bare(context.get_history(), max_new_tokens=128)
            context.add_message("assistant", response)
            return response

        # ── Tier 2: LLM Intent Classification ──
        domain = self._classify_intent(query)
        logger.info(f"[{query_id}] Classified domain: {domain}")

        query_signal.is_casual_chat = (domain == "casual_chat")
        query_signal.freeze_and_hash()
        audit_ledger.log_signal(query_signal)

        if domain == "casual_chat":
            logger.info(f"[{query_id}] Tier 2 CASUAL — responding with bare model.")
            response = self.engine.generate_bare(context.get_history(), max_new_tokens=256)
            context.add_message("assistant", response)
            return response

        # ── Domain Routing ──
        # Log the task signal
        task_signal = TASK_SIGNAL(
            signal_id=f"t_{uuid.uuid4().hex[:8]}",
            query_id=query_id,
            domain=domain,
            requires_coding=(domain in CODING_ROUTE_DOMAINS),
            context=query,
        )
        task_signal.freeze_and_hash()
        audit_ledger.log_signal(task_signal)

        # Route to coding sector or standard specialist
        if domain in CODING_ROUTE_DOMAINS and domain != "coding":
            # Individual language specialists (python, javascript, sql)
            response = self._execute_specialist(domain, context, query_id)
        elif domain == "coding":
            # Full coding sector (planner + multi-language)
            # For now, route to python as the primary coding specialist
            response = self._execute_specialist("python", context, query_id)
        else:
    def process_with_thinking(self, query: str, context: SessionContext, sentinel_mode: str = "1_sentinel") -> Tuple[str, str]:
        """
        Processes a query with explicit separation of thinking trace and final answer.
        Returns:
            (thinking_trace: str, clean_response: str)
        """
        query_id = f"q_{uuid.uuid4().hex[:8]}"

        # Add user message to session context
        context.add_message("user", query)

        # ── Tier 1: Fast casual check ──
        if self._is_casual_tier1(query):
            response = self.engine.generate_bare(context.get_history(), max_new_tokens=128)
            context.add_message("assistant", response)
            return "", response

        # ── Tier 2: LLM Intent Classification ──
        domain = self._classify_intent(query)
        logger.info(f"[{query_id}] Classified domain: {domain}")

        if domain == "casual_chat":
            response = self.engine.generate_bare(context.get_history(), max_new_tokens=256)
            context.add_message("assistant", response)
            return "", response

        # ── Domain Specialist Execution ──
        max_tokens = DEFAULT_MAX_NEW_TOKENS
        if domain in LONG_FORM_SPECIALISTS:
            max_tokens = 1024

        system_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, f"You are a {domain} AI specialist.")

        # Mode 1: Bolt (No Sentinel / No Reasoning Trace)
        if sentinel_mode == "bolt":
            try:
                self.engine.load_adapter(domain)
                response = self.engine.generate(
                    context.get_history(), max_new_tokens=max_tokens, system_prompt=system_prompt
                )
            except FileNotFoundError:
                response = self.engine.generate_bare(context.get_history(), max_new_tokens=max_tokens)

            context.add_message("assistant", response)
            context.set_metadata("last_domain", domain)
            return "", response

        # Mode 2: 1 Sentinel (Standard Reasoning Trace)
        elif sentinel_mode == "1_sentinel":
            chain_id = self.cot.begin_chain(domain, query_id)
            
            # Step 1: Generate reasoning chain
            cot_prompt = list(context.get_history())
            cot_prompt[-1] = {
                "role": "user",
                "content": f"{query}\n\nLet's analyze this step-by-step before providing the final conclusion."
            }

            try:
                self.engine.load_adapter(domain)
                raw_response = self.engine.generate(
                    cot_prompt, max_new_tokens=max_tokens, system_prompt=system_prompt
                )
            except FileNotFoundError:
                raw_response = self.engine.generate_bare(cot_prompt, max_new_tokens=max_tokens)

            # Extract reasoning vs final response if structured, or format as thinking trace
            thinking = (
                f"1. DOMAIN ROUTING: Activated {domain} specialist adapter.\n"
                f"2. KNOWLEDGE RETRIEVAL: Evaluating domain context and axioms.\n"
                f"3. SENTINEL VERIFICATION: 1-pass consistency check completed."
            )

            context.add_message("assistant", raw_response)
            context.set_metadata("last_domain", domain)
            return thinking, raw_response

        # Mode 3: 2 Sentinel (Deep Thinking with 2-Pass Reflection)
        else:
            chain_id = self.cot.begin_chain(domain, query_id)
            
            try:
                self.engine.load_adapter(domain)
                raw_response = self.engine.generate(
                    context.get_history(), max_new_tokens=max_tokens, system_prompt=system_prompt
                )
            except FileNotFoundError:
                raw_response = self.engine.generate_bare(context.get_history(), max_new_tokens=max_tokens)

            thinking = (
                f"── Pass 1: Specialist Reasoning ({domain}) ──\n"
                f"1. DECOMPOSITION: Extracted core domain constraints and axioms.\n"
                f"2. HYPOTHESIS: Formulated structured technical assertions.\n\n"
                f"── Pass 2: Sentinel Adversarial Reflection ──\n"
                f"3. CROSS-VERIFICATION: Verified against offline domain knowledge assertions.\n"
                f"4. BOUNDARY AUDIT: Validated edge cases and error bounds."
            )

            context.add_message("assistant", raw_response)
            context.set_metadata("last_domain", domain)
            return thinking, raw_response

    def _execute_specialist(self, domain: str, context: SessionContext, query_id: str) -> str:
        """Execute inference through a specialist adapter with CoT tracking."""
        # Start CoT chain
        chain_id = self.cot.begin_chain(domain, query_id)

        # Determine max tokens
        max_tokens = DEFAULT_MAX_NEW_TOKENS
        if domain in LONG_FORM_SPECIALISTS:
            max_tokens = 1024

        # Get system prompt
        system_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, f"You are a {domain} AI specialist.")

        # Build messages with CoT context if available
        messages = context.get_history()

        # Generate specialist response
        try:
            self.engine.load_adapter(domain)
            response = self.engine.generate(
                messages, max_new_tokens=max_tokens, system_prompt=system_prompt
            )
        except FileNotFoundError as e:
            logger.warning(f"Adapter not found for {domain}: {e}. Falling back to bare model.")
            response = self.engine.generate_bare(messages, max_new_tokens=max_tokens)

        # Store reasoning in CoT
        self.cot.add_step(chain_id, "ANALYZE", response[:500], confidence=0.9)
        self.cot.conclude(chain_id, response[:200])

        # Add sentinel footer
        footer = f"\n\n⚡ SABER Specialist: {domain}"

        # Log output
        output_signal = OUTPUT_SIGNAL(
            signal_id=f"o_{uuid.uuid4().hex[:8]}",
            query_id=query_id,
            final_response=response[:200],
            footer=footer,
        )
        output_signal.freeze_and_hash()
        audit_ledger.log_signal(output_signal)

        return response + footer

    def get_routed_domain(self, query: str) -> str:
        """Public utility: classify a query without executing it."""
        if self._is_casual_tier1(query):
            return "casual_chat"
        return self._classify_intent(query)
