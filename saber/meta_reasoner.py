import logging
from typing import List, Dict

from saber.specialist import SpecialistEngine
from saber.config import DOMAIN_SYSTEM_PROMPTS

logger = logging.getLogger("SABER_MetaReasoner")

SYNTHESIS_PROMPT = """You are the SABER Meta Reasoner. Multiple domain specialists have provided their analyses of the same query. Your job is to:

1. Identify areas of agreement across specialists.
2. Resolve any contradictions by favoring the more domain-relevant specialist.
3. Synthesize a single, coherent, comprehensive response.
4. Clearly attribute insights to the relevant domain when helpful.

Do NOT simply concatenate the responses. Produce a unified answer that reads as one expert analysis.

USER QUERY:
{query}

SPECIALIST RESPONSES:
{specialist_outputs}

SYNTHESIZED RESPONSE:"""


class MetaReasoner:
    """
    The Orchestrator's synthesis engine. Analyzes incoming complex tasks,
    determines which Domain Specialists need to be invoked, aggregates their
    individual CoT (Chain of Thought) claims, and synthesizes a final, verified
    response using the bare base model.
    
    Note: Coding Sector output BYPASSES the Meta Reasoner entirely.
    """

    def __init__(self, engine: SpecialistEngine):
        self.engine = engine

    def synthesize(self, query: str, specialist_responses: Dict[str, str]) -> str:
        """
        Synthesize conflicting or complementary claims from multiple specialists
        into a cohesive final output using the bare base model.
        
        Args:
            query: The original user query.
            specialist_responses: Dict mapping domain names to their responses.
        
        Returns:
            A single synthesized response string.
        """
        if len(specialist_responses) == 0:
            return "No specialist responses to synthesize."

        if len(specialist_responses) == 1:
            # Single specialist — no synthesis needed, return directly
            domain, response = next(iter(specialist_responses.items()))
            return response

        # Format specialist outputs for the synthesis prompt
        formatted_outputs = ""
        for domain, response in specialist_responses.items():
            formatted_outputs += f"\n--- {domain.upper()} SPECIALIST ---\n{response}\n"

        synthesis_prompt = SYNTHESIS_PROMPT.format(
            query=query,
            specialist_outputs=formatted_outputs,
        )

        messages = [{"role": "user", "content": synthesis_prompt}]

        logger.info(f"Synthesizing {len(specialist_responses)} specialist responses...")
        synthesized = self.engine.generate_bare(messages, max_new_tokens=1024, temperature=0.3)

        return synthesized

    def multi_specialist_query(self, query: str, domains: List[str]) -> str:
        """
        Execute a query across multiple specialists and synthesize the results.
        
        Args:
            query: The user's query.
            domains: List of domain names to query.
            
        Returns:
            Synthesized response from all specialists.
        """
        specialist_responses = {}

        for domain in domains:
            logger.info(f"Querying specialist: {domain}")
            try:
                system_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, f"You are a {domain} AI specialist.")
                messages = [{"role": "user", "content": query}]

                self.engine.load_adapter(domain)
                response = self.engine.generate(
                    messages, max_new_tokens=512, system_prompt=system_prompt
                )
                specialist_responses[domain] = response
            except Exception as e:
                logger.warning(f"Failed to query specialist '{domain}': {e}")
                specialist_responses[domain] = f"[Error: {e}]"

        return self.synthesize(query, specialist_responses)

