import uuid
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path

from saber.specialist import SpecialistEngine
from saber.coding.shared_memory import SharedMemory
from saber.cot_maintainer import CoTMaintainer
from saber.config import DOMAIN_SYSTEM_PROMPTS

logger = logging.getLogger("SABER_CodingPlanner")

# Language specialists available in the coding sector
LANGUAGE_SPECIALISTS = {
    "python": "python",
    "javascript": "javascript",
    "sql": "sql",
}

PLANNER_SYSTEM_PROMPT = """You are the SABER Coding Sector Planner. You decompose complex coding tasks into sub-tasks.

Given a coding query, respond with a JSON plan in this exact format:
{
    "subtasks": [
        {
            "id": "st_001",
            "language": "python",
            "description": "What this subtask should accomplish",
            "depends_on": []
        }
    ]
}

Available languages: python, javascript, sql
Only output the JSON, nothing else."""

SPECIALIST_PROMPT_TEMPLATE = """You are a {language} programming specialist working as part of a team.

PROJECT PLAN:
{plan}

YOUR ASSIGNED TASK:
{task_description}

{other_code}
{other_thoughts}

Write your code, then write unit tests for your code.
Provide your reasoning and any notes for other team members.

Format your response as:
```{language}
<your code here>
```

TESTS:
```{language}
<your tests here>
```

NOTES:
<your reasoning and notes here>
"""


class CodingPlanner:
    """
    Mini-orchestrator for the Coding Sector.
    
    Receives a coding query from the main Orchestrator, decomposes it into
    sub-tasks with language tags, dispatches them sequentially to language
    specialists (hot-swapping adapters), and assembles the final response.
    """

    def __init__(self, engine: SpecialistEngine):
        self.engine = engine
        self.memory = SharedMemory()
        self.cot = CoTMaintainer()

    def process(self, query: str) -> str:
        """
        Full coding sector pipeline:
        1. Decompose query into sub-tasks
        2. Dispatch each sub-task to the appropriate language specialist
        3. Assemble and return the final code
        """
        task_id = f"code_{uuid.uuid4().hex[:8]}"
        logger.info(f"[{task_id}] Coding Planner processing: {query[:80]}...")

        # Step 1: Decompose into sub-tasks using the bare base model
        plan = self._decompose(query)
        if not plan:
            # If decomposition fails, just route to python specialist directly
            logger.warning(f"[{task_id}] Plan decomposition failed. Routing directly to Python specialist.")
            return self._direct_generate(query, "python")

        # Store plan in shared memory
        self.memory.set(f"{task_id}_plan", plan)

        # Step 2: Execute sub-tasks in dependency order
        results = []
        for subtask in plan.get("subtasks", []):
            lang = subtask.get("language", "python")
            desc = subtask.get("description", query)
            st_id = subtask.get("id", "st_000")

            if lang not in LANGUAGE_SPECIALISTS:
                logger.warning(f"Unknown language '{lang}', defaulting to python.")
                lang = "python"

            logger.info(f"[{task_id}] Dispatching {st_id} to {lang} specialist...")
            result = self._execute_subtask(task_id, st_id, lang, desc, plan)
            results.append({"subtask_id": st_id, "language": lang, "output": result})

            # Store result in shared memory
            self.memory.set(f"{task_id}_{st_id}", {"language": lang, "output": result})

        # Step 3: Assemble final response
        assembled = self._assemble(results)
        return assembled

    def _decompose(self, query: str) -> Optional[Dict]:
        """Use the bare base model to decompose the query into sub-tasks."""
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        raw = self.engine.generate_bare(messages, max_new_tokens=512, temperature=0.0)

        # Try to parse JSON from the response
        try:
            # Find JSON in the response
            json_match = raw[raw.find("{"):raw.rfind("}") + 1]
            if json_match:
                return json.loads(json_match)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse planner JSON: {e}")

        return None

    def _execute_subtask(self, task_id: str, st_id: str, language: str, description: str, plan: Dict) -> str:
        """Execute a single sub-task using the language specialist adapter."""
        # Gather context from other completed subtasks
        other_code = ""
        other_thoughts = ""

        plan_str = json.dumps(plan, indent=2) if plan else "No plan available."

        prompt = SPECIALIST_PROMPT_TEMPLATE.format(
            language=language,
            plan=plan_str,
            task_description=description,
            other_code=other_code if other_code else "No code from other team members yet.",
            other_thoughts=other_thoughts if other_thoughts else "No notes from other team members yet.",
        )

        messages = [{"role": "user", "content": prompt}]

        # Use the language specialist adapter
        domain = LANGUAGE_SPECIALISTS[language]
        system_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, f"You are a {language} programming specialist.")

        try:
            self.engine.load_adapter(domain)
            result = self.engine.generate(messages, max_new_tokens=1024, system_prompt=system_prompt)
        except FileNotFoundError:
            logger.warning(f"Adapter for {domain} not found. Using bare model.")
            result = self.engine.generate_bare(messages, max_new_tokens=1024)

        return result

    def _direct_generate(self, query: str, language: str) -> str:
        """Fallback: directly generate code without decomposition."""
        domain = LANGUAGE_SPECIALISTS.get(language, "python")
        system_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, f"You are a {language} programming specialist.")

        messages = [{"role": "user", "content": query}]
        try:
            self.engine.load_adapter(domain)
            return self.engine.generate(messages, max_new_tokens=1024, system_prompt=system_prompt)
        except FileNotFoundError:
            return self.engine.generate_bare(messages, max_new_tokens=1024)

    def _assemble(self, results: List[Dict]) -> str:
        """Assemble all sub-task results into a single coherent response."""
        if len(results) == 1:
            return results[0]["output"] + "\n\n⚡ SABER Coding Sector"

        assembled = "# SABER Coding Sector — Multi-Language Output\n\n"
        for r in results:
            assembled += f"## {r['language'].upper()} ({r['subtask_id']})\n\n"
            assembled += r["output"] + "\n\n---\n\n"

        assembled += "⚡ SABER Coding Sector"
        return assembled
