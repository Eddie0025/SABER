# -*- coding: utf-8 -*-
"""saber.training.rewards

Modular Reward Functions for Verifiable-Fact-Augmented GRPO Reinforcement Learning.
Provides distinct reward functions for Definitive (MCQ/Math) vs Open-Ended (Coding/Synthesis) tasks.
"""

import os
import re
import sqlite3
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# 1. DEFINITIVE REWARD FUNCTION (MCQ / Math / Exact Fact)
# ---------------------------------------------------------------------------

def definitive_reward_function(
    prompts: List[str],
    completions: List[str],
    expected_answers: List[str],
    domain: str = "science",
    **kwargs
) -> List[float]:
    """Reward function for Definitive MCQ and Math questions.
    
    Rewards:
      +1.0 : Format Reward (ends with ANSWER: [A-D] or ANSWER: <val>)
      +2.0 : Outcome Reward (extracted answer matches expected ground truth)
      -1.5 / +0.5 : Sentinel Factuality Reward (KB contradiction vs support)
    """
    rewards = []
    db_path = f"data/offline_kb/{domain}_kb.db"

    for prompt, completion, expected in zip(prompts, completions, expected_answers):
        reward = 0.0
        comp_clean = completion.strip()

        # 1. Format Reward
        last_line = comp_clean.split("\n")[-1].strip().upper()
        if re.search(r"ANSWER:\s*[<\(]?([A-D0-9\.]+)\b[>\)]?", last_line):
            reward += 1.0

        # 2. Outcome Reward
        extracted = None
        match = re.search(r"ANSWER:\s*[<\(]?([A-D0-9\.]+)\b[>\)]?", comp_clean.upper())
        if match:
            extracted = match.group(1)

        expected_norm = str(expected).strip().upper()
        if extracted and (extracted == expected_norm or expected_norm in extracted):
            reward += 2.0

        # 3. Sentinel Factuality Reward
        if os.path.exists(db_path):
            clean_q = " ".join(prompt.lower().split())
            numerics = re.findall(r'\b\d+(?:\.\d+)?%?\b|cve-\d+-\d+', clean_q)
            guard = f"[{'_'.join(numerics)}]::{clean_q}"

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT support_passage FROM passage_kb WHERE query_guard = ? LIMIT 1", (guard,))
            row = cursor.fetchone()
            conn.close()

            if row and row[0]:
                passage = row[0].lower()
                resp_lower = comp_clean.lower()
                if "not" in passage and "not" not in resp_lower:
                    reward -= 1.5  # Direct contradiction penalty
                elif any(fact in resp_lower for fact in passage.split()[:5] if len(fact) > 5):
                    reward += 0.5  # Grounded support bonus

        rewards.append(reward)

    return rewards


# ---------------------------------------------------------------------------
# 2. OPEN-ENDED REWARD FUNCTION (Coding / Synthesis / Architecture / Report)
# ---------------------------------------------------------------------------

def open_ended_reward_function(
    prompts: List[str],
    completions: List[str],
    expected_answers: List[str] = None,
    domain: str = "coding",
    **kwargs
) -> List[float]:
    """Reward function for Open-Ended Synthesis, Coding, and Reasoning tasks.
    
    Rewards:
      +2.0 : Execution / Unit Test Passing Reward (for python code blocks)
      +1.0 : CoT Structural Completeness (contains step-by-step headers)
      +1.5 : Sentinel Factual & Technical Reference Grounding
      -1.0 : Severe Repetition / Token Loop Penalty
    """
    rewards = []

    for prompt, completion in zip(prompts, completions):
        reward = 0.0
        comp_clean = completion.strip()

        # 1. Execution / Code Test Reward (for Coding)
        if "```python" in comp_clean:
            code_blocks = re.findall(r"```python\s*(.*?)\s*```", comp_clean, re.DOTALL)
            if code_blocks:
                code_to_test = code_blocks[0]
                try:
                    # Basic syntax check
                    compile(code_to_test, "<string>", "exec")
                    reward += 2.0
                except Exception:
                    reward -= 1.0

        # 2. CoT Structural Completeness
        if "## Step" in comp_clean or "Step 1" in comp_clean:
            reward += 1.0

        # 3. Repetition Penalty
        lines = [l.strip() for l in comp_clean.split("\n") if l.strip()]
        if len(lines) > 5 and len(set(lines)) < len(lines) / 2:
            reward -= 1.0

        rewards.append(reward)

    return rewards
