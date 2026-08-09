import re
import logging
from typing import List

logger = logging.getLogger("GRPORewards")

# ============================================================================
# STANDARD GRPO REWARD FUNCTIONS
# ============================================================================
# Following DeepSeek-R1 and Qwen2.5 approach:
# - MCQ: exact match (binary 0/1)
# - Open-ended: keyword/fact overlap against reference answer
# - No LLM judge needed — GRPO learns reasoning quality from outcome rewards
# ============================================================================

def _extract_text(item) -> str:
    """Normalize TRL's various completion formats to a plain string."""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, list):
        return " ".join(
            msg.get("content", "") if isinstance(msg, dict) else str(msg)
            for msg in item
        ).strip()
    if isinstance(item, dict):
        return item.get("content", str(item)).strip()
    return str(item).strip()

def extract_mcq_answer(text: str) -> str:
    """Extract MCQ answer letter (A-D) from generated text."""
    text = text.strip()
    patterns = [
        r"(?:Final Answer|ANSWER|Correct Answer|The correct option is|Therefore, the correct answer is|Option)[\s:]*([A-D])\b",
        r"^([A-D])\.",
        r"\b([A-D])\.\s",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    cleaned = text.strip().upper()
    if cleaned in ["A", "B", "C", "D"]:
        return cleaned
    return ""

def is_mcq_prompt(prompt: str) -> bool:
    """Detect if a prompt contains MCQ options (A. B. C. D.)"""
    return bool(re.search(r"[A-D]\.\s+\S", prompt))

def _fact_overlap_reward(completion: str, reference: str) -> float:
    """
    Standard outcome-based reward for open-ended questions.
    Measures how many key facts from the reference appear in the completion.
    Returns 0.0 to 1.0.
    """
    if not reference or not completion:
        return 0.0
    
    # Extract meaningful words (skip stop words)
    stop = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may",
            "might", "can", "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "and", "but", "or", "not", "no", "so", "it", "its", "this", "that",
            "these", "those", "i", "me", "my", "we", "you", "your", "he", "she", "they",
            "them", "their", "what", "which", "who", "how", "when", "where", "why"}
    
    ref_words = set(reference.lower().split()) - stop
    comp_words = set(completion.lower().split()) - stop
    
    if not ref_words:
        return 0.5  # No key facts to check — neutral
    
    overlap = ref_words & comp_words
    score = len(overlap) / len(ref_words)
    
    # Bonus: penalize extremely short completions (< 5 words = likely garbage)
    if len(completion.split()) < 5:
        score *= 0.5
    
    return min(max(score, 0.0), 1.0)


def combined_reward(prompts, completions, answer=None, **kwargs) -> List[float]:
    """
    Standard GRPO reward function. No LLM judge needed.
    - MCQ: binary exact-match (0.0 or 1.0)
    - Open-ended: fact overlap against reference answer (0.0 to 1.0)
    """
    rewards = []
    
    if answer is None:
        answer = [""] * len(prompts)
    
    for i in range(len(prompts)):
        try:
            p = _extract_text(prompts[i])
            c = _extract_text(completions[i])
            a = str(answer[i]) if i < len(answer) else ""
            
            if is_mcq_prompt(p):
                prediction = extract_mcq_answer(c)
                truth = extract_mcq_answer(a) if len(a) > 1 else a.strip().upper()
                rewards.append(1.0 if prediction and prediction == truth else 0.0)
            else:
                rewards.append(_fact_overlap_reward(c, a))
        except Exception as e:
            logger.warning(f"Reward error for sample {i}: {e}")
            rewards.append(0.0)
            
    return rewards


# Kept for backward compatibility with trainer.py import
def log_grpo_reward_variance(trainer):
    pass
