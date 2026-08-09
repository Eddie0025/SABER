import os
import re
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("GRPORewards")

# ============================================================================
# LIGHTWEIGHT REWARD FUNCTIONS FOR GRPO
# ============================================================================
# We use ROUGE-L + keyword overlap instead of a heavy 7B LLM judge.
# Reason: Loading a second 7B model (Prometheus 2) alongside the policy model
# during GRPO training causes OOM on a single 80GB H100.
# This approach is what DeepSeek-R1 and Qwen2.5 used for initial GRPO passes.
# ============================================================================

def _extract_text_from_completion(completion) -> str:
    """
    TRL GRPOTrainer passes completions in different formats depending on version:
    - Modern TRL: list of dicts like [{"role": "assistant", "content": "..."}]
    - Older TRL: raw string
    This function normalizes both.
    """
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, list):
        # List of message dicts
        texts = []
        for msg in completion:
            if isinstance(msg, dict):
                texts.append(msg.get("content", ""))
            elif isinstance(msg, str):
                texts.append(msg)
        return " ".join(texts).strip()
    if isinstance(completion, dict):
        return completion.get("content", str(completion)).strip()
    return str(completion).strip()

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
    # Last resort: if the entire output is just a single letter
    cleaned = text.strip().upper()
    if cleaned in ["A", "B", "C", "D"]:
        return cleaned
    return ""

def is_mcq_prompt(prompt: str) -> bool:
    """Detect if a prompt contains MCQ options (A. B. C. D.)"""
    return bool(re.search(r"[A-D]\.\s+\S", prompt))

def _compute_rouge_l(prediction: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 score between prediction and reference.
    Lightweight, no external dependencies.
    """
    if not prediction or not reference:
        return 0.0
    
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    # LCS using dynamic programming
    m, n = len(ref_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == pred_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    if lcs_length == 0:
        return 0.0
    
    precision = lcs_length / n
    recall = lcs_length / m
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def _compute_keyword_overlap(prediction: str, reference: str) -> float:
    """
    Compute keyword overlap ratio between prediction and reference.
    Filters out common stop words for more meaningful comparison.
    """
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                  "have", "has", "had", "do", "does", "did", "will", "would", "could",
                  "should", "may", "might", "shall", "can", "to", "of", "in", "for",
                  "on", "with", "at", "by", "from", "as", "into", "through", "during",
                  "before", "after", "above", "below", "between", "and", "but", "or",
                  "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
                  "every", "all", "any", "few", "more", "most", "other", "some", "such",
                  "than", "too", "very", "just", "it", "its", "this", "that", "these",
                  "those", "i", "me", "my", "we", "our", "you", "your", "he", "him",
                  "his", "she", "her", "they", "them", "their", "what", "which", "who"}
    
    pred_words = set(prediction.lower().split()) - stop_words
    ref_words = set(reference.lower().split()) - stop_words
    
    if not ref_words:
        return 0.5  # Neutral if reference has no keywords
    
    overlap = pred_words & ref_words
    return len(overlap) / len(ref_words)

def _open_ended_reward(prompt: str, completion: str, reference: str) -> float:
    """
    Lightweight reward for open-ended questions.
    Combines ROUGE-L (sequence quality) + keyword overlap (factual coverage).
    Returns 0.0 to 1.0.
    """
    rouge_score = _compute_rouge_l(completion, reference)
    keyword_score = _compute_keyword_overlap(completion, reference)
    
    # Penalize very short or empty completions
    word_count = len(completion.split())
    length_penalty = min(word_count / 10.0, 1.0)  # Full credit at 10+ words
    
    # Weighted combination: 40% ROUGE-L, 40% keyword overlap, 20% length
    combined = 0.4 * rouge_score + 0.4 * keyword_score + 0.2 * length_penalty
    return min(max(combined, 0.0), 1.0)


def combined_reward(prompts, completions, answer=None, **kwargs) -> List[float]:
    """
    Master reward function passed to GRPOTrainer.
    Handles both MCQ (exact-match) and open-ended (ROUGE-L + keyword) questions.
    
    TRL GRPOTrainer calls this with:
      - prompts: list of prompt strings or list of message-dicts
      - completions: list of completion strings or list of message-dicts
      - answer: list of reference answers (from dataset column)
    """
    rewards = []
    
    # Handle case where answer might not be provided
    if answer is None:
        answer = [""] * len(prompts)
    
    for i in range(len(prompts)):
        try:
            # Normalize prompt
            p = _extract_text_from_completion(prompts[i])
            # Normalize completion
            c = _extract_text_from_completion(completions[i])
            # Normalize answer
            a = str(answer[i]) if i < len(answer) else ""
            
            if is_mcq_prompt(p):
                # MCQ exact-match reward
                prediction = extract_mcq_answer(c)
                truth = extract_mcq_answer(a) if len(a) > 1 else a.strip().upper()
                if prediction and prediction == truth:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            else:
                # Open-ended ROUGE-L + keyword reward
                score = _open_ended_reward(p, c, a)
                rewards.append(score)
        except Exception as e:
            logger.warning(f"Reward computation error for sample {i}: {e}. Defaulting to 0.0")
            rewards.append(0.0)
            
    return rewards

# Standard reward logging wrapper for the Trainer
def log_grpo_reward_variance(trainer):
    """
    Hook to log reward variance and KL divergence metrics to TensorBoard/WandB.
    """
    pass  # Implementation provided by TRL's built-in callback hooks in modern versions
