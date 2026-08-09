import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger("GRPORewards")

# --- Local Prometheus 2 Setup ---
PROMETHEUS_MODEL = "prometheus-eval/prometheus-7b-v2.0"
judge_tokenizer = None
judge_model = None

def load_judge_model():
    """Lazy loads the Prometheus 2 judge model on GPU in 4-bit alongside the policy model."""
    global judge_tokenizer, judge_model
    if judge_model is None:
        logger.info(f"Loading local LLM Judge: {PROMETHEUS_MODEL} on GPU (4-bit)...")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        judge_tokenizer = AutoTokenizer.from_pretrained(PROMETHEUS_MODEL)
        if judge_tokenizer.pad_token is None:
            judge_tokenizer.pad_token = judge_tokenizer.eos_token
            
        judge_model = AutoModelForCausalLM.from_pretrained(
            PROMETHEUS_MODEL,
            quantization_config=quantization_config,
            device_map="auto"
        )
        judge_model.eval()
        logger.info("Prometheus 2 judge loaded on GPU (4-bit) successfully.")

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

def call_prometheus_judge(prompt: str, completion: str, reference: str) -> float:
    """
    Calls the locally loaded Prometheus 2 model to act as an LLM Judge for open-ended reasoning.
    Returns a normalized reward between 0.0 and 1.0.
    """
    load_judge_model()
    
    rubric = (
        "Evaluate the response based on Technical Correctness (0-5), "
        "Completeness (0-3), and Hallucinations (0-2). "
        "A response with hallucinations must receive 0 for that section. "
        "Provide your evaluation output strictly as a JSON object with keys: "
        "'correctness', 'completeness', 'hallucinations', 'total_score'. "
        "Ensure 'total_score' is out of 10."
    )
    
    sys_prompt = "You are Prometheus, a rigorous technical evaluator. Follow the rubric perfectly."
    user_prompt = f"### Instruction:\n{prompt}\n\n### Response to Evaluate:\n{completion}\n\n### Reference Fact/Answer:\n{reference}\n\n### Rubric:\n{rubric}"
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        prompt_text = judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback if the tokenizer doesn't support chat templates
        prompt_text = f"{sys_prompt}\n\n{user_prompt}\n\nEvaluation:"
    
    inputs = judge_tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048).to(judge_model.device)
    
    try:
        with torch.no_grad():
            outputs = judge_model.generate(**inputs, max_new_tokens=150, temperature=0.1, do_sample=False)
        
        result_text = judge_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Robust JSON parsing from local LLM Judge output (handles markdown code fences)
        json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
        if json_match:
            judgement = json.loads(json_match.group(0))
        else:
            judgement = json.loads(result_text)
            
        total_score = float(judgement.get("total_score", judgement.get("score", 0)))
        
        # Normalize 0-10 to 0.0-1.0
        return min(max(total_score / 10.0, 0.0), 1.0)
        
    except Exception as e:
        logger.warning(f"Prometheus judge parsing failed: {e}. Falling back to 0.5 neutral reward.")
        return 0.5  # Neutral reward on parse failure, not 0.0 (which would punish the model unfairly)


def combined_reward(prompts, completions, answer=None, **kwargs) -> List[float]:
    """
    Master reward function passed to GRPOTrainer.
    Handles both MCQ (exact-match) and open-ended (Prometheus 2 LLM Judge) questions.
    
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
            # Normalize all inputs to plain strings
            p = _extract_text_from_completion(prompts[i])
            c = _extract_text_from_completion(completions[i])
            a = str(answer[i]) if i < len(answer) else ""
            
            if is_mcq_prompt(p):
                # MCQ exact-match reward (binary)
                prediction = extract_mcq_answer(c)
                truth = extract_mcq_answer(a) if len(a) > 1 else a.strip().upper()
                if prediction and prediction == truth:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            else:
                # Open-ended: Prometheus 2 LLM Judge reward
                score = call_prometheus_judge(p, c, a)
                rewards.append(score)
        except Exception as e:
            logger.warning(f"Reward computation error for sample {i}: {e}. Defaulting to 0.5")
            rewards.append(0.5)
            
    return rewards

# Standard reward logging wrapper for the Trainer
def log_grpo_reward_variance(trainer):
    """
    Hook to log reward variance and KL divergence metrics to TensorBoard/WandB.
    """
    pass  # Implementation provided by TRL's built-in callback hooks in modern versions
