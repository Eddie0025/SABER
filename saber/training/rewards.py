import os
import re
import json
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger("GRPORewards")

# --- Local Prometheus 2 Setup ---
PROMETHEUS_MODEL = "prometheus-eval/prometheus-7b-v2.0"
judge_tokenizer = None
judge_model = None

def load_judge_model():
    """Lazy loads the Prometheus 2 judge model in 4-bit to squeeze into the remaining 80GB VRAM alongside the policy model."""
    global judge_tokenizer, judge_model
    if judge_model is None:
        logger.info(f"Loading local LLM Judge: {PROMETHEUS_MODEL} in 4-bit quantization...")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        judge_tokenizer = AutoTokenizer.from_pretrained(PROMETHEUS_MODEL)
        judge_model = AutoModelForCausalLM.from_pretrained(
            PROMETHEUS_MODEL,
            quantization_config=quantization_config,
            device_map="auto"
        )
        judge_model.eval()

def extract_mcq_answer(text: str) -> str:
    text = text.strip()
    patterns = [
        r"(?:Final Answer|ANSWER|Correct Answer|The correct option is|Therefore, the correct answer is|Option)[\s:]*([A-D])\b",
        r"^([A-D])\.",
        r"(?:正确答案是|答案是|选项)[\s:]*([A-D])\b"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
            
    return ""

def is_mcq_prompt(prompt: str) -> bool:
    """Simple heuristic to detect if a prompt is an MCQ."""
    return bool(re.search(r"([A-D])\.\s+", prompt))

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
    
    prompt_text = judge_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = judge_tokenizer(prompt_text, return_tensors="pt").to(judge_model.device)
    
    try:
        with torch.no_grad():
            outputs = judge_model.generate(**inputs, max_new_tokens=150, temperature=0.1)
        
        result_text = judge_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Parse the JSON from the local LLM Judge
        judgement = json.loads(result_text)
        total_score = float(judgement.get("total_score", 0))
        
        # Normalize 0-10 to 0.0-1.0
        return min(max(total_score / 10.0, 0.0), 1.0)
        
    except Exception as e:
        logger.error(f"Local Prometheus inference failed or output invalid JSON: {e}")
        return 0.0

def combined_reward(prompts: List[str], completions: List[str], answer: List[str], **kwargs) -> List[float]:
    """
    Master reward function passed to GRPOTrainer.
    It dynamically applies the MCQ exact-match reward or the Prometheus API reward.
    """
    rewards = []
    
    for p, c, a in zip(prompts, completions, answer):
        if is_mcq_prompt(p):
            # Apply binary MCQ reward
            prediction = extract_mcq_answer(c)
            # If the truth is e.g. "B" and prediction is "B", reward = 1.0
            if prediction == str(a).strip().upper():
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            # Apply Prometheus 2 LLM-Judge reward for open-ended
            score = call_prometheus_judge(p, c, str(a))
            rewards.append(score)
            
    return rewards

# Standard reward logging wrapper for the Trainer
def log_grpo_reward_variance(trainer):
    """
    Hook to log reward variance and KL divergence metrics to TensorBoard/WandB.
    """
    pass # Implementation provided by TRL's built-in callback hooks in modern versions
