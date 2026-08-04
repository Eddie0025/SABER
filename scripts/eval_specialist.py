import os
import sys
import json
import re
import argparse
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from saber.training.dataset_loader import load_specialist_dataset
from saber.training.rewards import call_prometheus_judge

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("EvalGate")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Domains with structured MCQ options
MCQ_DOMAINS = {"cybersecurity", "science", "medical"}

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
            
    return None

def evaluate_mcq(model_obj, tokenizer, dataset, mode_name):
    correct = 0
    unparsed = 0
    total = len(dataset)
    
    if total == 0:
        return 0.0

    logger.info(f"\n--- Evaluation (MCQ): {mode_name} ---")
    for item in tqdm(dataset):
        question = item.get("question", "")
        answers = item.get("answers", item.get("choices", {}))
        truth = str(item.get("solution", item.get("answer", ""))).strip().upper()
        
        prompt_text = f"{question}\n"
        if isinstance(answers, dict):
            for k, v in answers.items():
                prompt_text += f"{k}. {v}\n"
        elif isinstance(answers, list) and len(answers) > 0:
            for idx, choice in enumerate(answers):
                prompt_text += f"{chr(65+idx)}. {choice}\n"
        prompt_text += "Please provide the correct option (e.g. ANSWER: A)."
        
        messages = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model_obj.device)
        with torch.no_grad():
            outputs = model_obj.generate(**inputs, max_new_tokens=60, pad_token_id=tokenizer.eos_token_id)
        
        output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        prediction = extract_mcq_answer(output_text)
        
        if prediction == truth:
            correct += 1
        elif prediction is None:
            unparsed += 1
            
    parsed_total = total - unparsed
    accuracy = (correct / parsed_total) * 100 if parsed_total > 0 else 0.0
    logger.info(f"[{mode_name}] Accuracy: {accuracy:.2f}% ({correct}/{parsed_total}) | Voided: {unparsed}")
    return accuracy

def evaluate_open_ended(model_obj, tokenizer, dataset, mode_name):
    scores = []
    total = min(20, len(dataset))
    if total == 0:
        return 0.0

    logger.info(f"\n--- Evaluation (Prometheus Judge): {mode_name} ---")
    for item in tqdm(dataset.select(range(total))):
        question = item.get("question", "")
        truth = item.get("answer", "")
        
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model_obj.device)
        with torch.no_grad():
            outputs = model_obj.generate(**inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)
        
        output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        score = call_prometheus_judge(question, output_text, truth)
        scores.append(score)
        
    avg_score = (sum(scores) / len(scores)) * 100.0 if scores else 0.0
    logger.info(f"[{mode_name}] Prometheus 2 Score: {avg_score:.2f}%")
    return avg_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specialist", type=str, required=True)
    args = parser.parse_args()
    
    # Structural models (like Orchestrator) skip benchmarks
    structural_models = ["orchestrator", "architecture_planner"]
    if args.specialist in structural_models:
        print(f"RESULT_PAYLOAD:{json.dumps({'is_structural': True, 'pass': True, 'base': 'N/A', 'adapter': 'N/A', 'delta': 'N/A'})}")
        return
        
    try:
        dataset = load_specialist_dataset(args.specialist)
        if dataset is None or len(dataset) == 0:
            print(f"RESULT_PAYLOAD:{json.dumps({'error': 'No dataset available', 'pass': False})}")
            return
            
        adapter_path = f"models/{args.specialist}_v2"
        if not os.path.exists(adapter_path):
            print(f"RESULT_PAYLOAD:{json.dumps({'error': f'Adapter {adapter_path} not found', 'pass': False})}")
            return
            
        # Sample holdout test set (80 for MCQ, 20 for Open-Ended/Code)
        is_mcq = args.specialist in MCQ_DOMAINS
        eval_sample_size = 80 if is_mcq else 20
        eval_dataset = dataset.shuffle(seed=42).select(range(min(eval_sample_size, len(dataset))))
        
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        logger.info(f"Loading Base Model for {args.specialist} Evaluation...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        
        # Wrap with PeftModel
        logger.info(f"Attaching adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        
        # Pass 1: BASE MODEL EVALUATION (disable adapter dynamically)
        with model.disable_adapter():
            if is_mcq:
                base_score = evaluate_mcq(model, tokenizer, eval_dataset, "BASE QWEN")
            else:
                base_score = evaluate_open_ended(model, tokenizer, eval_dataset, "BASE QWEN")
                
        # Pass 2: ADAPTER EVALUATION (adapter active)
        if is_mcq:
            adapter_score = evaluate_mcq(model, tokenizer, eval_dataset, f"{args.specialist.upper()} ADAPTER")
        else:
            adapter_score = evaluate_open_ended(model, tokenizer, eval_dataset, f"{args.specialist.upper()} ADAPTER")
            
        delta = adapter_score - base_score
        pass_gate = delta > 0.0  # Must strictly beat the base model
        
        payload = {
            "base": round(base_score, 2),
            "adapter": round(adapter_score, 2),
            "delta": round(delta, 2),
            "pass": pass_gate,
            "is_structural": False
        }
        
        # Print JSON payload for bash script consumption
        print(f"\nRESULT_PAYLOAD:{json.dumps(payload)}")
        
    except Exception as e:
        logger.error(f"Eval specialist encountered an error: {e}", exc_info=True)
        error_payload = {"error": str(e), "pass": False, "base": "Error", "adapter": "Error", "delta": "Error", "is_structural": False}
        print(f"\nRESULT_PAYLOAD:{json.dumps(error_payload)}")

if __name__ == "__main__":
    main()
