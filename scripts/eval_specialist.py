import os
import json
import re
import argparse
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from saber.training.dataset_loader import load_specialist_dataset

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("EvalGate")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

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

    logger.info(f"\n--- Evaluation: {mode_name} ---")
    for item in tqdm(dataset):
        # We assume the benchmark datasets have "question", "answers"/"choices", and "solution"/"answer"
        question = item.get("question", "")
        answers = item.get("answers", item.get("choices", {}))
        truth = item.get("solution", item.get("answer", ""))
        
        prompt_text = f"{question}\n"
        if isinstance(answers, dict):
            for k, v in answers.items():
                prompt_text += f"{k}. {v}\n"
        elif isinstance(answers, list):
            for idx, choice in enumerate(answers):
                prompt_text += f"{chr(65+idx)}. {choice}\n"
        prompt_text += "Please provide the correct option."
        
        messages = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model_obj.device)
        with torch.no_grad():
            outputs = model_obj.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
        
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specialist", type=str, required=True)
    args = parser.parse_args()
    
    # Identify if structural model
    structural_models = ["architecture_planner", "orchestrator"]
    if args.specialist in structural_models:
        print(f"RESULT_PAYLOAD:{json.dumps({'is_structural': True})}")
        return
        
    dataset = load_specialist_dataset(args.specialist)
    if dataset is None:
        print(f"RESULT_PAYLOAD:{json.dumps({'error': 'No dataset'})}")
        return
        
    # We take exactly 80 samples for the eval gate, to match CyberMetric-80 scale
    eval_dataset = dataset.shuffle(seed=42).select(range(min(80, len(dataset))))
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    
    base_acc = evaluate_mcq(model, tokenizer, eval_dataset, "BASE QWEN")
    
    adapter_path = f"models/{args.specialist}_v2"
    if not os.path.exists(adapter_path):
        print(f"RESULT_PAYLOAD:{json.dumps({'error': 'Adapter not found'})}")
        return
        
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    
    adapter_acc = evaluate_mcq(model, tokenizer, eval_dataset, f"{args.specialist.upper()} ADAPTER")
    
    delta = adapter_acc - base_acc
    pass_gate = delta > 0.0  # Must strictly beat the base model
    
    # For now, we stub the open-ended check to just output the MCQ results, 
    # as the open-ended Prometheus API check is built inside rewards.py for GRPO.
    
    payload = {
        "base": round(base_acc, 2),
        "adapter": round(adapter_acc, 2),
        "delta": round(delta, 2),
        "pass": pass_gate,
        "is_structural": False
    }
    
    # Must be exactly this prefix so bash script can grep it
    print(f"\nRESULT_PAYLOAD:{json.dumps(payload)}")

if __name__ == "__main__":
    main()
