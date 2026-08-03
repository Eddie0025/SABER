import os
import json
import re
import urllib.request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("EvalCyberMetric")

CYBERMETRIC_URL = "https://raw.githubusercontent.com/CyberMetric/CyberMetric/main/CyberMetric-80-v1.json"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "models/cybersecurity_v2"

def download_dataset():
    local_path = "CyberMetric-80-v1.json"
    if not os.path.exists(local_path):
        logger.info(f"Downloading {CYBERMETRIC_URL}...")
        urllib.request.urlretrieve(CYBERMETRIC_URL, local_path)
    with open(local_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_answer(text: str) -> str:
    """
    Strict regex grader according to architecture.md specification.
    Returns A, B, C, D or None if no strict match is found.
    """
    text = text.strip()
    
    # Common English patterns
    patterns = [
        r"(?:Final Answer|ANSWER|Correct Answer|The correct option is|Therefore, the correct answer is)[\s:]*([A-D])\b",
        r"^([A-D])\.",
        r"(?:正确答案是|答案是|选项)[\s:]*([A-D])\b" # Chinese patterns
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
            
    # Fallback checking end of text
    match = re.search(r"([A-D])\.$", text)
    if match:
        return match.group(1).upper()
        
    return None

def run_evaluation():
    dataset = download_dataset()
    logger.info(f"Loaded {len(dataset)} questions from CyberMetric-80.")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    logger.info("Loading Base Model in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    def evaluate_model(model_obj, mode_name):
        correct = 0
        unparsed = 0
        total = len(dataset)
        
        logger.info(f"\n--- Starting Evaluation: {mode_name} ---")
        for item in tqdm(dataset):
            question = item.get("question", "")
            choices = item.get("choices", [])
            truth = item.get("answer", "")
            
            prompt_text = f"{question}\n"
            for choice in choices:
                prompt_text += f"{choice}\n"
            prompt_text += "Please provide the correct option."
            
            messages = [{"role": "user", "content": prompt_text}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model_obj.device)
            with torch.no_grad():
                outputs = model_obj.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
            
            output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            prediction = extract_answer(output_text)
            
            if prediction == truth:
                correct += 1
            elif prediction is None:
                unparsed += 1
                
        accuracy = (correct / total) * 100
        logger.info(f"[{mode_name}] Accuracy: {accuracy:.2f}% ({correct}/{total}) | Unparsed: {unparsed}")
        return accuracy

    # Run Base Model
    base_acc = evaluate_model(model, "BASE QWEN")
    
    # Load and run Adapter
    logger.info(f"\nLoading DoRA Adapter from {ADAPTER_PATH}...")
    try:
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        adapter_acc = evaluate_model(model, "CYBER ADAPTER")
        
        logger.info("\n=== FINAL RESULTS ===")
        logger.info(f"Base Score:    {base_acc:.2f}%")
        logger.info(f"Adapter Score: {adapter_acc:.2f}%")
        delta = adapter_acc - base_acc
        logger.info(f"Delta:         {delta:+.2f}%")
    except Exception as e:
        logger.error(f"Failed to load adapter: {e}")

if __name__ == "__main__":
    run_evaluation()
