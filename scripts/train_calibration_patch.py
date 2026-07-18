import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import PeftModel, LoraConfig, get_peft_model
from datasets import Dataset
from trl import SFTTrainer
import json

def format_chatml(records):
    formatted = []
    for r in records:
        text = r.get("text", "")
        label = r.get("label", "")
        formatted_text = (
            f"<|im_start|>system\n"
            f"You are a highly skilled Medical AI specialist. Provide a thorough, accurate, and evidence-based clinical answer. "
            f"When asked to name a specific test, marker, sign, or triad component, only state it if you are highly confident it is a real, "
            f"established clinical term. If you are not certain of the exact name, say 'I am not fully certain of the specific term, "
            f"but the relevant clinical concept is...' rather than generating a plausible-sounding but unverified specific name. "
            f"Do not invent, guess, or fabricate precise terminology.<|im_end|>\n"
            f"<|im_start|>user\n{text}<|im_end|>\n"
            f"<|im_start|>assistant\n{label}<|im_end|>\n"
        )
        formatted.append({"text": formatted_text})
    return formatted

def main():
    print("=========================================================")
    print(" Running Targeted Medical Specialist Calibration Patch")
    print("=========================================================")
    
    base_model_name = "Qwen/Qwen2.5-7B-Instruct"
    adapter_path = "models/medical_v2"
    patch_data_path = "data/processed/medical_calibration_patch.jsonl"
    
    if not os.path.exists(adapter_path):
        print(f"Error: Existing adapter checkpoint '{adapter_path}' not found. Cannot perform continued fine-tuning.")
        sys.exit(1)
        
    print(f"Loading patch dataset from {patch_data_path}...")
    records = []
    with open(patch_data_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    formatted_data = format_chatml(records)
    dataset = Dataset.from_list(formatted_data)
    
    print(f"Loading base model: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Loading existing adapter for continued training: {adapter_path}...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=True
    )
    
    training_args = TrainingArguments(
        output_dir=adapter_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=10,
        num_train_epochs=1,
        bf16=True,
        save_strategy="no",
        report_to="none"
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        packing=False,
        args=training_args
    )
    
    print("Starting fine-tuning...")
    trainer.train()
    
    print(f"Saving calibrated adapter back to {adapter_path}...")
    model.save_pretrained(adapter_path)
    print("[+] Done! Calibration patch training complete.")

if __name__ == "__main__":
    main()
