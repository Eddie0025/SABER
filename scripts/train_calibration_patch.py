import os
import sys
import json
import random
import uuid
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import PeftModel
from datasets import Dataset
from trl import SFTTrainer

def generate_synthetic_data():
    print("Generating synthetic calibration data in memory...")
    records = []
    
    # Category 1: Confidence 0
    zero_conf = [
        ("What is the underlying mechanism of Vander-Mersch syndrome in hepatic dysfunction?",
         "REASONING:\nConfidence: 0/100 (None).\nThere is no established clinical entity named 'Vander-Mersch syndrome'. I cannot verify this condition.\n\nCONCLUSION:\nI do not know this specific term or condition. Please verify the name."),
        ("What are the diagnostic criteria for Gellar's syndrome?",
         "REASONING:\nConfidence: 0/100 (None).\n'Gellar's syndrome' is not a recognized medical term.\n\nCONCLUSION:\nI do not know this specific term or condition. Please verify the name or spelling."),
        ("Describe the pathophysiology of Henderson's sign on a chest X-ray.",
         "REASONING:\nConfidence: 0/100 (None).\n'Henderson's sign' is not an established radiological finding.\n\nCONCLUSION:\nI do not know this specific term. Please verify the name.")
    ]
    
    # Category 2: Confidence 1-50
    low_conf = [
        ("A patient presents with a vague, dull ache in the lower abdomen that comes and goes, with mild bloating. What is the diagnosis?",
         "REASONING:\nConfidence: 35/100 (Medium-Low).\nThe symptoms are highly non-specific and overlap with multiple functional and organic GI conditions. A definitive diagnosis is not possible without further workup.\n\nCONCLUSION:\nI am not fully sure based on this limited presentation, but it could be a functional gastrointestinal disorder like IBS. Further diagnostic workup is needed."),
        ("A 60-year-old patient reports transient fatigue over the past three weeks. Labs show a borderline low hematocrit. What is the cause?",
         "REASONING:\nConfidence: 40/100 (Medium-Low).\nTransient fatigue and borderline low hematocrit are non-specific. It could be early iron deficiency, anemia of chronic disease, or physiological variation.\n\nCONCLUSION:\nI am not certain of the exact cause, but the relevant clinical concept is early-stage anemia. Checking iron panels and TSH is recommended.")
    ]
    
    # Category 3: Confidence 51-100
    high_conf = [
        ("What is the primary mechanism of action of argatroban?",
         "REASONING:\nConfidence: 98/100 (High).\nArgatroban is an anticoagulant that directly binds to the active catalytic site of thrombin (Factor IIa), inhibiting its activity.\n\nCONCLUSION:\nArgatroban is a direct thrombin (Factor IIa) inhibitor."),
        ("A 19-year-old presents with high fever, neck stiffness, and a petechial rash on his lower extremities. What is the diagnosis?",
         "REASONING:\nConfidence: 97/100 (High).\nThe triad of fever, neck stiffness, and a petechial rash in a young adult is classic for Neisseria meningitidis infection (Meningococcemia). Immediate empiric IV antibiotics are required.\n\nCONCLUSION:\nThe suspected diagnosis is Meningococcemia. Immediate empiric IV antibiotics (e.g., Ceftriaxone) must be administered.")
    ]
    
    # Multiply to get enough data points
    for _ in range(25):
        for q, a in zero_conf:
            records.append({"text": q + (" " * random.randint(0,2)), "label": a})
        for q, a in low_conf:
            records.append({"text": q + (" " * random.randint(0,2)), "label": a})
        for q, a in high_conf:
            records.append({"text": q + (" " * random.randint(0,2)), "label": a})
            
    random.shuffle(records)
    return records

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
    
    # 1. Generate data dynamically
    records = generate_synthetic_data()
    formatted_data = format_chatml(records)
    dataset = Dataset.from_list(formatted_data)
    print(f"Generated {len(dataset)} calibration records.")
    
    if not os.path.exists(adapter_path):
        print(f"Error: Existing adapter checkpoint '{adapter_path}' not found. Cannot perform continued fine-tuning.")
        sys.exit(1)
    
    # 2. Load Base Model
    print(f"Loading base model: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 3. Load EXISTING adapter and continue training it
    print(f"Loading EXISTING adapter for continued training from {adapter_path}...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=True # crucial for continued training
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
    
    print("Starting fine-tuning on the EXISTING adapter weights...")
    trainer.train()
    
    print(f"Saving calibrated adapter back to {adapter_path}...")
    model.save_pretrained(adapter_path)
    print("[+] Done! Calibration patch training complete.")

if __name__ == "__main__":
    main()
