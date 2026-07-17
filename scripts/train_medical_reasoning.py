import os
import sys
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Make sure we can import the trainer from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from saber.training.trainer import TrainConfig, train

def prep_dataset():
    print("=========================================================")
    print(" Downloading FreedomIntelligence/medical-o1-reasoning-SFT")
    print("=========================================================")
    
    # Load the reasoning dataset from HuggingFace
    try:
        dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Make sure you have the datasets library installed: pip install datasets")
        sys.exit(1)
        
    print(f"Loaded {len(dataset)} records.")
    
    # We will sample ~3000 records as requested by the user
    target_count = min(3500, len(dataset))
    sampled_dataset = dataset.select(range(target_count))
    
    output_path = "data/processed/medical_reasoning.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Formatting {target_count} records to {output_path}...")
    
    with open(output_path, "w") as f:
        for row in sampled_dataset:
            # The exact keys depend on the HF dataset schema, but typical CoT datasets use these:
            question = row.get("Question") or row.get("instruction") or row.get("question") or ""
            cot = row.get("Complex_CoT") or row.get("cot") or row.get("reasoning") or ""
            response = row.get("Response") or row.get("response") or row.get("answer") or ""
            
            # If the dataset has a single 'Response' that already contains the CoT, use that
            if cot:
                label = f"REASONING:\n{cot}\n\nCONCLUSION:\n{response}"
            else:
                label = response
                
            record = {
                "text": question,
                "label": label
            }
            f.write(json.dumps(record) + "\n")
            
    print("Dataset preparation complete.")
    return output_path

def merge_adapter(base_model_name, adapter_path, merged_output_path):
    print("=========================================================")
    print(" Merging existing medical_v2 adapter into base model")
    print("=========================================================")
    
    if os.path.exists(merged_output_path):
        print(f"Merged model already exists at {merged_output_path}, skipping merge.")
        return
        
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu", # Load on CPU to avoid VRAM OOM during merge
        trust_remote_code=True
    )
    
    print(f"Applying adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Merging weights (this may take a few minutes)...")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to {merged_output_path}...")
    model.save_pretrained(merged_output_path)
    tokenizer.save_pretrained(merged_output_path)
    
    print("Merge complete! Freeing memory...")
    del model
    del base_model
    import gc
    gc.collect()

def main():
    # 1. Download and format the dataset
    data_path = prep_dataset()
    
    # 2. Train directly from the clean base model to avoid carrying over any Chat Doctor style
    base_model = "Qwen/Qwen2.5-7B-Instruct"
    merged_model = base_model
    print("Training directly from clean base model to avoid Chat Doctor contamination.")
        
    # 3. Train the model (medical_v3) using the existing trainer architecture
    print("=========================================================")
    print(" Commencing Phase 2 Training (medical_v3)")
    print("=========================================================")
    
    cfg = TrainConfig(
        domain="medical",
        data_path=data_path,
        base_model=merged_model,
        output_dir="models/medical_v3",
        epochs=3, # 3 epochs over 3500 samples
        batch_size=8,
        learning_rate=1e-5, # Lower learning rate since it's already fine-tuned
        lora_r=16,
        gpu_id=0,
        packing=True,
        max_seq_length=2048
    )
    
    train(cfg)
    print("=========================================================")
    print(" Training Complete! Medical V3 model is ready.")
    print("=========================================================")

if __name__ == "__main__":
    main()
