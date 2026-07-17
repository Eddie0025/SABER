import os
import sys
import json
from datasets import load_dataset

def main():
    print("=========================================================")
    print(" Preparing Clean Medical Reasoning Dataset")
    print("=========================================================")
    
    try:
        dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en", split="train")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)
        
    print(f"Loaded {len(dataset)} records from medical-o1.")
    
    # Target count is ~3,500 samples
    target_count = min(3500, len(dataset))
    sampled_dataset = dataset.select(range(target_count))
    
    output_path = "data/processed/medical.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Formatting and saving to {output_path}...")
    
    with open(output_path, "w") as f:
        for row in sampled_dataset:
            question = row.get("Question") or row.get("instruction") or row.get("question") or ""
            cot = row.get("Complex_CoT") or row.get("cot") or row.get("reasoning") or ""
            response = row.get("Response") or row.get("response") or row.get("answer") or ""
            
            if cot:
                label = f"REASONING:\n{cot}\n\nCONCLUSION:\n{response}"
            else:
                label = response
                
            record = {
                "text": question,
                "label": label
            }
            f.write(json.dumps(record) + "\n")
            
    print("Done! Clean medical dataset created successfully.")

if __name__ == "__main__":
    main()
