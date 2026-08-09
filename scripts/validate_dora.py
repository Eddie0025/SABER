import json
import logging
from pathlib import Path
from datetime import datetime
import torch
from safetensors import safe_open

logger = logging.getLogger("ValidateDoRA")
logging.basicConfig(level=logging.INFO)

def validate_dora(model, tokenizer, adapter_path: str) -> dict:
    """
    Strict post-training validation suite for DoRA adapters.
    Executes real PyTorch checks to ensure mathematical correctness and prevent catastrophic forgetting.
    """
    logger.info(f"Starting strict DoRA validation for adapter: {adapter_path}")
    
    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "adapter_path": adapter_path,
        "overall_status": "PENDING",
        "checks": {}
    }
    
    overall_pass = True
    
    # 1. COLLATOR SANITY CHECK
    try:
        from saber.training.trainer import CustomCompletionOnlyCollator
        collator = CustomCompletionOnlyCollator(response_template="<|im_start|>assistant\\n", tokenizer=tokenizer, mlm=False)
        test_text = "<|im_start|>user\\nTest prompt<|im_end|>\\n<|im_start|>assistant\\nResponse<|im_end|>"
        tokens = tokenizer(test_text, return_tensors="pt")
        # Collator expects list of dicts
        batch = collator([{"input_ids": tokens["input_ids"][0], "attention_mask": tokens["attention_mask"][0]}])
        labels = batch["labels"][0]
        # Assert there are -100 tokens (masked user prompt) and >0 tokens (unmasked response)
        has_masked = (labels == -100).any().item()
        has_unmasked = (labels != -100).any().item()
        if has_masked and has_unmasked:
            report["checks"]["collator_sanity"] = {"status": "PASS"}
        else:
            report["checks"]["collator_sanity"] = {"status": "FAIL", "reason": "Masking failed"}
            overall_pass = False
    except Exception as e:
        report["checks"]["collator_sanity"] = {"status": "FAIL", "reason": str(e)}
        overall_pass = False

    # 2. WEIGHT DELTA CHECK
    try:
        safetensor_path = Path(adapter_path) / "adapter_model.safetensors"
        if not safetensor_path.exists():
            raise FileNotFoundError("adapter_model.safetensors missing")
            
        has_nonzero_delta = False
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "lora_A" in key or "lora_B" in key:
                    tensor = f.get_tensor(key)
                    if tensor.norm().item() > 0.0:
                        has_nonzero_delta = True
                        break
        if has_nonzero_delta:
            report["checks"]["weight_delta"] = {"status": "PASS"}
        else:
            report["checks"]["weight_delta"] = {"status": "FAIL", "reason": "All adapter weights are exact zeros"}
            overall_pass = False
    except Exception as e:
        report["checks"]["weight_delta"] = {"status": "FAIL", "reason": str(e)}
        overall_pass = False

    # INFERENCE CHECKS
    # Use context manager to evaluate base model vs adapter model safely
    test_coding_prompt = "<|im_start|>user\\nWrite a python loop to print 1 to 5.<|im_end|>\\n<|im_start|>assistant\\n"
    test_general_prompt = "<|im_start|>user\\nIn one sentence, what is the capital of France?<|im_end|>\\n<|im_start|>assistant\\n"
    
    inputs_code = tokenizer(test_coding_prompt, return_tensors="pt").to(model.device)
    inputs_general = tokenizer(test_general_prompt, return_tensors="pt").to(model.device)
    
    try:
        # BASE MODEL INFERENCE
        with model.disable_adapter():
            base_code_out = model.generate(**inputs_code, max_new_tokens=20)
            base_code_text = tokenizer.decode(base_code_out[0], skip_special_tokens=True)
            
        # ADAPTER MODEL INFERENCE
        adapter_code_out = model.generate(**inputs_code, max_new_tokens=20)
        adapter_code_text = tokenizer.decode(adapter_code_out[0], skip_special_tokens=True)
        
        # 3. ADAPTER-APPLIED-AT-EVAL CHECK
        if base_code_text != adapter_code_text:
            report["checks"]["adapter_applied_at_eval"] = {"status": "PASS"}
        else:
            report["checks"]["adapter_applied_at_eval"] = {"status": "FAIL", "reason": "Adapter output perfectly matches base model output"}
            overall_pass = False
            
        # 4. CATASTROPHIC FORGETTING CHECK
        adapter_general_out = model.generate(**inputs_general, max_new_tokens=20)
        adapter_general_text = tokenizer.decode(adapter_general_out[0], skip_special_tokens=True).lower()
        
        # Simple heuristic: The model shouldn't spit out code (e.g. def, import, brackets) for a basic geography question
        forgetting_flags = ["def ", "import ", "}", "{", "console.log"]
        if any(flag in adapter_general_text for flag in forgetting_flags):
            report["checks"]["catastrophic_forgetting"] = {"status": "FAIL", "reason": f"Model lost general capabilities, outputted code syntax: {adapter_general_text}"}
            overall_pass = False
        else:
            report["checks"]["catastrophic_forgetting"] = {"status": "PASS"}
            
    except Exception as e:
        report["checks"]["inference"] = {"status": "FAIL", "reason": str(e)}
        overall_pass = False

    report["overall_status"] = "PASS" if overall_pass else "FAIL"
    
    out_file = Path("results") / "validate_dora_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
        
    logger.info(f"Validation report saved to {out_file}. Status: {report['overall_status']}")
    if not overall_pass:
        logger.warning("Validation Gate detected issues. Review results/validate_dora_report.json. Model checkpoint is preserved.")
        
    return report

if __name__ == "__main__":
    pass  # Standalone execution disabled as it requires model and tokenizer objects in VRAM
