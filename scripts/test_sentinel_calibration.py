import os
import sys
import json
from saber.llm_engine import LLMEngine
from saber.sentinel import Sentinel
from saber.signal import Signal, SignalType

def main():
    print("=========================================================")
    # Set model paths
    model_path = "models/medical_v2"
    base_model = "Qwen/Qwen2.5-7B-Instruct"
    
    if not os.path.exists(model_path):
        print(f"Error: Calibrated model '{model_path}' not found locally.")
        sys.exit(1)
        
    print("Loading LLM Engine...")
    engine = LLMEngine(model_path)
    engine.__enter__()
    
    questions = {
        "cushing": "What is the classic triad of Cushing's reflex (raised ICP)?",
        "boerhaave": "What is the classic finding of Boerhaave syndrome on imaging?"
    }
    
    # Calibrated Prompt
    system_prompt = (
        "You are a highly skilled Medical AI specialist. Provide a thorough, accurate, and evidence-based clinical answer. "
        "When asked to name a specific test, marker, sign, or triad component, only state it if you are highly confident it is a real, "
        "established clinical term. If you are not certain of the exact name, say 'I am not fully certain of the specific term, "
        "but the relevant clinical concept is...' rather than generating a plausible-sounding but unverified specific name. "
        "Do not invent, guess, or fabricate precise terminology."
    )
    
    for key, q in questions.items():
        print("\n" + "="*60)
        print(f" TESTING: {q}")
        print("="*60)
        
        # 1. Generate raw calibrated response
        print("[+] Generating Raw Calibrated Response...")
        raw_response = engine.generate(q, system_prompt=system_prompt)
        print(f"--- RAW RESPONSE ---\n{raw_response}\n--------------------\n")
        
        # 2. Run Sentinel Verification
        print("[+] Simulating Sentinel Verification...")
        # Construct fake Output Signal
        out_sig = Signal(
            signal_type=SignalType.OUTPUT_SIGNAL,
            query_id=f"test_{key}",
            source_id="SPEC-MEDICAL",
            target_id="MANAGER",
            payload={"claims": [{"statement": raw_response, "confidence": 0.9}]}
        ).freeze_and_hash()
        
        # We temporarily patch/monkeypatch Sentinel's print function to capture what search returns
        # or we just rely on Sentinel's normal prints since it does print:
        # "[Sentinel] Online: searching for '...'"
        
        ver_res = Sentinel.verify_interpretation(
            specialist_domain="medical",
            original_signal=out_sig,
            compiled_text=raw_response
        )
        
        print(f"\n--- SENTINEL VERDICT ---")
        print(f"Signal Type: {ver_res.signal_type.name}")
        print(f"Payload: {json.dumps(ver_res.payload, indent=2)}")
        print("------------------------\n")
        
    engine.__exit__(None, None, None)

if __name__ == "__main__":
    main()
