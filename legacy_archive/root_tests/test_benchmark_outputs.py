import sys, os
from saber.config import SaberConfig
from saber.llm_engine import LLMEngine
from saber.specialists.science import ScienceSpecialist
from saber.signal import Signal, SignalType
from saber.sentinel import Sentinel
from scripts.run_final_benchmark import load_hf_dataset, build_mcq_prompt, parse_mcq_answer
import random

os.environ["SABER_BENCHMARK_MODE"] = "1"
config = SaberConfig()
gpqa = load_hf_dataset("idavidrein/gpqa", "gpqa_diamond", split="train")

for i, row in enumerate(gpqa):
    if i >= 1: break
    corr = row.get("correct_answer") or row.get("Correct Answer")
    inc1 = row.get("incorrect_answer1") or row.get("Incorrect Answer 1")
    inc2 = row.get("incorrect_answer2") or row.get("Incorrect Answer 2")
    inc3 = row.get("incorrect_answer3") or row.get("Incorrect Answer 3")
    q_text = row.get("question") or row.get("Question")
    choices = [corr, inc1, inc2, inc3]
    random.seed(42)
    random.shuffle(choices)
    choices_str = "\n".join([f"{chr(65+j)}: {c}" for j, c in enumerate(choices)])
    correct_char = chr(65 + choices.index(corr))
    q = build_mcq_prompt(q_text, choices_str)
    
    print(f"Question:\n{q}")
    print(f"Expected: {correct_char}")
    
    # Base
    with LLMEngine(config.base_model) as engine:
        ans1 = engine.generate(q).strip()
    print(f"\n--- BASE RAW ---\n{ans1}\nParsed: {parse_mcq_answer(ans1)}")

    # Adapter
    spec = ScienceSpecialist()
    spec.load_model("models/science_v2")
    with LLMEngine(spec.meta.model_path) as engine:
        ans2 = engine.generate(q).strip()
    print(f"\n--- ADAPTER RAW ---\n{ans2}\nParsed: {parse_mcq_answer(ans2)}")

    # CoT
    task_sig = Signal(signal_type=SignalType.TASK_SIGNAL, query_id="test-1", source_id="TEST", target_id="science", payload={"objective": q}).freeze_and_hash()
    out_sig = spec.handle_signal(task_sig)
    ans3 = out_sig.payload.get("raw_response", "")
    if not ans3 and out_sig.payload.get("claims"):
        ans3 = out_sig.payload["claims"][0].get("statement", "")
    print(f"\n--- COT RAW ---\n{ans3}\nParsed: {parse_mcq_answer(ans3)}")

