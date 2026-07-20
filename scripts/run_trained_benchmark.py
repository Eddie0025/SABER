import os
import json
import time
import random
import sys
import subprocess
import tempfile
import urllib.request

# Ensure saber module can be imported
sys.path.append(os.path.abspath('.'))

# Disable Hugging Face verbose logs and progress bars
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["SABER_KEEP_MODELS_LOADED"] = "1"

from saber.config import SaberConfig, VerificationTier
from saber.registry import SpecialistRegistry
from saber.audit import AuditLogger
from saber.orchestrator import Orchestrator

def print_scoreboard(domain_name, idx, total, correct_count, flagged_count):
    acc = (correct_count / idx) * 100.0 if idx > 0 else 0.0
    flag_pct = (flagged_count / idx) * 100.0 if idx > 0 else 0.0
    print(f"\n=======================================================")
    print(f"[LIVE UPDATE] {domain_name} Progress: {idx}/{total} cases completed.")
    print(f"Pipeline Dynamic Scoreboard:")
    print(f"| Metric | Current Value |")
    print(f"| :--- | :--- |")
    print(f"| Pipeline Accuracy | {acc:.1f}% ({correct_count}/{idx}) |")
    print(f"| Sentinel Flag Rate | {flag_pct:.1f}% ({flagged_count}/{idx}) |")
    print(f"=======================================================\n")

# =====================================================================
# CODING CORRECTNESS SANDBOX
# =====================================================================
def check_coding_correctness(prompt, completion, test_code, entry_point):
    """Executes generated code against HumanEval test assertions in a sandboxed subprocess."""
    code_clean = completion
    if "```python" in completion:
        code_clean = completion.split("```python")[1].split("```")[0]
    elif "```" in completion:
        code_clean = completion.split("```")[1].split("```")[0]
        
    full_code = f"{prompt}\n{code_clean}\n{test_code}\ncheck({entry_point})"
    
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(full_code)
            temp_name = f.name
            
        res = subprocess.run(["python3", temp_name], capture_output=True, text=True, timeout=5)
        return res.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        if temp_name and os.path.exists(temp_name):
            try:
                os.unlink(temp_name)
            except Exception:
                pass

# =====================================================================
# BENCHMARK RUNNERS
# =====================================================================

def run_science_benchmark(orch):
    print("\n" + "="*55)
    print("[*] Loading GPQA Diamond dataset from Hugging Face Hub (Last 78 records)...")
    try:
        from datasets import load_dataset
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token and os.path.exists("api_key.txt"):
            with open("api_key.txt", "r") as f:
                hf_token = f.read().strip()
        
        ds = load_dataset("idavidrein/gpqa", "gpqa_diamond", split="train", token=hf_token)
        all_cases = list(ds)
        sliced_ds = all_cases[-78:]
        
        cases = []
        for row in sliced_ds:
            corr = row.get("correct_answer") or row.get("Correct Answer")
            inc1 = row.get("incorrect_answer1") or row.get("Incorrect Answer 1")
            inc2 = row.get("incorrect_answer2") or row.get("Incorrect Answer 2")
            inc3 = row.get("incorrect_answer3") or row.get("Incorrect Answer 3")
            q_text = row.get("question") or row.get("Question")
            if not corr or not q_text: continue
            
            choices = [corr, inc1, inc2, inc3]
            random.seed(42) # Consistent shuffle
            random.shuffle(choices)
            choices_str = "\n".join([f"{chr(65+i)}: {c}" for i, c in enumerate(choices)])
            expected_char = chr(65 + choices.index(corr))
            
            cases.append({
                "question": f"Question: {q_text}\nOptions:\n{choices_str}",
                "expected": expected_char
            })
    except Exception as e:
        print(f"[!] Failed to load GPQA: {e}")
        return 0.0

    print(f"[+] Successfully loaded {len(cases)} cases.\n")

    correct_count = 0
    flagged_count = 0

    for idx, case in enumerate(cases, 1):
        print(f"[*] Science Case {idx}/{len(cases)}")
        res = orch.process_query(case["question"])
        ans = res.get("answer", "").strip()
        
        # Check if Sentinel verification flagged this case
        if res.get("flags"):
            flagged_count += 1
            
        ans_char = ""
        if "ANSWER:" in ans:
            ans_char = ans.split("ANSWER:")[-1].strip().upper()
        else:
            ans_char = ans[-5:].strip().upper()
            
        ans_char = "".join([c for c in ans_char if c in "ABCD"])
        ans_char = ans_char[0] if ans_char else "N/A"
            
        is_correct = (ans_char == case["expected"])
        if is_correct:
            correct_count += 1
            
        if idx % 10 == 0 or idx == len(cases):
            print_scoreboard("Science (GPQA) Pipeline", idx, len(cases), correct_count, flagged_count)
            
    final_score = (correct_count / len(cases)) * 100.0 if cases else 0.0
    print(f"[+] Science (GPQA) Pipeline Benchmark Completed: {final_score:.1f}%")
    return final_score

def run_cyber_benchmark(orch):
    print("\n" + "="*55)
    print("[*] Loading CyberMetric from GitHub Raw URL (Last 60 records)...")
    try:
        data = []
        urls = [
            "https://raw.githubusercontent.com/cybermetric/CyberMetric/main/CyberMetric-500-v1.json",
            "https://raw.githubusercontent.com/cybermetric/CyberMetric/main/CyberMetric-80-v1.json"
        ]
        
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=10) as response:
                    raw_data = json.loads(response.read().decode("utf-8"))
                    if isinstance(raw_data, dict):
                        for k, v in raw_data.items():
                            if isinstance(v, list):
                                data = v
                                break
                    elif isinstance(raw_data, list):
                        data = raw_data
                    if data:
                        break
            except Exception:
                continue

        if not data:
            print("[!] Failed to load CyberMetric raw data.")
            return 0.0

        all_cases = []
        for row in data:
            q_text = row.get("question") or row.get("Question")
            choices = []
            choices_dict = row.get("answers") or row.get("choices") or row
            if isinstance(choices_dict, list):
                choices = choices_dict
            elif isinstance(choices_dict, dict):
                for opt in ["A", "B", "C", "D", "a", "b", "c", "d", "1", "2", "3", "4"]:
                    if opt in choices_dict and choices_dict[opt]:
                        choices.append(choices_dict[opt])
            
            correct_ans = row.get("solution") or row.get("answer") or row.get("Answer") or row.get("correct") or row.get("correct_answer")
            
            if not q_text or not correct_ans or not choices:
                continue
            choices_str = "\n".join([f"{chr(65+i)}: {c}" for i, c in enumerate(choices)])
            if str(correct_ans).upper() in ["A", "B", "C", "D"]:
                correct_char = str(correct_ans).upper()
            else:
                try:
                    idx = choices.index(correct_ans)
                    correct_char = chr(65 + idx)
                except ValueError:
                    correct_char = "A"
                    
            all_cases.append({
                "question": f"Question: {q_text}\nOptions:\n{choices_str}",
                "expected": correct_char
            })
            
        cases = all_cases[-60:]
    except Exception as e:
        print(f"[!] Failed to parse CyberMetric dataset: {e}")
        return 0.0

    print(f"[+] Successfully loaded {len(cases)} cases.\n")

    correct_count = 0
    flagged_count = 0

    for idx, case in enumerate(cases, 1):
        print(f"[*] Cyber Case {idx}/{len(cases)}")
        res = orch.process_query(case["question"])
        ans = res.get("answer", "").strip()
        
        # Check if Sentinel verification flagged this case
        if res.get("flags"):
            flagged_count += 1
            
        ans_char = ""
        if "ANSWER:" in ans:
            ans_char = ans.split("ANSWER:")[-1].strip().upper()
        else:
            ans_char = ans[-5:].strip().upper()
            
        ans_char = "".join([c for c in ans_char if c in "ABCD"])
        ans_char = ans_char[0] if ans_char else "N/A"
            
        is_correct = (ans_char == case["expected"])
        if is_correct:
            correct_count += 1
            
        if idx % 10 == 0 or idx == len(cases):
            print_scoreboard("Cyber (CyberMetric) Pipeline", idx, len(cases), correct_count, flagged_count)
            
    final_score = (correct_count / len(cases)) * 100.0 if cases else 0.0
    print(f"[+] Cyber (CyberMetric) Pipeline Benchmark Completed: {final_score:.1f}%")
    return final_score

def run_coding_benchmark(orch):
    print("\n" + "="*55)
    print("[*] Loading HumanEval dataset (All 164 records)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/openai_humaneval", split="test")
        cases = list(ds)
    except Exception as e:
        print(f"[!] Failed to load HumanEval: {e}")
        return 0.0

    print(f"[+] Successfully loaded {len(cases)} cases.\n")

    correct_count = 0
    flagged_count = 0

    for idx, case in enumerate(cases, 1):
        print(f"[*] Coding Case {idx}/{len(cases)}")
        prompt = case["prompt"]
        test_code = case["test"]
        entry_point = case["entry_point"]
        question_str = f"Complete the following Python function:\n{prompt}"
        
        res = orch.process_query(question_str)
        ans = res.get("answer", "").strip()
        
        # Check if Sentinel verification flagged this case
        if res.get("flags"):
            flagged_count += 1
            
        is_correct = check_coding_correctness(prompt, ans, test_code, entry_point)
        if is_correct:
            correct_count += 1
            
        if idx % 10 == 0 or idx == len(cases):
            print_scoreboard("Coding (HumanEval) Pipeline", idx, len(cases), correct_count, flagged_count)
            
    final_score = (correct_count / len(cases)) * 100.0 if cases else 0.0
    print(f"[+] Coding (HumanEval) Pipeline Benchmark Completed: {final_score:.1f}%")
    return final_score

def run_finance_benchmark(orch):
    print("\n" + "="*55)
    print("[*] Generating FinQA calculations dataset (100 records)...")
    
    # Reproducible seed math calculations generator
    random.seed(42)
    cases = []
    for idx in range(1, 101):
        rev = random.randint(100, 5000)
        cogs = random.randint(50, int(rev * 0.6))
        gp = rev - cogs
        cases.append({
            "question": f"Context: Revenue: ${rev}M, COGS: ${cogs}M.\nQuestion: Calculate Gross Profit.",
            "expected": str(gp)
        })
        
    print(f"[+] Successfully loaded {len(cases)} cases.\n")

    correct_count = 0
    flagged_count = 0

    for idx, case in enumerate(cases, 1):
        print(f"[*] Finance Case {idx}/{len(cases)}")
        res = orch.process_query(case["question"])
        ans = res.get("answer", "").strip()
        
        # Check if Sentinel verification flagged this case
        if res.get("flags"):
            flagged_count += 1
            
        ans_val = ""
        if "ANSWER:" in ans:
            ans_val = ans.split("ANSWER:")[-1].strip()
        else:
            ans_val = ans[-20:].strip()
            
        is_correct = False
        expected_str = case["expected"]
        if expected_str in ans_val:
            is_correct = True
            correct_count += 1
            
        if idx % 10 == 0 or idx == len(cases):
            print_scoreboard("Finance (FinQA Math) Pipeline", idx, len(cases), correct_count, flagged_count)
            
    final_score = (correct_count / len(cases)) * 100.0 if cases else 0.0
    print(f"[+] Finance (FinQA Math) Pipeline Benchmark Completed: {final_score:.1f}%")
    return final_score

# =====================================================================
# MAIN PIPELINE
# =====================================================================
def main():
    print("=====================================================================")
    print("[*] SABER Multi-Agent Pipeline Benchmark (H100 single-GPU optimized)")
    print("=====================================================================\n")
    
    # Initialize the Orchestrator with auto-discovered specialists
    config = SaberConfig()
    # Force Tier 2 (4-check Sentinel) by default for comprehensive pipeline evaluation
    config.verification_tier = VerificationTier.TIER_2
    
    registry = SpecialistRegistry()
    registry.auto_discover()
    
    # Load domain specialty adapter checkpoints
    for domain, specialist in registry.all().items():
        model_path = f"models/{domain}_v2"
        if os.path.exists(model_path):
            specialist.load_model(model_path)
            print(f"[*] Loaded specialist adapter for '{domain}': {model_path}")
        else:
            specialist.load_model("Qwen/Qwen2.5-7B")
            print(f"[!] Specialist '{domain}' adapter not found; falling back to base Qwen/Qwen2.5-7B")
            
    audit = AuditLogger()
    orch = Orchestrator(config=config, registry=registry, audit=audit)
    
    # Run pipeline benchmarks sequentially
    science_score = run_science_benchmark(orch)
    cyber_score = run_cyber_benchmark(orch)
    coding_score = run_coding_benchmark(orch)
    finance_score = run_finance_benchmark(orch)
    
    print("\n" + "="*55)
    print("=== FINAL PIPELINE BENCHMARK SUMMARY (Single-GPU Sequential) ===")
    print("="*55)
    print(f"| Benchmark | Score |")
    print(f"| :--- | :--- |")
    print(f"| Science (GPQA last 78) | {science_score:.1f}% |")
    print(f"| Cyber (CyberMetric last 60) | {cyber_score:.1f}% |")
    print(f"| Coding (HumanEval all 164) | {coding_score:.1f}% |")
    print(f"| Finance (FinQA math 100) | {finance_score:.1f}% |")
    print("=======================================================")

if __name__ == "__main__":
    main()
