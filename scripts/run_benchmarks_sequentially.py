import os
import json
import time
import random
import sys
import subprocess
import tempfile
import urllib.request

# Disable Hugging Face verbose logs and cache models in memory (crucial for H100)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["SABER_KEEP_MODELS_LOADED"] = "1"

# Ensure SABER modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saber.llm_engine import LLMEngine
from saber.sentinel import Sentinel
from saber.signal import Signal, SignalType

def print_scoreboard(domain_name, idx, total, correct_count, sentinel_flagged_count):
    acc = (correct_count / idx) * 100.0 if idx > 0 else 0.0
    flag_pct = (sentinel_flagged_count / idx) * 100.0 if idx > 0 else 0.0
    print(f"\n=======================================================")
    print(f"[LIVE UPDATE] {domain_name} Progress: {idx}/{total} cases completed.")
    print(f"Dynamic Scoreboard:")
    print(f"| Metric | Current Value |")
    print(f"| :--- | :--- |")
    print(f"| Local Accuracy (CoT) | {acc:.1f}% ({correct_count}/{idx}) |")
    print(f"| Sentinel Flag Rate | {flag_pct:.1f}% ({sentinel_flagged_count}/{idx}) |")
    print(f"=======================================================\n")

def parse_claims(raw_response):
    claims_texts = []
    if "CLAIMS:" in raw_response:
        try:
            # Split out claims block
            after_claims = raw_response.split("CLAIMS:")[1]
            # End of claims block is either CODE: or ANSWER: or end of string
            end_marker = "ANSWER:" if "ANSWER:" in after_claims else "CODE:"
            claims_block = after_claims.split(end_marker)[0]
            for line in claims_block.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    clean_claim = line.lstrip("1234567890.- ").strip()
                    if clean_claim:
                        claims_texts.append(clean_claim)
        except Exception:
            pass
    if not claims_texts:
        claims_texts = [raw_response[:100]] # Fallback
    return claims_texts

def run_sentinel_check(domain, idx, claims_texts, raw_response):
    claim_objects = [{"statement": c} for c in claims_texts]
    signal = Signal(
        signal_type=SignalType.OUTPUT_SIGNAL,
        query_id=f"{domain.upper()}_{idx}",
        source_id="local_engine",
        target_id="sentinel",
        payload={"claims": claim_objects}
    )
    try:
        verification_signal = Sentinel.verify_interpretation(domain, signal, raw_response)
        return verification_signal.signal_type == SignalType.FLAG_SIGNAL
    except Exception as e:
        print(f"[!] Sentinel Error: {e}")
        return False

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
# BENCHMARK EVALUATORS
# =====================================================================

def run_science_benchmark(model_path):
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
    
    system_prompt = (
        "You are an expert scientific reasoner. First, think step by step to deduce the answer. "
        "Second, output exactly 3 factual claims that support your reasoning. "
        "Finally, state the correct option letter (A, B, C, or D).\n\n"
        "Use this strict format:\n"
        "REASONING: <your step by step thought process>\n"
        "CLAIMS:\n1. <claim 1>\n2. <claim 2>\n3. <claim 3>\n"
        "ANSWER: <A, B, C, or D>"
    )

    correct_count = 0
    sentinel_flagged_count = 0

    print(f"[*] Initializing local Science Specialist model from {model_path}...")
    with LLMEngine(model_path, max_new_tokens=1024) as engine:
        for idx, case in enumerate(cases, 1):
            print(f"[*] Science Case {idx}/{len(cases)}")
            raw_response = engine.generate(case["question"], system_prompt=system_prompt)
            
            claims_texts = parse_claims(raw_response)
            flagged = run_sentinel_check("science", idx, claims_texts, raw_response)
            if flagged:
                sentinel_flagged_count += 1
                
            ans_char = ""
            if "ANSWER:" in raw_response:
                ans_char = raw_response.split("ANSWER:")[-1].strip().upper()
            else:
                ans_char = raw_response[-5:].strip().upper()
                
            ans_char = "".join([c for c in ans_char if c in "ABCD"])
            ans_char = ans_char[0] if ans_char else "N/A"
                
            is_correct = (ans_char == case["expected"])
            if is_correct:
                correct_count += 1
                
            if idx % 10 == 0 or idx == len(cases):
                print_scoreboard("Science (GPQA)", idx, len(cases), correct_count, sentinel_flagged_count)
                
    final_score = (correct_count / len(cases)) * 100.0 if cases else 0.0
    print(f"[+] Science (GPQA) Benchmark Completed: {final_score:.1f}%")
    return final_score

def run_cyber_benchmark(model_path):
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
    
    system_prompt = (
        "You are an expert cybersecurity specialist. First, think step by step to deduce the answer. "
        "Second, output exactly 3 factual claims that support your reasoning. "
        "Finally, state the correct option letter (A, B, C, or D).\n\n"
        "Use this strict format:\n"
        "REASONING: <your step by step thought process>\n"
        "CLAIMS:\n1. <claim 1>\n2. <claim 2>\n3. <claim 3>\n"
        "ANSWER: <A, B, C, or D>"
    )

    correct_count = 0
    sentinel_flagged_count = 0

    print(f"[*] Initializing local Cyber Specialist model from {model_path}...")
    with LLMEngine(model_path, max_new_tokens=1024) as engine:
        for idx, case in enumerate(cases, 1):
            print(f"[*] Cyber Case {idx}/{len(cases)}")
            raw_response = engine.generate(case["question"], system_prompt=system_prompt)
            
            claims_texts = parse_claims(raw_response)
            flagged = run_sentinel_check("cyber", idx, claims_texts, raw_response)
            if flagged:
                sentinel_flagged_count += 1
                
            ans_char = ""
            if "ANSWER:" in raw_response:
                ans_char = raw_response.split("ANSWER:")[-1].strip().upper()
            else:
                ans_char = raw_response[-5:].strip().upper()
                
            ans_char = "".join([c for c in ans_char if c in "ABCD"])
            ans_char = ans_char[0] if ans_char else "N/A"
                
            is_correct = (ans_char == case["expected"])
            if is_correct:
                correct_count += 1
                
            if idx % 10 == 0 or idx == len(cases):
                print_scoreboard("Cyber (CyberMetric)", idx, len(cases), correct_count, sentinel_flagged_count)
                
    final_score = (correct_count / len(cases)) * 100.0 if cases else 0.0
    print(f"[+] Cyber (CyberMetric) Benchmark Completed: {final_score:.1f}%")
    return final_score

def run_coding_benchmark(model_path):
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
    
    system_prompt = (
        "You are an expert coding specialist. First, think step by step to solve the task. "
        "Second, output exactly 3 factual claims about the logic/complexity. "
        "Finally, output the complete Python implementation wrapped inside a ```python block.\n\n"
        "Use this strict format:\n"
        "REASONING: <your step by step thought process>\n"
        "CLAIMS:\n1. <claim 1>\n2. <claim 2>\n3. <claim 3>\n"
        "CODE:\n```python\n<your code here>\n```"
    )

    correct_count = 0
    sentinel_flagged_count = 0

    print(f"[*] Initializing local Coding Specialist model from {model_path}...")
    with LLMEngine(model_path, max_new_tokens=1024) as engine:
        for idx, case in enumerate(cases, 1):
            print(f"[*] Coding Case {idx}/{len(cases)}")
            prompt = case["prompt"]
            test_code = case["test"]
            entry_point = case["entry_point"]
            question_str = f"Complete the following Python function:\n{prompt}"
            
            raw_response = engine.generate(question_str, system_prompt=system_prompt)
            
            claims_texts = parse_claims(raw_response)
            flagged = run_sentinel_check("coding", idx, claims_texts, raw_response)
            if flagged:
                sentinel_flagged_count += 1
                
            is_correct = check_coding_correctness(prompt, raw_response, test_code, entry_point)
            if is_correct:
                correct_count += 1
                
            if idx % 10 == 0 or idx == len(cases):
                print_scoreboard("Coding (HumanEval)", idx, len(cases), correct_count, sentinel_flagged_count)
                
    final_score = (correct_count / len(cases)) * 100.0 if cases else 0.0
    print(f"[+] Coding (HumanEval) Benchmark Completed: {final_score:.1f}%")
    return final_score

def run_finance_benchmark(model_path):
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
    
    system_prompt = (
        "You are an expert financial analyst. First, think step by step to solve the question. "
        "Second, output exactly 3 factual claims that support your reasoning. "
        "Finally, state the correct numerical answer.\n\n"
        "Use this strict format:\n"
        "REASONING: <your step by step thought process>\n"
        "CLAIMS:\n1. <claim 1>\n2. <claim 2>\n3. <claim 3>\n"
        "ANSWER: <numeric value>"
    )

    correct_count = 0
    sentinel_flagged_count = 0

    print(f"[*] Initializing local Finance Specialist model from {model_path}...")
    with LLMEngine(model_path, max_new_tokens=1024) as engine:
        for idx, case in enumerate(cases, 1):
            print(f"[*] Finance Case {idx}/{len(cases)}")
            raw_response = engine.generate(case["question"], system_prompt=system_prompt)
            
            claims_texts = parse_claims(raw_response)
            flagged = run_sentinel_check("finance", idx, claims_texts, raw_response)
            if flagged:
                sentinel_flagged_count += 1
                
            ans_val = ""
            if "ANSWER:" in raw_response:
                ans_val = raw_response.split("ANSWER:")[-1].strip()
            else:
                ans_val = raw_response[-20:].strip()
                
            is_correct = False
            expected_str = case["expected"]
            if expected_str in ans_val:
                is_correct = True
                correct_count += 1
                
            if idx % 10 == 0 or idx == len(cases):
                print_scoreboard("Finance (FinQA Math)", idx, len(cases), correct_count, sentinel_flagged_count)
                
    final_score = (correct_count / len(cases)) * 100.0 if cases else 0.0
    print(f"[+] Finance (FinQA Math) Benchmark Completed: {final_score:.1f}%")
    return final_score

# =====================================================================
# MAIN PIPELINE
# =====================================================================
def main():
    print("=====================================================================")
    print("[*] SABER Single-GPU Sequential Benchmark Runner (H100 optimized)")
    print("=====================================================================\n")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Run sequentially
    science_score = run_science_benchmark(os.path.join(base_dir, "models", "science_v2"))
    cyber_score = run_cyber_benchmark(os.path.join(base_dir, "models", "cyber_v2"))
    coding_score = run_coding_benchmark(os.path.join(base_dir, "models", "coding_v2"))
    finance_score = run_finance_benchmark(os.path.join(base_dir, "models", "finance_v2"))
    
    print("\n" + "="*55)
    print("=== FINAL BENCHMARK SUMMARY (Single-GPU Sequential) ===")
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
