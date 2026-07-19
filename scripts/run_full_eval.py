import os
import sys
import importlib.util
import json
import subprocess

# Ensure fpdf2 is installed
try:
    from fpdf import FPDF
except ImportError:
    print("[*] Installing fpdf2 library dynamically...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2"])
    from fpdf import FPDF

# Ensure saber module can be imported
sys.path.append(os.path.abspath('.'))
from saber.llm_engine import LLMEngine
from saber.registry import SpecialistRegistry

def clean_txt(text):
    if not isinstance(text, str):
        return str(text)
    # Replace common unicode chars with ASCII equivalents
    replacements = {
        "\u2013": "-", "\u2014": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2022": "*", "\u2714": "[Yes]",
        "\u2718": "[No]", "\u00e9": "e", "\u00e0": "a", "\u00f9": "u",
        "\u00e7": "c", "\u2248": "~", "\u2265": ">=", "\u2264": "<="
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Encode as latin-1, replacing unmappable characters
    return text.encode("latin-1", "replace").decode("latin-1")

def load_test_cases(script_name):
    script_path = os.path.join("scripts", script_name)
    spec = importlib.util.spec_from_file_location("eval_script", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "TEST_CASES")

class PDFReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 15)
        self.cell(0, 10, "SABER Specialist Evaluation Report", border=False, new_x="LMARGIN", new_y="NEXT", align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

def main():
    domains_config = {
        "medical": {
            "script": "eval_medical_30.py",
            "model_path": "models/medical_v2",
            "system_prompt": "You are a highly skilled Medical AI specialist. Provide a thorough, accurate, and evidence-based clinical answer."
        },
        "cyber": {
            "script": "eval_cyber_30.py",
            "model_path": "models/cyber_v2",
            "system_prompt": "You are a highly skilled Cybersecurity AI specialist. Provide a thorough, accurate, and evidence-based analytical answer."
        },
        "science": {
            "script": "eval_science_30.py",
            "model_path": "models/science_v2",
            "system_prompt": "You are a highly skilled Scientific AI specialist. Provide a thorough, accurate, and evidence-based clinical/scientific answer."
        },
        "coding": {
            "script": "eval_coding_30.py",
            "model_path": "models/coding_v2",
            "system_prompt": "You are a highly skilled Coding AI specialist. Provide a thorough, accurate, and evidence-based coding answer."
        },
        "architecture": {
            "script": "eval_architecture_30.py",
            "model_path": "models/architecture_v2",
            "system_prompt": "You are a highly skilled Systems Architecture AI specialist. Provide a thorough, accurate, and evidence-based analytical answer."
        },
        "finance": {
            "script": "eval_finance_30.py",
            "model_path": "models/finance_v2",
            "system_prompt": "You are a highly skilled Financial AI specialist. Provide a thorough, accurate, and evidence-based analytical answer."
        },
        "meta_reasoner": {
            "script": "eval_meta_reasoner_30.py",
            "model_path": "models/meta_reasoner_v2",
            "system_prompt": "You are the SABER Meta-Reasoner. Resolve contradictions, arbitrate preferences, identify hallucinations, and output a unified recommendation."
        },
        "orchestrator": {
            "script": "eval_orchestrator_30.py",
            "model_path": "models/orchestrator_v2",
            "system_prompt": "orchestrator_special_prompt"
        }
    }

    # Discover domains for the Orchestrator system prompt
    registry = SpecialistRegistry()
    registry.auto_discover()
    available_domains = ", ".join(registry.list_domains())
    orchestrator_prompt = (
        "You are the SABER Orchestrator. Your sole responsibility is to evaluate "
        "the user's prompt and route it to the correct specialist domains based on the required technical expertise. "
        "For complex system-building, application design, or pipeline engineering requests, you must route to "
        "both 'architecture' (for design) and 'coding' (for implementation) in addition to the specific domain "
        f"(e.g., 'finance', 'medical', 'science'). Available specialists: {available_domains}. "
        "DO NOT answer the user's question. You must output strict JSON matching "
        "the following schema: "
        "{\"route\": [\"domain1\", \"domain2\"], \"confidence\": 0.99, \"multi_domain\": true, \"query_summary\": \"...\"}"
    )

    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)

    for domain_name, cfg in domains_config.items():
        print(f"\n=========================================================")
        print(f" Running Evaluation for: {domain_name.upper()}")
        print(f" Loading Model: {cfg['model_path']}")
        print(f"=========================================================")

        # Load test cases from the respective script
        try:
            test_cases = load_test_cases(cfg["script"])
            print(f"Loaded {len(test_cases)} test cases.")
        except Exception as e:
            print(f"[!] Error loading test cases from {cfg['script']}: {e}")
            continue

        # Setup system prompt
        sys_prompt = orchestrator_prompt if cfg["system_prompt"] == "orchestrator_special_prompt" else cfg["system_prompt"]

        # Run model inference sequentially
        results = []
        try:
            with LLMEngine(cfg["model_path"]) as engine:
                for idx, case in enumerate(test_cases, 1):
                    q = case["question"]
                    print(f"[{domain_name.upper()}] Case {idx}/{len(test_cases)}")
                    
                    history = [{"role": "system", "content": sys_prompt}]
                    try:
                        ans = engine.generate_with_history(history, new_user_message=q)
                    except Exception as e:
                        ans = f"[ERROR GENERATING]: {e}"
                    
                    results.append({
                        "id": case.get("id", idx),
                        "category": case.get("category", "General"),
                        "question": q,
                        "answer": ans
                    })
        except Exception as e:
            print(f"[!] Failed to load or execute model {cfg['model_path']}: {e}")
            continue

        # Add section to PDF
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, f"Domain: {domain_name.upper()}", border=False, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        for res in results:
            pdf.set_font("Helvetica", "B", 10)
            q_header = f"Case {res['id']} [{res['category']}]: {res['question']}"
            pdf.multi_cell(0, 6, clean_txt(q_header))
            pdf.ln(1)
            
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5, clean_txt(f"SABER: {res['answer']}"))
            pdf.ln(4)

    # Save PDF output
    report_path = "saber_evaluation_report.pdf"
    pdf.output(report_path)
    print(f"\n=========================================================")
    print(f" Evaluation Completed! PDF report generated at: {report_path}")
    print(f"=========================================================")

if __name__ == "__main__":
    main()
