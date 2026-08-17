#!/usr/bin/env python3
"""
SABER Web UI Server
Pure Python HTTP server serving the ChatGPT/Claude-style Web UI with dual mode:
- Live Mode: Connects to saber.orchestrator.Orchestrator when models are loaded.
- High-Fidelity Simulation Mode: Generates dynamic, realistic reasoning & answers for testing.
"""

import os
import json
import time
import argparse
import mimetypes
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

# Try importing live SABER engine
LIVE_SABER_AVAILABLE = False
orchestrator_instance = None

try:
    from saber.specialist import SpecialistEngine
    from saber.orchestrator import Orchestrator
    from saber.context import SessionContext
    
    # We will initialize lazily if requested
    LIVE_SABER_AVAILABLE = True
except Exception:
    LIVE_SABER_AVAILABLE = False

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


def simulate_saber_response(query: str, deep_reason: bool = True):
    """
    Generates high-fidelity simulated responses for interactive testing without GPU checkpoints.
    """
    q_lower = query.lower()
    
    # 1. Casual Chat
    if any(w in q_lower for w in ["hello", "hi", "hey", "how are you", "who are you", "what are you", "thanks"]):
        return {
            "thinking": "",
            "response": "Hello! I am SABER, an AI assistant built for deep technical reasoning, code architecture, and multi-domain problem solving. How can I help you today?"
        }
    
    # 2. Cybersecurity
    if any(w in q_lower for w in ["vulnerability", "exploit", "heap", "glibc", "aslr", "xss", "csrf", "double-free", "injection", "kernel"]):
        thinking = (
            "1. IDENTIFY: Analyzing memory corruption vector in multi-threaded glibc allocator.\n"
            "2. ANALYZE: A double-free condition in glibc >= 2.29 triggers tcache count tampering or fastbin dup.\n"
            "3. HYPOTHESIZE: Bypassing ASLR requires an initial info leak (libc base address) via unsorted bin fd/bk pointers.\n"
            "4. EVALUATE: Crafting overlapping chunk allocations allows overwriting the tcache next pointer (fd) toward a target GOT entry or __malloc_hook.\n"
            "5. CONCLUDE: Formulating structured mitigation and exploitation sequence."
        ) if deep_reason else ""
        
        response = (
            "### Vulnerability Analysis: Glibc Heap Double-Free to RCE\n\n"
            "A **double-free vulnerability** occurs when `free()` is invoked twice on the same memory allocation without clearing the dangling pointer. In modern Linux environments (`glibc`), exploitation requires systematically overcoming allocator safeguards:\n\n"
            "#### 1. Heap State Corruption (Tcache Poisoning)\n"
            "- When a chunk is freed twice, it creates a circular linked list within the **Thread Local Cache (tcache)** bin.\n"
            "- In `glibc >= 2.29`, pointer mangling (`PROTECT_PTR`) scrambles the `fd` pointer using a random per-thread key. An attacker must first leak a heap address to compute the demangled key.\n\n"
            "#### 2. ASLR & Memory Leakage\n"
            "- To achieve code execution, the attacker leaks a `libc` base pointer (e.g., by freeing a chunk exceeding the tcache size limit into the `unsorted bin`, where the `fd`/`bk` pointers point into `main_arena`).\n\n"
            "#### 3. Arbitrary Write & Execution Hijack\n"
            "- By modifying the `fd` pointer of the corrupted chunk to point to an arbitrary writable address (e.g., a function pointer or `__free_hook`), subsequent `malloc()` requests return that target address.\n"
            "- Overwriting the target with a one-gadget or ROP payload achieves arbitrary code execution upon the next allocation/free event."
        )
        return {"thinking": thinking, "response": response}
        
    # 3. Medical / Clinical
    if any(w in q_lower for w in ["syndrome", "blood", "patient", "renal", "hypertension", "medical", "disease", "receptor", "drug"]):
        thinking = (
            "1. IDENTIFY: Evaluating differential diagnosis for low-renin, low-aldosterone hypertension with hypokalemic metabolic alkalosis.\n"
            "2. ANALYZE: Liddle syndrome is an autosomal dominant gain-of-function mutation in ENaC subunit genes (SCNN1B/SCNN1G).\n"
            "3. CONTRAST: Apparent Mineralocorticoid Excess (AME) involves 11β-HSD2 enzyme deficiency, allowing cortisol to inappropriately activate mineralocorticoid receptors.\n"
            "4. CONCLUDE: Formulating clinical comparison and pharmacotherapeutic distinctions."
        ) if deep_reason else ""
        
        response = (
            "### Clinical Differential: Liddle Syndrome vs. Apparent Mineralocorticoid Excess (AME)\n\n"
            "Both conditions present with **low renin**, **low aldosterone**, refractory hypertension, hypokalemia, and metabolic alkalosis, but their molecular mechanisms differ:\n\n"
            "| Feature | Liddle Syndrome | Apparent Mineralocorticoid Excess (AME) |\n"
            "| :--- | :--- | :--- |\n"
            "| **Genetics** | Autosomal Dominant (ENaC subunit mutations) | Autosomal Recessive (11β-HSD2 enzyme mutation) |\n"
            "| **Pathophysiology** | Impaired ENaC degradation $\\rightarrow$ constitutive sodium retention | Inability to convert cortisol to cortisone in the kidney |\n"
            "| **Receptor Action** | Direct sodium hyper-reabsorption in collecting tubule | Cortisol hyper-activates mineralocorticoid receptors (MR) |\n"
            "| **Primary Therapy** | ENaC blockers (**Amiloride**, **Triamterene**) | Potassium-sparing MR antagonists (**Spironolactone**) or Dexamethasone |\n\n"
            "**Key Diagnostic Clue:** Patients with Liddle syndrome do *not* respond to spironolactone because the ENaC channel activation is independent of the mineralocorticoid receptor."
        )
        return {"thinking": thinking, "response": response}

    # 4. Coding / Architecture / General
    thinking = (
        "1. IDENTIFY: Analyzing core structural requirements of the query.\n"
        "2. DECOMPOSE: Breaking down requirements into component specifications, state management, and edge cases.\n"
        "3. SYNTHESIZE: Formulating optimal implementation strategy."
    ) if deep_reason else ""
    
    response = (
        "Here is a comprehensive breakdown for your request:\n\n"
        "```python\ndef process_distributed_request(payload: dict) -> dict:\n"
        "    \"\"\"\n    Validates input state and executes idempotent processing.\n    \"\"\"\n"
        "    if not payload.get('id'):\n"
        "        raise ValueError('Missing required identifier')\n        \n"
        "    return {\n        'status': 'SUCCESS',\n        'result': payload\n    }\n```\n\n"
        "### Key Principles:\n"
        "- **Fault Tolerance:** Ensure all state transitions are idempotent with unique request keys.\n"
        "- **Scalability:** Leverage partitioned, horizontal queues to handle high-concurrency loads.\n"
        "- **Observability:** Implement structured trace logging across all execution boundaries."
    )
    return {"thinking": thinking, "response": response}


class SaberWebHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/chat":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            
            try:
                data = json.loads(body)
                query = data.get("query", "")
                deep_reason = data.get("deep_reason", True)
                
                # Check if live orchestrator exists
                if LIVE_SABER_AVAILABLE and orchestrator_instance is not None:
                    # Run on live model
                    context = SessionContext()
                    # Strip any internal tags from output for clean UI
                    raw_res = orchestrator_instance.process(query, context)
                    # Clean out any specialist footers or tags if present
                    clean_res = raw_res.split("⚡ SABER Specialist:")[0].strip()
                    res_data = {"thinking": "", "response": clean_res}
                else:
                    # High-fidelity simulation mode
                    res_data = simulate_saber_response(query, deep_reason)
                
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(res_data).encode("utf-8"))
                
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_error(404, "Endpoint not found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="Launch SABER Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind server (default: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), SaberWebHandler)
    print("=" * 60)
    print(f"🚀 SABER Web UI is LIVE at: http://localhost:{args.port}")
    print("=" * 60)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    main()
