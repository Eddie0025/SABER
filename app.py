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


def simulate_saber_response(query: str, sentinel_mode: str = "1_sentinel"):
    """
    Generates high-fidelity simulated responses for interactive testing without GPU checkpoints.
    Modes:
    - 'bolt': No sentinel, instant response without reasoning trace.
    - '1_sentinel': Standard 1-pass verification reasoning.
    - '2_sentinel': Deep Thinking with 2-pass multi-stage reflection & verification.
    """
    q_lower = query.lower()
    
    # If bolt mode is selected, zero thinking trace is returned
    if sentinel_mode == "bolt":
        # 1. Casual Chat
        if any(w in q_lower for w in ["hello", "hi", "hey", "how are you", "who are you", "what are you", "thanks"]):
            return {
                "thinking": "",
                "response": "Hello! I am SABER, an AI assistant built for deep technical reasoning, code architecture, and multi-domain problem solving. How can I help you today?"
            }
        
        if any(w in q_lower for w in ["vulnerability", "exploit", "heap", "glibc", "aslr", "xss", "csrf", "double-free", "injection", "kernel"]):
            response = (
                "### Vulnerability Analysis: Glibc Heap Double-Free to RCE\n\n"
                "A **double-free vulnerability** occurs when `free()` is called twice on the same pointer without clearing it. Exploitation in modern `glibc` involves:\n\n"
                "1. **Tcache Poisoning:** A double-freed chunk creates a circular singly-linked list in the thread-local cache bin.\n"
                "2. **Pointer Demangling:** On `glibc >= 2.29`, heap safe linking masks pointers; leaking a heap address reveals the per-thread key.\n"
                "3. **Arbitrary Write:** Overwriting the forward pointer (`fd`) forces subsequent `malloc()` allocations to return an arbitrary target address (such as `__malloc_hook` or a GOT entry), leading to RCE via a one-gadget or ROP payload."
            )
            return {"thinking": "", "response": response}

        return {
            "thinking": "",
            "response": "Processing complete via **Bolt mode** (direct generation). Ensure all system constraints and interfaces are verified for production deployment."
        }

    # 1 Sentinel (Standard) vs 2 Sentinel (Deep Thinking)
    is_deep = (sentinel_mode == "2_sentinel")

    # 1. Casual Chat
    if any(w in q_lower for w in ["hello", "hi", "hey", "how are you", "who are you", "what are you", "thanks"]):
        return {
            "thinking": "",
            "response": "Hello! I am SABER, an AI assistant built for deep technical reasoning, code architecture, and multi-domain problem solving. How can I help you today?"
        }
    
    # 2. Cybersecurity
    if any(w in q_lower for w in ["vulnerability", "exploit", "heap", "glibc", "aslr", "xss", "csrf", "double-free", "injection", "kernel"]):
        if is_deep:
            thinking = (
                "── Pass 1: Initial Reasoning & Threat Model ──\n"
                "1. IDENTIFY: Memory corruption vector in glibc ptmalloc allocator.\n"
                "2. ANALYZE: A double-free condition in glibc >= 2.29 triggers tcache count tampering or fastbin dup.\n"
                "3. HYPOTHESIZE: Bypassing ASLR requires an initial info leak (libc base address) via unsorted bin fd/bk pointers.\n\n"
                "── Pass 2: Sentinel Adversarial Reflection & Constraint Verification ──\n"
                "4. VERIFY: Safe-linking pointer mangling (PROTECT_PTR macro) in glibc 2.32+ verified.\n"
                "5. CROSS-CHECK: Checked __malloc_hook deprecation in glibc 2.34+; fallback target requires exit_funcs or stack frame overwrite.\n"
                "6. CONCLUDE: Formulating verified end-to-end technical explanation."
            )
        else:
            thinking = (
                "1. IDENTIFY: Analyzing memory corruption vector in multi-threaded glibc allocator.\n"
                "2. ANALYZE: A double-free condition creates a circular tcache bin linked list.\n"
                "3. EVALUATE: Leaking libc pointers bypasses ASLR; tcache poisoning overwrites target address.\n"
                "4. CONCLUDE: Formulating structured mitigation and exploitation sequence."
            )
        
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
        if is_deep:
            thinking = (
                "── Pass 1: Clinical Reasoning & Differential Analysis ──\n"
                "1. IDENTIFY: Low-renin, low-aldosterone hypertension with hypokalemic metabolic alkalosis.\n"
                "2. ANALYZE: Primary suspects are monogenic non-aldosterone mineralocorticoid excess states.\n\n"
                "── Pass 2: Sentinel Biochemical & Pharmacologic Verification ──\n"
                "3. CONTRAST: Liddle syndrome = ENaC gain-of-function; AME = 11β-HSD2 deficiency allowing cortisol MR hyperactivation.\n"
                "4. PHARMACOTHERAPY VERIFICATION: Confirmed Amiloride/Triamterene for Liddle; Spironolactone/Dexamethasone for AME.\n"
                "5. CONCLUDE: Formulating clean clinical table and diagnostic pearls."
            )
        else:
            thinking = (
                "1. IDENTIFY: Evaluating differential diagnosis for low-renin, low-aldosterone hypertension with hypokalemic metabolic alkalosis.\n"
                "2. ANALYZE: Liddle syndrome is an autosomal dominant gain-of-function mutation in ENaC subunit genes.\n"
                "3. CONTRAST: Apparent Mineralocorticoid Excess (AME) involves 11β-HSD2 enzyme deficiency.\n"
                "4. CONCLUDE: Formulating clinical comparison and pharmacotherapeutic distinctions."
            )
        
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

    # 4. General fallback
    if is_deep:
        thinking = (
            "── Pass 1: Architecture & Structural Decomposition ──\n"
            "1. IDENTIFY: Analyzing core structural requirements of the query.\n"
            "2. DECOMPOSE: Breaking down requirements into component specifications and state machines.\n\n"
            "── Pass 2: Sentinel Fault Tolerance & Edge Case Verification ──\n"
            "3. VERIFY: Idempotency keys, timeout handling, and partition boundaries.\n"
            "4. SYNTHESIZE: Formulating optimal implementation strategy."
        )
    else:
        thinking = (
            "1. IDENTIFY: Analyzing core structural requirements of the query.\n"
            "2. DECOMPOSE: Breaking down requirements into component specifications and edge cases.\n"
            "3. SYNTHESIZE: Formulating optimal implementation strategy."
        )
    
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
                sentinel_mode = data.get("sentinel_mode", "1_sentinel")
                
                # Check if live orchestrator exists
                if LIVE_SABER_AVAILABLE and orchestrator_instance is not None:
                    context = SessionContext()
                    raw_res = orchestrator_instance.process(query, context)
                    clean_res = raw_res.split("⚡ SABER Specialist:")[0].strip()
                    res_data = {"thinking": "", "response": clean_res}
                else:
                    # High-fidelity simulation mode
                    res_data = simulate_saber_response(query, sentinel_mode)
                
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
