# SABER v2.0 — Bulletproof Master Technical Implementation Plan

> **SABER** (Specialist Agent-Based Expert Reasoning) is a SOTA multi-specialist AI framework built on a single resident open-weight 7B base model (`Qwen2.5-7B-Instruct`). SABER combines **High-Rank Weight-Decomposed Low-Rank Adaptation (DoRA Phase 1 SFT)** with **Verifiable-Fact-Augmented Group Relative Policy Optimization (GRPO Phase 2 RL)** across 5 specialized domain experts (`Science`, `Cybersecurity`, `Finance`, `Coding`, `Architecture`) and 2 core orchestrating layers (`Orchestrator`, `Meta-Reasoner`).

---

## 1. System Architecture & Component Specifications

### 1.1 Single-Base VRAM Hot-Swap Architecture

SABER runs on a **Single Resident 7B Base Model** in VRAM (~4.7 GB VRAM footprint in 4-bit quantization, native `bfloat16` on H100):
- **Zero Multi-Model RAM Contention**: Only 1 base model (`Qwen2.5-7B-Instruct`) resides in memory.
- **Dynamic DoRA Adapter Hot-Swapping**: Adapters (`models/orchestrator_v2`, `models/science_v2`, etc.) are plugged on demand with **<50ms swap overhead**.

```
                                USER MESSAGE
                                     │
                                     ▼
          ┌─────────────────────────────────────────────────────┐
          │  RESIDENT BASE MODEL (bare Qwen2.5-7B-Instruct)     │
          │  - Single resident 7B base model in VRAM            │
          └──────────────────────────┬──────────────────────────┘
                                     │
                 2-TIERED INTENT GATE (is_casual_chat)
                 - Tier 1: Pattern Match & Dictionary (<1ms)
                 - Tier 2: 7B LLM Semantic Intent Gate (<15ms)
                                     │
                  ┌──────────────────┴──────────────────┐
                  │ CASUAL_CHAT                         │ DOMAIN_QUERY
                  ▼                                     ▼
      [ Bare 7B Native Output ]           ┌───────────────────────────────────┐
      - Instant warm response (<30ms)     │ 1. Plug Orchestrator DoRA Adapter │
      - Zero adapter loading overhead     │    -> Dissect & route to domains  │
                                          │ 2. Hot-Swap Specialist Adapter    │
                                          │    -> Deep CoT Reasoning & KB     │
                                          │ 3. Sentinel Grounding & Synthesis │
                                          └─────────────────┬─────────────────┘
                                                            │
                                                            ▼
                                                     FINAL RESPONSE
                                              (With Dynamic Safety Footer)
```

---

### 1.2 Component Breakdown

#### A. 2-Tiered Base-First Intent Gate (`saber/orchestrator.py`)
- **Tier 1 (Fast Pattern Match <1ms)**: Alphanumeric normalized dictionary matching for greetings, pleasantries, closing remarks (`hi`, `hello`, `thanks!`, `good morning`, `who created you`).
- **Tier 2 (7B Semantic Intent Gate <15ms)**: 4-token prompt gate evaluating semantic intent for unstructured casual talk.
- **Casual Chat Fast Path**: Bare 7B base model responds natively in **<30ms** with zero adapter loading overhead.

#### B. Orchestrator DoRA Adapter (`models/orchestrator_v2`)
- Activated strictly when a `DOMAIN_QUERY` is identified.
- **Few-Shot Semantic Intent Routing**: Dissects polysemous or complex queries (e.g. computer virus -> cyber vs biological virus -> science) and outputs structured JSON domain activations.
- **Ambiguity Detection**: Rejects vague queries ($\ge 0.70$ threshold) with clarification prompts.

#### C. Domain Specialist Experts (`saber/specialists/*`)
Five specialized domain experts executed via high-rank DoRA adapters ($r=128, \alpha=256$, ~500M parameters):
1. **Science Specialist** (`saber/specialists/science.py`): Physics, chemistry, calculus, biology, GPQA Diamond.
2. **Cybersecurity Specialist** (`saber/specialists/cybersecurity.py`): Vulnerability analysis, exploit payloads, MITRE ATT&CK, SecBench.
3. **Finance Specialist** (`saber/specialists/finance.py`): EBITDA calculations, 10-K SEC filings, portfolio valuation, FinQA Math.
4. **Coding Specialist** (`saber/specialists/coding.py`): Algorithmic synthesis, Python code generation, HumanEval.
5. **Architecture Specialist** (`saber/specialists/architecture.py`): Microservices design, Kubernetes, Kafka, gRPC, distributed systems.

#### D. Sentinel Verification Kernel (`saber/sentinel.py`)
Independent verification authority operating via **External-Evidence Grounding** (not internal LLM memory) to eliminate self-checking bias:
- **Fast Local SQLite KB Lookup (<5ms)**: Queries indexed SQLite databases (`data/offline_kb/*_kb.db`) using **Numeric-Safe Cache Guarding** (`[numeric_guards]::clean_query`).
- **Live Web Grounding & Dynamic Auto-Caching**: Searches live web for un-cached facts and **automatically writes snippets back to local SQLite KB** (`save_to_local_kb`) for instant 0ms future hits.
- **Self-Correcting Rewrites**: Provides grounded corrections in `FLAG_SIGNAL` payloads for specialist auto-rewrites.

#### E. Meta-Reasoning Layer (`saber/meta_reasoner.py`)
Context-aware synthesis coordinator:
- **Format-Aware Response Structuring**:
  - **MCQ Tasks**: CoT reasoning trace + `ANSWER: <LETTER>`.
  - **Coding Tasks**: Executable ```python code block.
  - **Open-Ended Tasks**: Structured 6-section analysis (`CLAIM EXTRACTION`, `CONFIDENCE ANALYSIS`, `CONFLICT DETECTION`, `TRADEOFF EVALUATION`, `RESOLUTION PATH`, `FINAL ANSWER`).
- **Dynamic Safety Footer Attachment**:
  - **Online Mode**: `⚡ Verified by SABER Sentinel (Online Web Grounded & Dynamic KB)`
  - **Offline Mode**: `🔒 Verified by SABER Sentinel (Offline Local KB Mode — Air-Gapped)`

#### F. Thread-Safe Audit Ledger (`saber/audit.py`)
JSON-Lines audit log (`logs/audit.jsonl`) recording complete query trajectories, signals, Sentinel flags, verification passes, and confidence metrics.

---

## 2. Dataset Sourcing & 260,000 Record Distribution

| Domain | Source Datasets | Raw Volume | Pristine Filtered Volume |
|---|---|---|---|
| **Science** | ScienceQA, SciQ, CAMEL-AI Physics/Math, NuminaMath-CoT | ~65,000 | **40,000 CoT Records** |
| **Cybersecurity** | MITRE ATT&CK STIX 2.1, Sec-Instruct, CyberQA, Synthetic Vulnerability | ~45,000 | **40,000 CoT Records** |
| **Finance** | ConvFinQA, FinQA, Finance-Alpaca, SEC 10-K Transcripts | ~42,000 | **40,000 CoT Records** |
| **Coding** | APPS, CodeContests, LeetCode, CodeFeedback-Filtered | ~60,000 | **40,000 CoT Records** |
| **Architecture** | SystemDesign-CoT, Kubernetes/Kafka Specs, Microservice Bench | ~45,000 | **40,000 CoT Records** |
| **Meta-Reasoner** | Multi-Specialist Synthetic Conflict & Tradeoff Reconciliation | ~30,000 | **30,000 CoT Records** |
| **Orchestrator** | Intent Disambiguation, Polysemous Routing, Ambiguity Corpus | ~30,000 | **30,000 CoT Records** |
| **TOTAL** | **High-Grade Multi-Domain CoT Corpus** | **~317,000** | **260,000 Records** |

---

## 3. Two-Phase Training Paradigm

### 3.1 Phase 1: High-Rank Weight-Decomposed LoRA (DoRA SFT)
- **Base Model**: `Qwen/Qwen2.5-7B-Instruct`
- **Method**: DoRA (`use_dora=True`, rank $r=128$, $\alpha=256$, dropout $0.05$)
- **Target Modules**: All 7 linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) — **~500M trainable parameters (~7.2% of model)**.
- **Optimization**: Native `bfloat16`, batch size per device = 8, gradient accumulation = 4 (effective batch = 32), 4 epochs, learning rate = 2e-4, PyTorch gradient checkpointing enabled.
- **Hardware Acceleration**: Single NVIDIA H100 80GB GPU (~13.5 hours total runtime).

### 3.2 Phase 2: Verifiable-Fact-Augmented GRPO Reinforcement Learning
- **Group Rollouts**: $G = 8$ parallel trajectory generations per prompt.
- **Composite Reward Signal**:
  $$\text{Reward} = R_{\text{Format}} (+1.0) + R_{\text{Outcome}} (+2.0) - \lambda \cdot R_{\text{Sentinel\_Factuality}}$$
  - **Definitive Tasks (Science, Cyber, Finance)**: Format (+1.0 for ANSWER: [A-D]), Outcome (+2.0 for ground truth match), Sentinel Factuality (-1.5 contradiction / +0.5 support / 0.0 neutral).
  - **Open-Ended Tasks (Coding, Architecture, Meta, Orchestrator)**: Sandboxed Code Execution (+2.0 with 2s timeout), CoT Completeness (+1.0), Token Repetition Penalty (-1.5 for 3-gram uniqueness <60%).

---

## 4. Benchmark & Zero-Leakage Evaluation Matrix

### 4.1 Dual-Tier Benchmark Hierarchy (Frontier vs. Mid-Tier)

To benchmark SABER comprehensively against both frontier closed models (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro) and open-source base models (Qwen2.5-7B, Llama-3.1-8B), evaluation is conducted across a two-tier difficulty spectrum:

| Domain Expert Area | Mid-Tier Baseline Benchmarks (Fast Iterative Testing) | Top-Tier Frontier Benchmarks (High Difficulty Ceiling) | Evaluated Capability | SABER Target Score |
| :--- | :--- | :--- | :--- | :---: |
| **🔬 Science** | **SciQ / ScienceQA**<br>*(13,600+ college/HS questions)* | **GPQA Diamond**<br>*(198 PhD-level "Google-proof" questions)* | Multi-step physics, chemistry, & calculus reasoning | **$> 65.0\%$** |
| **🛡️ Cybersecurity** | **CyberMetric-800**<br>*(800 security & vuln QA)* | **SecBench / SecQA-Hard**<br>*(100 multi-stage vulnerability exploits)* | MITRE ATT&CK TTPs, payload analysis, exploit prevention | **$> 82.0\%$** |
| **💻 Coding** | **HumanEval / MBPP**<br>*(164 Python / 974 basic algorithms)* | **SWE-bench Verified / LiveCodeBench**<br>*(500 real GitHub issues / competitive)* | Repository-level bug fixes, unit test compliance (Pass@1) | **$> 78.0\%$** |
| **📈 Finance** | **Finance-Alpaca**<br>*(20,000 corporate & accounting QA)* | **FinQA / ConvFinQA**<br>*(SEC 10-K filings & numerical reasoning)* | Financial report parsing, numerical math & NPV/EBITDA | **$> 75.0\%$** |
| **🏗️ Architecture** | **SystemDesign-Basic**<br>*(100 core microservice patterns)* | **ArchBench-Hard**<br>*(50 trade-off system scaling specs)* | High-availability design, Kubernetes, Kafka, gRPC trade-offs | **$> 80.0\%$** |

---

### 4.2 Multi-Domain Evaluation Strategy

Multi-domain prompts require **cross-specialist coordination** (activating 2 or more domain specialists simultaneously, followed by Meta-Reasoner synthesis). SABER evaluates multi-domain intelligence using 4 core cross-domain intersections:

```
                          MULTI-DOMAIN INTERSECTION MATRIX
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
┌──────────────┐                 ┌──────────────┐                 ┌──────────────┐
│  QUANT-FIN   │                 │  CYBER-ARCH  │                 │  SCI-COMP    │
│  (Fin + Code)│                 │ (Cyber+Arch) │                 │(Science+Code)│
└──────────────┘                 └──────────────┘                 └──────────────┘
```

1. **Finance + Coding (Quant Financial Engineering)**:
   - *Example*: *"Write a Python script to compute Black-Scholes option pricing with greeks and handle zero volatility edge cases."*
2. **Cybersecurity + System Architecture (Zero-Trust Infrastructure)**:
   - *Example*: *"Design a zero-trust Kubernetes microservices architecture to mitigate CVE-2023-24380 with network policies."*
3. **Science + Coding (Scientific Computation)**:
   - *Example*: *"Implement a Runge-Kutta 4th Order numerical integrator in Python to model a damped harmonic oscillator."*
4. **Finance + Cybersecurity (Smart Contract Audit)**:
   - *Example*: *"Audit an Automated Market Maker (AMM) contract for reentrancy vulnerabilities and calculate impermanent loss."*

---

### 4.3 Automated Open-Ended Evaluation Methodology

Open-ended answers (code blocks, architectural designs, trade-off analyses) are evaluated without human graders using a 3-mode automated protocol:

#### Mode A: Sandboxed Code Execution (Coding & Math)
- Extracts executable code blocks (```python) and executes them inside an isolated `multiprocessing.Process` with a 2.0s hard timeout.
- **Metric**: **Pass@1 Rate** (Binary 1.0 for passing unit test suite; 0.0 for syntax errors, timeouts, or assertion failures).

#### Mode B: Rubric-Based LLM-as-a-Judge (Open-Ended Synthesis)
- Uses a strong judge model (`GPT-4o` or `Claude 3.5 Sonnet`) evaluating outputs on a strict **5-Dimensional 25-Point Rubric**:
  1. *Factual Accuracy* (1-5): Are the technical facts correct?
  2. *Completeness* (1-5): Does the response answer all sub-parts of the prompt?
  3. *Structural Coherence* (1-5): Is the answer logically formatted?
  4. *Domain Depth* (1-5): Does the response contain expert-level detail without AI fluff?
  5. *Cross-Domain Accord* (1-5): Do multi-specialist contributions agree cleanly without contradiction?
- **Metric**: **Quality Score ($S \in [0.0, 1.0]$)**.

#### Mode C: Sentinel Contradiction Rate (Grounding Integrity)
- Passes open-ended answers through Sentinel to check against offline ground-truth SQLite passages or DuckDuckGo web snippets.
- **Metric**: **Grounding Accuracy Rate (%)** and **Contradiction Rate (%)**.

---

## 5. Sequential Execution Command Reference

When GPU training is initiated, run the 5 pipeline scripts sequentially:

```bash
git pull

# Step 1: Download & build 260,000 CoT dataset corpus
python3 scripts/1_build_datasets.py

# Step 2: Build & audit offline SQLite KBs (139k+ support passages)
python3 scripts/2_build_kb.py

# Step 3: Phase 1 High-Rank DoRA SFT (train Science domain specialist)
python3 scripts/3_train_dora.py --domain science --epochs 4

# Step 4: Phase 2 Verifiable-Fact-Augmented GRPO Reinforcement Learning
python3 scripts/4_train_grpo.py --domain science --generations 8

# Step 5: Run unified 5-mode benchmark evaluation
python3 scripts/5_run_benchmark.py
```
