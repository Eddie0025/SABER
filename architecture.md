# SABER — System Architecture

> **SABER** (Specialist Agent-Based Expert Reasoning) is a modular multi-specialist AI framework built on open-weight 7B base models (Qwen 2.5-7B). SABER pairs **High-Rank Weight-Decomposed Low-Rank Adaptation (DoRA)** with **Verifiable-Fact-Augmented Group Relative Policy Optimization (GRPO)**. Domain-specialized expert models generate verified, chain-of-thought-backed answers grounded by a 2-pass asynchronous verification kernel (**Sentinel**) with offline/online knowledge base routing and synthesized by a meta-reasoning layer.

---

## End-to-End Pipeline

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR (bare Qwen 2.5-7B)                                 │
│  The base 7B model — no LoRA adapter. Uses its general LLM       │
│  capabilities for all pre-routing checks.                        │
│                                                                  │
│  1. Ambiguity Detection                                         │
│  2. Query Completeness Check (asks follow-ups if info missing)  │
│  3. Few-Shot Domain Classification with confidence gate          │
│  4. Specialist Selection + Task Decomposition                   │
│     (OR Generalist Fallback for out-of-domain queries)          │
│  5. Verification Tier Assignment                                │
│                                                                  │
│  Once routing is finalized, three things happen simultaneously:  │
│  ──► Sends simple routing ping to 0.5B Chatbot                  │
│  ──► Sends original query + specialist list to Meta-Reasoner    │
│  ──► Dispatches domain-specific sub-tasks to specialists        │
└───────┬──────────────────┬───────────────────────┬───────────────┘
        │                  │                       │
        │ simple ping      │ query + spec list     │ TASK_SIGNALs
        ▼                  ▼                       ▼
┌─────────────────┐  ┌─────────────────────┐  ┌───────────────────────────┐
│ CHATBOT (0.5B)  │  │ META-REASONER       │  │ SPECIALIST EXECUTION      │
│ Runs on CPU     │  │ (reads & holds)     │  │ (sequential)              │
│                 │  │                     │  │                           │
│                 │  │                     │  │ 6. FLAG → rewrite         │
│                 │  │                     │  │ 7. GREEN_CHIT → pass      │
└─────────────────┘  └─────────────────────┘  └─────────────┬─────────────┘
                                                            │
                              verified outputs arrive       │
                              one by one as specialists     │
                              complete                      │
                                                            │
                     ┌──────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────────────┐
│  META-REASONING LAYER (activates once ALL outputs received)      │
│  (Has CoT Maintainer plugged in for synthesis reasoning)         │
│                                                                  │
│  Already has: Original query (read at start)                     │
│  Now receives: All specialist claims + CoT chains                │
│                                                                  │
│  1. CLAIM EXTRACTION — list key claims from each specialist      │
│  2. CONFIDENCE ANALYSIS — weight specialist confidence levels    │
│  3. CONFLICT DETECTION — identify contradictions between specs   │
│  4. TRADEOFF EVALUATION — analyse disagreements                  │
│  5. RESOLUTION PATH — reconcile conflicting views                │
│  6. FINAL ANSWER — produce one coherent, detailed answer         │
│                                                                  │
│  Uses the original query to ensure the final answer directly     │
│  addresses what the user actually asked.                          │
└──────────────────────┬───────────────────────────────────────────┘
                       │ final synthesized answer
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│  DECISION LEDGER (written to audit log)                          │
│  - Full specialist responses, all flags, all corrections,        │
│    verification history, disagreements, meta-reasoning path,     │
│    final confidence score                                        │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
                 Final Answer → User (replaces chatbot message)
```

---

## Component Details

### 0. Chatbot (Qwen 2.5-0.5B)

**Model**: `Qwen/Qwen2.5-0.5B` (350M parameters)

A lightweight model that **loads with the chat interface** and runs on CPU. It does NOT fire immediately when the user sends a query — it waits for the **Orchestrator to finalize routing** and then receives a simple status ping.

Once pinged, the chatbot generates a **generic boilerplate status message** to keep the user occupied (e.g., *"Your query is being analyzed — processing your request..."*). This message is displayed to the user while the specialists work in the background.

- **Runs on CPU** — no GPU contention with the specialist pipeline.
- **Loads once at startup** — always warm, instant response once pinged.
- **Generic keeping-occupied strategy** — outputs standard status updates ("Analyzing...", "Running verification...", etc.) instead of domain-specific text.
- **Replaced transparently** — when the backend pipeline returns the real answer, it replaces the chatbot message in the UI.
- **Also handles simple greetings** — if the user sends a non-domain message ("hi", "thanks", etc.), the chatbot handles it directly with generic pleasantries without invoking the full pipeline.

This model does NOT participate in any reasoning, routing, or verification.

### 1. Orchestrator (bare Qwen 2.5-7B)

**File**: `saber/orchestrator.py`

The entry point. It is the **bare Qwen 2.5-7B base model** — no LoRA adapter, no domain fine-tuning. It uses its general-purpose LLM capabilities to handle all pre-routing checks: ambiguity detection, query completeness verification, and domain classification. It does NOT generate domain answers — it only validates, classifies, routes, and dispatches.

Once routing is finalized, the Orchestrator does three things simultaneously:
1. **Pings the 0.5B Chatbot** with a simple status update notification
2. **Sends the original query + activated specialist list** to the Meta-Reasoning Layer (it reads and holds quietly until all specialist outputs arrive)
3. **Decomposes the query into domain-specific sub-tasks** and dispatches them to the activated specialists

| Step | What It Does |
|------|-------------|
| **Ambiguity Detection** | Scores query ambiguity (0–1) based on word count, pronoun density, and domain keyword coverage. Queries ≥ 0.70 are rejected with a clarification request. |
| **Query Completeness Check** | Uses the 7B model to assess whether the query contains enough information for the specialists to produce a useful answer. If critical information is missing, the system asks targeted follow-up questions before proceeding (see below). |
| **Domain Classification** | Few-shot LLM classification using the 7B base model (see below). |
| **Confidence Gate** | If the classifier's confidence is < 95%, the query is re-classified with expanded context (full specialist descriptions + example queries). Routing must be near-100% accurate — wrong routing makes everything downstream useless. |
| **Specialist Selection + Task Decomposition** | Activates specialists whose relevance score ≥ 0.30. For multi-domain queries, the Orchestrator uses the 7B model to decompose the query into **domain-specific sub-tasks** (see below). |
| **Tier Assignment** | Assigns verification depth: Tier 0 (no checks), Tier 1 (2 checks), Tier 2 (4 checks), Tier 3 (6 checks). |

#### Query Completeness Check

Before any routing happens, the Orchestrator uses the 7B model to determine if the query has enough detail to produce a meaningful answer. The model evaluates:

- **Missing constraints** — e.g., *"Calculate the force"* but no mass or acceleration values provided.
- **Underspecified scope** — e.g., *"How do I secure my system?"* — which system? What threat model? What's already in place?
- **Missing context** — e.g., *"What's wrong with this code?"* but no code attached.

If the query is incomplete, the system returns **targeted follow-up questions** to the user — not a generic "please provide more detail" but specific questions about what's missing (e.g., *"What mass and acceleration values should I use?"*). The pipeline does NOT proceed until the user provides enough information. This prevents specialists from generating answers based on assumptions that may be wrong.

Once the query is deemed complete, it moves to domain classification.

#### Domain Classification — Few-Shot LLM Routing

The Orchestrator uses the bare Qwen 2.5-7B to classify queries. The classification prompt is built from two components:

**1. Specialist Reference Table**

A structured table is injected into the prompt listing every registered specialist, its domain name, and its specific areas of expertise. The model reads this table to understand what each specialist can handle:

```
| Domain        | Specializations                                                    |
|---------------|--------------------------------------------------------------------|
| science       | Physics, chemistry, math, calculus, quantum mechanics, thermodynamics |
| cyber         | Vulnerabilities, malware, firewalls, MITRE ATT&CK, CVEs, pentesting |
| coding        | Algorithms, debugging, optimization, code review, data structures    |
| finance       | Revenue analysis, EBITDA, portfolio management, valuation, hedging   |
| medical       | Diagnosis, treatment protocols, pathology, pharmacology, clinical    |
| architecture  | System design, microservices, Kubernetes, scaling, cloud infra       |
```

This table is dynamically generated from the Specialist Registry, so adding a new specialist automatically updates the routing table — no hardcoded lists to maintain.

**2. Synthetic Few-Shot Examples**

Before the actual query, the prompt includes a few passes of synthetic example queries with their correct routing decisions. These are hand-crafted to cover:

- **Clear single-domain queries** — showing straightforward classification
- **Edge cases** — queries that could plausibly belong to multiple domains
- **Cross-domain queries** — showing when multiple specialists should be activated
- **Trick queries** — e.g., "What's the half-life of this investment?" (finance, not science despite "half-life")

Example passes in the prompt:
```
Query: "Explain the MITRE ATT&CK framework's credential access tactics"
→ {"domains": [{"name": "cyber", "confidence": 0.98}]}

Query: "Calculate the derivative of e^(2x) * sin(x)"  
→ {"domains": [{"name": "science", "confidence": 0.99}]}

Query: "What's the risk exposure of our cloud infrastructure to supply chain attacks?"
→ {"domains": [{"name": "cyber", "confidence": 0.85}, {"name": "architecture", "confidence": 0.75}]}
```

The model sees the reference table + these synthetic examples, then classifies the actual user query. It outputs a JSON list of activated domains with confidence scores.

**Confidence Gate**: If the highest confidence score is **below 95%**, the Orchestrator re-runs classification with an **expanded prompt** — more synthetic examples, step-by-step reasoning instructions, and the full specialist capability descriptions. This two-pass approach ensures near-perfect routing accuracy.

**Generalist Fallback (Out-of-Domain queries)**: If after the two-pass classification the highest confidence score is *still* below the 0.30 activation threshold (e.g., for general knowledge or translation tasks like *"What is the capital of Japan?"*), the Orchestrator **bypasses the specialist pipeline entirely**. It uses the Qwen 7B base model with internet access to answer the query directly. Future work will introduce specialized agents (e.g., for creative writing) to handle these cases.

No keyword heuristic fallback is used. Routing accuracy is mission-critical and only the LLM has the semantic understanding to get it right.

#### Task Decomposition — Multi-Domain Query Handling

For **single-domain queries**, the original query is passed directly to the specialist as its task.

For **multi-domain queries**, the Orchestrator uses the 7B model to decompose the query into **domain-specific sub-tasks**. Each specialist's TASK_SIGNAL contains both:

- **The original full query** — for context, so the specialist understands the bigger picture
- **A focused sub-task** — reformulated in the specialist's domain language, telling it exactly what aspect to address

Example decomposition for *"What's the risk exposure of our cloud infrastructure to supply chain attacks?"*:

```
TASK_SIGNAL → cyber specialist:
  original_query: "What's the risk exposure of our cloud infrastructure
                   to supply chain attacks?"
  focused_task:   "Assess supply chain attack vectors, threat actors,
                   and vulnerability exposure for cloud-hosted systems."

TASK_SIGNAL → architecture specialist:
  original_query: "What's the risk exposure of our cloud infrastructure
                   to supply chain attacks?"
  focused_task:   "Evaluate the cloud architecture's resilience, defense-
                   in-depth posture, and single points of failure against
                   external dependency compromise."
```

This decomposition is cheap — one LLM call on the bare 7B that's already loaded from the classification step. The meta-reasoner later stitches the domain-specific answers back together using the original query as its guide.

### 2. Domain Specialists

**Files**: `saber/specialist.py` (base class), `saber/specialists/*.py` (implementations)

Each specialist is a separate Python class with its own LoRA adapter. Currently 6 specialists:

| Specialist | Domain | LoRA Adapter |
|-----------|--------|-------------|
| ScienceSpecialist | `science` | `models/science_v2/` |
| CybersecuritySpecialist | `cyber` | `models/cyber_v2/` |
| CodingSpecialist | `coding` | `models/coding_v2/` |
| FinanceSpecialist | `finance` | `models/finance_v2/` |
| MedicalSpecialist | `medical` | `models/medical_v2/` |
| ArchitectureSpecialist | `architecture` | `models/architecture_v2/` |

**How a specialist processes a query:**

1. Receives a `TASK_SIGNAL` from the pipeline.
2. Confirms understanding via `CONFIRMATION_SIGNAL`.
3. Loads its domain LoRA adapter via `LLMEngine` (context manager — only one model in memory at a time).
4. Generates its answer with the **CoT Maintainer** plugged in during execution — the reasoning chain is built step by step (`IDENTIFY → ANALYZE → HYPOTHESIZE → EVIDENCE → EVALUATE → CONCLUDE`).
5. Parses the raw LLM output into structured **Claim** objects (Pydantic models with `statement`, `confidence`, `domain`, `status`).
6. Packages claims + full CoT chain into a `COT_SIGNAL`.
7. The Sentinel runs its verification checks on this specialist's output (see §4).
8. If verification passes (GREEN_CHITs), the output is finalized. If flags are raised, the answer is rewritten.

### 3. Chain-of-Thought (CoT) Maintainer

**File**: `saber/cot_maintainer.py`

A bidirectional working memory module **plugged into each specialist during execution, and into the Meta-Reasoning Layer during synthesis**. The model reads from and writes to it as it reasons.

**Step Types**: `IDENTIFY` → `ANALYZE` → `HYPOTHESIZE` → `EVIDENCE` → `EVALUATE` → `CONCLUDE`

Each step records:
- `step_number`, `action`, `content`, `confidence`
- `evidence_refs` (sources cited)
- `depends_on` (which prior steps this step builds on — forms a DAG)
- `timestamp`

**Post-Processing Cleanup** (runs after specialist finishes):
- Deduplication: Steps with >85% text similarity are merged (keeps higher-confidence version).
- Loop detection: Prevents autoregressive repetition.
- Consecutive action consolidation: Merges adjacent steps with the same action type.

The cleaned CoT chain is exported into the COT_SIGNAL and travels with the claims through the rest of the pipeline.

### 4. Sentinel Verification Kernel

**File**: `saber/sentinel.py`

The independent verification authority. **Critically uses the unbiased base model** (not the fine-tuned LoRA adapter) to prevent self-checking bias.

**Two levels of verification on each specialist's output:**

#### Level 1: Signal Integrity (Cryptographic)
- Recomputes SHA-256 hash over the Signal payload.
- Compares against `integrity_hash` set when the Signal was created.
- Mismatch → immediate `FLAG_SIGNAL` with severity `CRITICAL` (tampering or corruption).

#### Level 2: Semantic Verification (LLM + Web Grounding)
- **Verification Batching**: Instead of verifying claims one by one, Sentinel batches all claims from a specialist into a single verification prompt (e.g., "Verify these 8 claims") to drastically reduce LLM call overhead. Latency remains an explicit tradeoff for high-tier checks, but batching prevents it from scaling linearly.
- **Online mode**: Queries the Chrome Search API for factual verification of claims. Search results are injected as "ground truth" into the verification prompt.
- **Offline mode**: Falls back to logical consistency checks only.
- **Targeted verification routing**: Different aspects get checked by different reviewers:
  - Task alignment → `task_relevance` check (Did the specialist actually answer the specific sub-task in the TASK_SIGNAL, or did it go on a factually correct tangent?)
  - Cyber content → `technical_accuracy` by cyber, `logical_reasoning` by science
  - Science content → `factual_accuracy` + `mathematical_reasoning` by science
  - Medical content → `clinical_accuracy` by medical, `logical_reasoning` by science
- Runs the configured number of verification cycles (tier-dependent).
- Each cycle: if all checks return `CONFIRMED` → `GREEN_CHIT`. If errors found → `FLAG_SIGNAL` with structured JSON (issue_type, severity, evidence, reasoning, proposed_fix).
- **Self-Correcting Rewrites**: When Sentinel flags an answer, it explicitly provides the correct grounded fact from the web in the `FLAG_SIGNAL` (via `proposed_fix`). The specialist simply rewrites its answer to integrate the provided correction, practically eliminating the risk of infinite rewrite loops.

#### CoT Step-Level Verification
- Checks that the first step is `IDENTIFY` and last step is `CONCLUDE`.
- Detects sharp confidence drops (>0.3 between consecutive steps).
- Verifies each step logically follows from the previous steps (LLM check).

### 5. Meta-Reasoning Layer

**File**: `saber/meta_reasoner.py`

Purely a synthesis engine — it does NOT generate domain answers.

**Two-phase operation:**

#### Phase 1: Read & Hold (happens at dispatch time)
When the Orchestrator finalizes routing, it sends two things to the Meta-Reasoner:
1. The **original user query** — so the meta-reasoner understands what the user actually asked.
2. The **list of activated specialists** — so it knows exactly how many outputs to expect.

The Meta-Reasoner reads these, holds them quietly, and waits. It does nothing until all expected specialist outputs have arrived.

#### Phase 2: Synthesis (activates once ALL specialist outputs received)
Once every activated specialist has sent its verified claims + CoT chain, the Meta-Reasoner checks them all off against its expected list and begins synthesis. Because it already knows the original query, it can ensure the final answer directly addresses what the user asked — not just what the specialists chose to talk about.

**The Meta-Reasoner has its own CoT Maintainer plugged in.** Instead of suffering from context bloat by trying to synthesize massive raw specialist CoTs in one shot, it uses its own CoT module to step-by-step reason through the claims, evaluate conflicts, and plan the final synthesis.

Structured 6-section analysis:

1. **CLAIM EXTRACTION** — Lists the key claims from each specialist.
2. **CONFIDENCE ANALYSIS** — Assesses specialist confidence levels and assigns weights.
3. **CONFLICT DETECTION** — Identifies contradictions or gaps between specialists. Writes "None" if clean.
4. **TRADEOFF EVALUATION** — Analyses tradeoffs when specialists disagree.
5. **RESOLUTION PATH** — Explains how conflicting views are reconciled.
6. **FINAL ANSWER** — The single, coherent, detailed answer to the user's query.

**Disagreement Detection**: If specialist confidence scores have a spread > 0.3, it flags a disagreement and logs it.

**Confidence Scoring**:
```
final_confidence = avg(specialist_claim_confidences) − (0.02 × total_flags_raised) − (0.01 × verification_cycles_run)
```

### 6. Signal Schema

**File**: `saber/signal.py`

All inter-component communication uses typed, Pydantic-validated Signal objects. No raw strings or ad-hoc dicts.

**Signal Types** in lifecycle order:
```
QUERY_SIGNAL         →  User query enters system
TASK_SIGNAL          →  Orchestrator assigns task to specialist
CONFIRMATION_SIGNAL  →  Specialist acknowledges task
COT_SIGNAL           →  Specialist returns claims + CoT chain
VERIFICATION_SIGNAL  →  Sentinel check result (GREEN_CHIT)
FLAG_SIGNAL          →  Sentinel flags an error
PATCH_SIGNAL         →  Correction applied
AUDIT_SIGNAL         →  Internal audit events
HEALTH_SIGNAL        →  Specialist health updates
```

Every Signal has:
- `signal_id` (UUID), `signal_type`, `query_id`, `source_id`, `target_id`
- `timestamp`, `version` (2.0.0), `priority`, `confidence`
- `integrity_hash` (SHA-256 of payload)
- `payload` (arbitrary dict)

### 7. LLM Engine

**File**: `saber/llm_engine.py`

Context-managed dynamic model loader. Only one model in memory at any time.

```python
with LLMEngine("models/science_v2") as engine:
    response = engine.generate(prompt, system_prompt=system_prompt)
# Model auto-unloaded, GPU cache cleared
```

- Auto-detects PEFT/LoRA adapters (reads `adapter_config.json`).
- Device fallback: CUDA → MPS → CPU.
- Weight caching mode for benchmarks (`SABER_KEEP_MODELS_LOADED=1`).
- Generation: `temperature=0.3`, `top_p=0.9`, `repetition_penalty=1.05`.
- Handles Qwen ChatML format with `<|im_end|>` stop tokens.

### 8. Audit Logger & Decision Ledger

**File**: `saber/audit.py`

Thread-safe, append-only JSONL logger. Every event in the system is recorded.

**Decision Ledger** (one per query):
- Query text + selected specialists
- Initial specialist responses (claims + confidence)
- All flags raised + all corrections applied
- Per-cycle verification history
- Specialist disagreement records
- Meta-reasoning path (all 6 sections)
- Final resolution text + final confidence

**Failure Taxonomy** (`saber/errors.py`):
| Category | When |
|----------|------|
| `ROUTING_FAILURE` | No specialists activated |
| `SPECIALIST_FAILURE` | Specialist crash or task rejection |
| `VERIFICATION_FAILURE` | Signal integrity or semantic check fails |
| `CONSENSUS_FAILURE` | No valid specialist outputs |
| `SYNTHESIS_FAILURE` | Meta-reasoning synthesis fails |
| `SYSTEM_FAILURE` | Infrastructure failures |

### 9. Chat Context Memory (Multi-turn Support)

**File**: `saber/memory.py`

To support multi-turn conversations without overflowing the context window, SABER uses a rolling Chat Context Memory:
- **Injection**: The Orchestrator injects this summarized memory into the prompt during Ambiguity/Completeness checks, and passes it down to the specialists so they have full conversation history.
- **Updates**: After generating the final answer, the Meta-Reasoner adds a summarized entry of the user's query and the final answer to the memory ledger.
- **Compression**: To save space and preserve attention, every 5 new entries, the Meta-Reasoner compresses and re-summarizes the entire history into a dense, high-level context block.

---

## Training Pipeline

**Files**: `saber/training/trainer.py`, `saber/training/dataset_loader.py`, `scripts/build_offline_kb.py`, `scripts/validate_kb_coverage.py`

### 1. Two-Phase Training Paradigm

#### Phase 1: High-Rank Weight-Decomposed LoRA (DoRA SFT)
- **Base Model**: `Qwen/Qwen2.5-7B-Instruct`
- **Method**: DoRA (`use_dora=True`, rank $r=64$, $\alpha=128$, dropout $0.05$)
- **Target Modules**: All linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) — expanding trainable parameter capacity to **~3.5% (~245M parameters)**.
- **Precision & Optimization**: Native `bfloat16`, batch size per device = 8, gradient accumulation = 4 (effective batch = 32), 3 epochs, learning rate = 2e-4.
- **Hardware Acceleration**: Optimized for single NVIDIA H100 80GB GPU.
- **Format Normalization**: On-the-fly SFT chat template alignment matching benchmark structure (`Question: ... Options: A: ... Answer the following multiple choice question. The last line of your response MUST strictly follow the format: ANSWER: LETTER`).

#### Phase 2: Verifiable-Fact-Augmented GRPO Reinforcement Learning
- **Group Rollouts**: $G = 4\text{--}8$ parallel trajectory generations per prompt.
- **Composite Reward Signal**:
  $$\text{Reward} = R_{\text{Format}} (+1.0) + R_{\text{Outcome}} (+2.0) - \lambda \cdot R_{\text{Sentinel\_Factuality}}$$
  - **Format Reward ($R_{\text{Format}}$)**: $+1.0$ if trajectory terminates with strict `ANSWER: [A-D]`.
  - **Outcome Reward ($R_{\text{Outcome}}$)**: $+2.0$ if extracted option matches ground truth.
  - **Sentinel Factuality Reward ($R_{\text{Sentinel\_Factuality}}$)**:
    - Direct Contradiction against reference passage: **$-1.5$ penalty**
    - Direct Grounded Support from reference passage: **$+0.5$ bonus**
    - Uncovered / External Knowledge Claim: **$0.0$ neutral pass-through** (prevents penalizing valid background reasoning).

#### Offline Knowledge Base (0ms RL Grounding)
- **Indexing**: 139,973 reference support passages extracted directly from dataset context fields into local indexed SQLite databases (`data/offline_kb/*_kb.db`).
- **Pre-flight Audit**: Substantive reference text coverage $>99.0\%$ across all 8 training domains.
- **Numeric-Safe Caching**: Cache key pre-guarding (`[guard]::clean_query`) preserving numerical, percentage, date, and CVE statistical precision.
- **Hysteresis Audit Loop**: Rollout Disagreement Rate (RDR $> 5\%$) dynamically decays $\lambda \rightarrow 0.1$ for 100 steps and auto-restores to $0.5$ once signal stability returns, monitored by TensorBoard Neutral Claim Ratio (NCR).

---

### Training ↔ Evaluation Separation (Zero Data Leakage)

| Domain | Trained On | Evaluated On | Overlap |
|--------|-----------|-------------|---------|
| Science | ScienceQA, SciQ, CAMEL-AI | **GPQA Diamond** (198 cases) | None |
| Cyber | MITRE STIX, CyberQA, Synthetic ATT&CK | **SecBench** (100 cases) | None |
| Coding | APPS, CodeContests, LeetCode, CodeFeedback | **HumanEval** (164 cases) | None |
| Finance | Finance-Alpaca, ConvFinQA, Finance-Instruct | **FinQA Math** (80 cases) | None |
