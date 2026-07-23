# SABER (Specialized Architecture for Business, Engineering, & Reasoning)
## Comprehensive Technical Specification & System Reference

---

# 1. Executive System Summary & Architectural Vision

SABER is an autonomous, multi-agent AI framework engineered to solve the fundamental failure modes of generalist Large Language Models (LLMs)—specifically domain hallucination, mathematical error compounding, context window decay, and shallow technical reasoning. 

Rather than relying on a single monolithic 70B+ parameter model, SABER employs a **Dynamic Mixture-of-Specialists (MoS)** architecture backed by a 4-Tier Verification Guard (Sentinel) and a Verifiable-Fact Reinforcement Learning (GRPO) Meta-Reasoner.

```
                                  [ USER QUERY ]
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │   1. INTENT & ROUTING GATE  │
                         │   - Polysemy Disambiguator  │
                         │   - Ambiguity Scorer (0..1) │
                         │   - Intent Classifier       │
                         └──────────────┬──────────────┘
                                        │
            ┌───────────────────────────┼───────────────────────────┐
            │ (Domain Activation Score ≥ 0.40)                      │
            ▼                           ▼                           ▼
┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐
│   SPECIALIST: CYBER   │   │   SPECIALIST: FINANCE │   │   SPECIALIST: CODING  │
│  (DoRA r=64 / α=128)  │   │  (DoRA r=64 / α=128)  │   │  (DoRA r=64 / α=128)  │
└───────────┬───────────┘   └───────────┬───────────┘   └───────────┬───────────┘
            │                           │                           │
            └───────────────────────────┼───────────────────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │  2. SENTINEL VERIFICATION   │
                         │   - Factuality Guard        │
                         │   - Numeric Consistency     │
                         │   - SQLite KB Retrieval     │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │   3. META-REASONER LAYER    │
                         │   - Multi-Agent Synthesis   │
                         │   - Claim Conflict Graph    │
                         │   - GRPO Verifiable RL      │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                                 [ FINAL RESPONSE ]
```

---

# 2. Layer-by-Layer Technical Specification

## 2.1 Layer 1: Orchestrator & Intent Gating Engine (`saber/orchestrator.py`)

The Orchestrator acts as the central ingress router for all queries, processing input through a three-stage pipeline before executing specialist models:

1. **Intent Gate (2-Tiered)**:
   - **Tier 1 (Fast Exact Match)**: Evaluates queries against a hash set of casual greetings, slang, and pleasantries (`hi`, `hello`, `who are you`, `thanks`). Execution latency is $< 1\text{ms}$. If triggered, returns direct conversational output without waking specialist models or GPU memory.
   - **Tier 2 (Semantic Intent Gate)**: For ambiguous phrases, invokes an ultra-fast LLM prompt to classify input into `CASUAL_CHAT` or `DOMAIN_QUERY`.

2. **Polysemous Contextual Disambiguator**:
   - Resolves ambiguous technical terms whose domain changes based on surrounding context.
   - *Example*: The term `"virus"` triggers **Cyber** if keywords like `{"computer", "network", "smb", "payload", "port", "malware", "exe"}` are present.

3. **Domain Scoring & Specialist Activation**:
   - Scores each registered specialist using keyphrase stemming (`PorterStemmer` equivalent via `_stem()`) and exact phrase matching.
   - Primary domain triggers add a base activation of $+0.70$. Secondary hits add proportional confidence.
   - Specialists with `score >= 0.40` (the default `activation_threshold`) are marked as activated.
   - In **Benchmark Mode** (`SABER_BENCHMARK_MODE=1`), multi-activation is suppressed, forcing single-domain routing to isolate specialist metrics.

---

## 2.2 Layer 2: Domain Specialists & Fine-Tuning Methodology (`saber/training/trainer.py`, `scripts/3_train_dora.py`)

SABER contains 4 active specialized domain experts, plus Meta-Reasoner and Orchestrator tuning targets. Each specialist uses **Weight-Decomposed Low-Rank Adaptation (DoRA)** fine-tuned on base `Qwen/Qwen2.5-7B-Instruct`.

### DoRA Mathematical Formulation
DoRA decouples the pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ into a magnitude vector $m \in \mathbb{R}^{1 \times k}$ and a directional matrix $V \in \mathbb{R}^{d \times k}$:

$$W = m \odot \frac{V + \Delta V}{\|V + \Delta V\|_c}$$

Where directional updates $\Delta V$ are parameterized via standard LoRA low-rank decomposition matrices $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$:

$$\Delta V = \frac{\alpha}{r} (B \cdot A)$$

### Training Hyperparameters & Execution Settings

| Hyperparameter | Value / Specification | Rationale |
| :--- | :--- | :--- |
| **Base Model** | `Qwen/Qwen2.5-7B-Instruct` | State-of-the-art 7B reasoning foundation with native 32k context |
| **DoRA Rank ($r$)** | `64` | High rank enables deep domain-specific feature updates |
| **DoRA Alpha ($\alpha$)** | `128` | Scaling factor ($\alpha / r = 2.0$) for strong adapter weight injection |
| **Target Modules** | All Linear (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) | Comprehensive adapter coverage across self-attention and MLP blocks |
| **Per-Device Batch Size** | `8` | Fits safely within 80GB H100 VRAM alongside FlashAttention-2 |
| **Gradient Accumulation**| `2` | Maintains an effective batch size of **16** |
| **Optimizer** | `AdamW (paged_adamw_32bit)` | Prevents memory fragmentation during backward pass |
| **Learning Rate** | $2.0 \times 10^{-4}$ | Cosine annealing schedule with 3% linear warmup |
| **Sequence Packing** | `True` (`ConstantLengthDataset`) | Packs sequences to exact 2048 length, eliminating padding token waste |
| **Max Sequence Length** | `2048` | Optimal context window for single-turn technical reasoning |

---

## 2.3 Layer 3: Sentinel Verification Layer (`saber/sentinel.py`)

The Sentinel acts as an automated factual and numerical auditor that verifies specialist outputs before they reach the user or the Meta-Reasoner.

```
               [ Specialist Output Draft ]
                            │
                            ▼
              ┌───────────────────────────┐
              │ 1. Numeric Extractor      │
              │    (Regex & Range Check)  │
              └─────────────┬─────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │ 2. Offline SQLite KB      │
              │    (Passage Retrieval)    │
              └─────────────┬─────────────┘
                            │
                            ▼
              ┌───────────────────────────┐
              │ 3. Factuality Guard       │
              │    - Contradiction Check  │
              │    - Hallucination Flag   │
              └───────────────────────────┘
```

1. **Numeric Consistency Verification**:
   - Extracts all numbers, monetary amounts, percentages, and CVE identifiers using regular expressions: `\b\d+(?:\.\d+)?%?\b|CVE-\d{4}-\d+`.
   - Computes expected values against query context. If numeric discrepancies exist between input data and model claims, Sentinel flags a `NUMERIC_MISMATCH`.

2. **Offline Knowledge Base Fact-Grounding**:
   - Queries a localized SQLite database (`data/offline_kb/{domain}_kb.db`) containing indexed, authoritative domain passages.
   - Evaluates passage cosine similarity using TF-IDF token overlap to confirm factual claims.

---

## 2.4 Layer 4: Meta-Reasoner & Verifiable-Fact GRPO (`saber/meta_reasoner.py`, `scripts/4_train_grpo.py`)

When a complex query activates multiple domain specialists (e.g., a query touching on both **Cyber** and **Coding**), the Meta-Reasoner synthesizes their individual responses using Group Relative Policy Optimization (GRPO).

### GRPO Objective & Multi-Reward Function
GRPO samples a group of $G = 4$ outputs $\{o_1, o_2, o_3, o_4\}$ from the old policy $\pi_{\theta_{old}}$ for a given prompt $q$. It optimizes policy $\pi_\theta$ using relative advantage normalization across the group without requiring a separate critic model:

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[ \frac{1}{G} \sum_{i=1}^G \min\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) A_i \right) \right]$$

Where the group advantage $A_i$ is computed from normalized total rewards $R_i$:

$$A_i = \frac{R_i - \text{mean}(\{R_1..R_G\})}{\text{std}(\{R_1..R_G\}) + 10^{-8}}$$

### Reward Function Components ($R_i = R_{outcome} + R_{fact} + R_{format}$)
1. **Outcome Reward ($R_{outcome}$)**:
   - $+2.0$ for exact ground-truth option match on benchmark problems.
   - $-2.0$ for incorrect option selection.
2. **Sentinel Factuality Reward ($R_{fact}$)**:
   - $+0.5$ if generated reasoning is fully supported by Sentinel SQLite KB lookup.
   - $-1.5$ if Sentinel flags a hallucination or numeric contradiction.
3. **Format & CoT Structure Reward ($R_{format}$)**:
   - $+0.3$ if response contains valid Markdown headers (`## Step 1 [IDENTIFY]`, `## Step 2 [ANALYZE]`, etc.).
   - $-0.5$ for missing structural markers or raw unstructured text dumps.

---

# 3. Data Extraction Pipeline & Dataset Specs (`saber/training/dataset_loader.py`)

The dataset pipeline automatically fetches, cleans, filters, and formats raw open-source datasets from HuggingFace into normalized JSONL format (`data/processed/{domain}.jsonl`).

```
  ┌───────────────────┐
  │ HuggingFace Hub   │
  └─────────┬─────────┘
            │  (load_dataset)
            ▼
  ┌───────────────────┐
  │ Quality Filter    │ ──> Drop length < 30 chars, empty fields, & AI refusal phrases
  └─────────┬─────────┘
            │
            ▼
  ┌───────────────────┐
  │ CoT Converter     │ ──> Convert 30% of eligible long records into structured CoT steps
  └─────────┬─────────┘
            │
            ▼
  ┌───────────────────┐
  │ JSONL Exporter    │ ──> Output to data/processed/{domain}.jsonl
  └───────────────────┘
```

## 3.1 Quality Filtering Rules (`_quality_filter`)
Every raw dataset record must pass three strict quality gates before being accepted into training:
1. **Minimum Label Length**: `len(label) >= 30` characters. Single-word answers, raw letters, or incomplete snippets are discarded.
2. **Non-Empty Text & Label**: Records missing either prompt or completion are rejected.
3. **AI Refusal & Hedging Purge**: Any completion containing hedging or refusal strings is dropped:
   - `"unknown"`, `"i don't know"`, `"cannot be determined"`, `"not enough information"`, `"i cannot answer"`, `"i am an ai"`, `"as an ai language model"`, `"unclear from the context"`.

## 3.2 Chain-of-Thought (CoT) Conversion Engine (`_convert_to_cot`)
To instill structured step-by-step reasoning in specialist models, **30%** of all eligible records (where `len(label) >= 200` characters with structural breaks) are deterministically converted into multi-step Markdown format using `random.seed(42)`:

```markdown
## Step 1 [IDENTIFY]
The query asks about: <Query Summary>

## Step 2 [ANALYZE]
<Section 1 Content>

## Step 3 [EVIDENCE]
<Section 2 Content>

## Step 4 [CONCLUDE]
<Final Conclusion / Answer>
```

---

## 3.3 Complete Domain Data Sources & Field Mappings

### 1. Cyber Specialist Dataset (`data/processed/cyber.jsonl`)
- **Target Size**: ~11,558 raw records (~10,980 train / 578 eval).
- **Sources & Field Extraction**:
  1. **MITRE ATT&CK STIX 2.1 Enterprise Data**:
     - *Source*: MITRE Official Enterprise STIX JSON (`raw.githubusercontent.com/mitre-attack/attack-stix-data`)
     - *Extracted Fields*: `name` $\rightarrow$ `text` (prompt), `description` $\rightarrow$ `label` (completion) for `attack-pattern`, `malware`, and `intrusion-set` STIX objects.
  2. **InfoSec Security QA**:
     - *HuggingFace Repository*: `pAILabs/infosec-security-qa`
     - *Extracted Fields*: `question` $\rightarrow$ `text`, `answer` $\rightarrow$ `label`
  3. **Trendyol Cybersecurity Instruction Dataset**:
     - *HuggingFace Repository*: `Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset`
     - *Extracted Fields*: `user` / `instruction` $\rightarrow$ `text`, `assistant` / `output` $\rightarrow$ `label`
  4. **CyberMetric Expert Benchmark**:
     - *HuggingFace Repository*: `AcerSeb/CyberMetric`
     - *Extracted Fields*: `question` $\rightarrow$ `text`, `answer` + `explanation` $\rightarrow$ `label`
  5. **Synthetic Incident Response Scenarios**:
     - *Source*: Hand-crafted STIX/MITRE incident scenarios covering T1566 (Spearphishing), T1059 (PowerShell), T1053 (Scheduled Tasks), and T1078 (Valid Accounts).

---

### 2. Finance Specialist Dataset (`data/processed/finance.jsonl`)
- **Target Size**: ~13,646 raw records (~12,963 train / 683 eval).
- **Sources & Field Extraction**:
  1. **FinQA (Financial Numerical Reasoning)**:
     - *HuggingFace Repository*: `financial_phrasebank`, `ebgraphs/finqa`
     - *Fields Extracted*: `pre_text` + `table` + `post_text` $\rightarrow$ `text`, `exe_ans` + `program` $\rightarrow$ `label`
     - *Transformation*: Formats financial context tables and mandates numerical step breakdown.
  2. **TAT-QA (Table-And-Text Financial QA)**:
     - *HuggingFace Repository*: `nextplusplus/tat-qa`
     - *Fields Extracted*: `question` $\rightarrow$ `text`, `answer` $\rightarrow$ `label`
  3. **SEC 10-K Filings & Earnings Call Subsets**:
     - *HuggingFace Repository*: `gbd/sec-earnings-call-qa`
     - *Fields Extracted*: `context` + `question` $\rightarrow$ `text`, `answer` $\rightarrow$ `label`

---

### 3. Coding Specialist Dataset (`data/processed/coding.jsonl`)
- **Target Size**: ~20,905 raw records (~19,859 train / 1,046 eval).
- **Composition**: 60% Algorithmic Problem Solving / 40% Syntax & Implementation.
- **Sources & Field Extraction**:
  1. **Python Code Instructions (Alpaca 18k)**:
     - *HuggingFace Repository*: `iamtarun/python_code_instructions_18k_alpaca`
     - *Extracted Fields*: `instruction` + `input` $\rightarrow$ `text`, `output` $\rightarrow$ `label`
  2. **Python Codes 25k**:
     - *HuggingFace Repository*: `flytech/python-codes-25k`
     - *Extracted Fields*: `instruction` / `text` $\rightarrow$ `text`, `output` $\rightarrow$ `label`
  3. **DeepMind CodeContests**:
     - *HuggingFace Repository*: `deepmind/code_contests`
     - *Extracted Fields*: `description` $\rightarrow$ `text`, `solutions` (Python 3 filter) $\rightarrow$ `label`
  4. **MBPP (Mostly Basic Python Problems)**:
     - *HuggingFace Repository*: `google-research-datasets/mbpp`
     - *Extracted Fields*: `text` (prompt) $\rightarrow$ `text`, `code` $\rightarrow$ `label`
  5. **LeetCode Python Solutions**:
     - *HuggingFace Repository*: `greencode/leetcode-python`
     - *Extracted Fields*: `instruction` $\rightarrow$ `text`, `output` $\rightarrow$ `label`

---

### 4. Architecture Specialist Dataset (`data/processed/architecture.jsonl`)
- **Target Size**: ~9,543 raw records (~9,065 train / 478 eval).
- **Sources & Field Extraction**:
  1. **CodeFeedback Filtered Instruction (Architecture Split)**:
     - *HuggingFace Repository*: `m-a-p/CodeFeedback-Filtered-Instruction`
     - *Extracted Fields*: `query` $\rightarrow$ `text`, `answer` $\rightarrow$ `label` (filtered for system design keywords like `scalable`, `microservice`, `kubernetes`, `docker`, `load balancer` with $\ge 800$ char structured explanations).
  2. **System Design & Distributed Systems Synthetic Generator**:
     - *Source*: Programmatically synthesized domain design patterns covering microservices, caching, database sharding, CAP theorem, and event-driven queues.

---

### 5. Orchestrator Dataset (`data/processed/orchestrator.jsonl`)
- **Target Size**: ~10,000 synthetic routing records.
- **Goal**: Fine-tunes the router to emit valid JSON domain arrays (e.g. `["cyber", "coding"]`).
- **Fields**:
  - `text`: User query.
  - `label`: `["<domain_1>", "<domain_2>"]`

---

### 6. Meta-Reasoner Dataset (`data/processed/meta_reasoner.jsonl`)
- **Target Size**: ~5,000 multi-domain synthesis records.
- **Goal**: Fine-tunes the Meta-Reasoner to resolve conflicting specialist claims.
- **Fields**:
  - `text`: Concatenated specialist outputs `[Specialist: Cyber] ... [Specialist: Coding] ...`
  - `label`: Unified consensus synthesis with conflict resolution.

---

# 4. Benchmarking & Ablation Methodology (`scripts/run_final_benchmark.py`)

To prove the superiority of the SABER architecture, the benchmark suite evaluates performance across **5 distinct operational modes**:

```
                         [ 5-MODE ABLATION SUITE ]
                                     │
    ┌────────────────┬───────────────┼───────────────┬────────────────┐
    ▼                ▼               ▼               ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│    MODE 1    │ │    MODE 2    │ │    MODE 3    │ │    MODE 4    │ │    MODE 5    │
│ Base Model   │ │ Single SFT   │ │ MoS Router   │ │ Router +     │ │ Complete     │
│ (Un-tuned)   │ │ (No Router)  │ │ (No Sentinel)│ │ Sentinel     │ │ SABER Stack  │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

## 4.1 Benchmark Operational Modes

1. **Mode 1: Base Model (Zero-Shot)**
   - Plain `Qwen2.5-7B-Instruct` base model without adapters or routing.
   - Evaluates base pre-training knowledge.

2. **Mode 2: Single Specialist SFT (No Router)**
   - Base model with a single active DoRA adapter enabled globally across all queries.
   - Demonstrates the failure of single-specialist over-fitting on out-of-domain queries.

3. **Mode 3: MoS Router + Specialists (No Sentinel)**
   - Orchestrator active, routing queries to activated DoRA specialists. Sentinel verification disabled.
   - Measures pure multi-specialist domain capability.

4. **Mode 4: MoS Router + Specialists + Sentinel Verification**
   - Orchestrator and Specialists active with Sentinel fact-grounding and numeric verification enabled.
   - Measures reduction in hallucinations and numeric errors.

5. **Mode 5: Full SABER Stack (MoS + Sentinel + GRPO Meta-Reasoner)**
   - Complete production stack: Orchestrator, Specialists, Sentinel KB, and GRPO Meta-Reasoning consensus synthesis.
   - Measures end-to-end multi-agent performance.

---

## 4.2 Benchmark Evaluation Suite & Scoring Metrics

The benchmark suite tests models against strict Multiple Choice Questions (MCQs) and numerical exact-match benchmarks:

1. **Cybersecurity Benchmarks**:
   - **SecBench / CyberMetric-800**: Tests vulnerability identification, MITRE ATT&CK mapping, and exploit analysis.
2. **Finance Benchmarks**:
   - **FinQA Math Benchmark**: 80 exact numerical calculation problems testing gross profit, EBITDA, and ratio calculations.
3. **Coding Benchmarks**:
   - **HumanEval / MBPP**: Python functional correctness and algorithmic complexity.
4. **Architecture Benchmarks**:
   - **System Design Trade-off Suite**: Evaluates distributed systems choices and CAP theorem trade-offs.

### Answer Parsing Engine (`parse_mcq_answer` / `parse_exact_answer`)
To eliminate evaluation bias, raw model responses are evaluated using a 5-Pass Parser:
- **Pass 1**: Strict last line regex `ANSWER: <LETTER>`.
- **Pass 2**: Multi-line regex scan for `ANSWER: <LETTER>`.
- **Pass 3**: Natural language phrase match (`"the correct answer is B"`).
- **Pass 4**: Single trailing letter isolation.
- **Pass 5**: Option value string matching against prompt options.

---


