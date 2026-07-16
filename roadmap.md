# SABER v2 Roadmap

## Phase 1 — Research Infrastructure Foundation (Build Now)

**Goal:** Create the measurement and verification infrastructure before specialist training finishes.

Without this phase, SABER is an architecture.
With this phase, SABER becomes a research system.

---

# 1. Structured Flag Taxonomy

### Purpose

Convert verification from free-form criticism into measurable data.

### Required Fields

```text
flag_id
flag_type
severity
confidence
evidence
reasoning
suggested_fix
specialist_source
timestamp
```

### Required Categories

```text
FACTUAL_ERROR
REASONING_ERROR
LOGIC_GAP
MISSING_EVIDENCE
DOMAIN_CONFLICT
CALCULATION_ERROR
SECURITY_ASSUMPTION_ERROR
DIAGNOSTIC_INCONSISTENCY
FINANCIAL_ANALYSIS_ERROR
```

### Metrics To Capture

```text
Flags Raised
Flags Accepted
Flags Rejected
Flags By Category
Flags By Specialist
```

### Research Value

Allows SABER to quantify what types of mistakes occur.

---

# 2. Multi-Pass Verification Engine

### Purpose

Enable iterative correction.

### Workflow

```text
Question
↓
Initial Answer
↓
Verification Pass
↓
Flags
↓
Rewrite
↓
Verification Pass
↓
Final Answer
```

### Store

```text
verification_passes
flags_resolved
flags_remaining
revision_count
```

### Research Value

Allows measurement of error detection and correction rates.

---

# 3. Intelligent Rewrite Engine

### Purpose

Replace simple correction appending.

### Current

```text
Answer
+
Correction Note
```

### New

```text
Answer
↓
Flag Analysis
↓
LLM Rewrite
↓
Improved Answer
```

### Metrics

```text
rewrite_success_rate
accuracy_before_rewrite
accuracy_after_rewrite
```

### Research Value

Measures whether verification actually improves answers.

---

# 4. Dedicated Benchmark Dataset

### Purpose

Create a permanent SABER evaluation suite.

### Dataset Structure

#### Cyber

```text
50 Questions
```

Categories:

```text
Incident Response
Threat Hunting
MITRE ATT&CK
Risk Assessment
Cloud Security
Detection Engineering
```

---

#### Science

```text
50 Questions
```

Categories:

```text
Physics
Chemistry
Biology
Mathematics
Statistics
```

---

#### Cross-Domain

```text
50 Questions
```

Examples:

```text
Hospital Ransomware

Medical + Cyber

Autonomous Vehicle Failure

Engineering + Safety + AI
```

### Required Fields

```json
{
  "question_id": "",
  "question": "",
  "domain": "",
  "difficulty": "",
  "ground_truth": "",
  "reasoning_points": []
}
```

### Rule

Training data must never appear in benchmark data.

---

# 5. Benchmark Execution Framework

### Purpose

Generate publishable comparisons.

### Modes

#### MODE_BASE

```text
Raw Base Model
```

#### MODE_SELF_CRITIQUE

```text
Base Model
+
Self Reflection
```

#### MODE_SPECIALIST

```text
Specialist Only
```

#### MODE_SPECIALIST_VERIFY

```text
Specialist
+
Verification
```

#### MODE_SABER

```text
Full System
```

### Output

```csv
Question ID
Mode
Accuracy
Latency
Flags
Corrections
Confidence
```

### Research Value

This is the most important experiment generator.

---

# 6. Failure Classification System

### Purpose

Understand why SABER fails.

### Categories

```text
ROUTING_FAILURE
SPECIALIST_FAILURE
VERIFICATION_FAILURE
CONSENSUS_FAILURE
KNOWLEDGE_FAILURE
SYSTEM_FAILURE
```

### Metrics

```text
Failure Distribution
Failure Frequency
Failure Severity
```

### Research Value

Enables root-cause analysis.

---

# 7. Decision Ledger v2

### Purpose

Create a complete reasoning audit trail.

### Required Fields

```text
Query
Selected Specialists
Initial Responses
Flags
Corrections
Verification History
Disagreements
Final Resolution
Final Confidence
```

### Research Value

Forms the basis of SABER's explainability claim.

---

# Phase 2 — Verification Intelligence Layer

**Start after Phase 1 is complete.**

---

# 8. Specialist Disagreement Engine

### Purpose

Measure expert disagreement.

### Store

```text
Specialist Confidence
Specialist Position
Disagreement Score
```

### Trigger

```text
High Disagreement
↓
Additional Verification
```

### Research Value

Focuses resources where uncertainty exists.

---

# 9. Verification Effectiveness Metrics

### Purpose

Measure whether verification helps.

### Metrics

```text
Errors Detected
Errors Corrected
False Flags
Missed Errors
Correction Success Rate
```

### Research Value

Directly tests SABER's main hypothesis.

---

# 10. Targeted Verification Routing

### Purpose

Avoid irrelevant reviews.

### Example

Instead of:

```text
Medical Reviews Cyber
```

Use:

```text
Cyber Reviews Technical Content

Science Reviews Logic

Meta-Reasoning Layer Resolves Conflict
```

### Research Value

Improves efficiency and realism.

---

# 11. Selective Activation Tracking

### Purpose

Support efficiency claims.

### Metrics

```text
Active Specialists
Active Parameters
Tokens Consumed
Inference Cost
Memory Usage
Latency
```

### Research Value

Supports the "flagship performance with fewer active parameters" hypothesis.

---

# Phase 3 — Publication Layer

**Only after all specialists exist and benchmarks are running.**

---

# 12. Human Audit Study

### Purpose

Validate the Decision Ledger.

### Test

Group A:

```text
Raw AI Answer
```

Group B:

```text
SABER Ledger
```

Measure:

```text
Trust
Understanding
Audit Speed
Error Detection
```

---

# 13. Large Scale Benchmark Campaign

Run:

```text
Cyber-v1
Science-v1
Medical-v2
Finance-v1
```

across:

```text
Base
Self Critique
Specialist
Verification
Full SABER
```

### Target Output

Tables and graphs for publication.

---

# 14. Research Paper Metrics

Final metrics should include:

```text
Accuracy

Error Detection Rate

Error Correction Rate

Reliability

Latency

Token Cost

Active Parameters

Failure Distribution

Disagreement Score

Human Audit Score
```

---

# Current Recommendation

Right now I'd tell the agents:

```text
Finish EVERYTHING in Phase 1.

Begin coding Phase 2 immediately afterward.

Do NOT start Phase 3 until:
- Cyber-v1 benchmarked
- Science-v1 finished
- Medical-v2 finished

At that point run the first full SABER benchmark campaign.
```
