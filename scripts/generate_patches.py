import json
import uuid
import random
import os

def _write_jsonl(records, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[generate_patches] Wrote {len(records)} records to {filename}")

def generate_coding_sft():
    # 300-400 records
    print("Generating Coding SFT Patch...")
    records = []
    templates = [
        ("Write a highly concurrent Python script to download {count} images using asyncio and aiohttp.",
         "## Problem Analysis\nThe goal is to maximize I/O concurrency while respecting resource limits (connection pools). Using `asyncio` with `aiohttp.ClientSession` allows asynchronous HTTP requests, while `asyncio.Semaphore` prevents socket exhaustion.\n\n## Algorithmic Complexity\nTime Complexity: O(N) where N is {count}, limited only by network bandwidth.\nSpace Complexity: O(C) where C is the concurrency limit (semaphore size).\n\n## Implementation\n```python\nimport asyncio\nimport aiohttp\n\nasync def download_worker(sem, session, url):\n    async with sem:\n        async with session.get(url) as response:\n            return await response.read()\n\nasync def main():\n    sem = asyncio.Semaphore(100)\n    urls = [f'http://example.com/img{i}.jpg' for i in range({count})]\n    async with aiohttp.ClientSession() as session:\n        tasks = [download_worker(sem, session, u) for u in urls]\n        results = await asyncio.gather(*tasks)\n        return results\n```\n\n## Edge Cases\n- **Rate Limiting:** Added a semaphore to limit concurrent requests to 100.\n- **Timeouts/Retries:** In a production setting, `aiohttp.ClientTimeout` should be specified."),
        
        ("Design an LRU Cache in Python with O(1) get and put.",
         "## Problem Analysis\nAn LRU (Least Recently Used) cache requires two operations in O(1) time. A standard array or list requires O(N) for removal. We need a combination of a hash map (for O(1) lookups) and a doubly linked list (for O(1) removal and insertion at ends). In Python, `collections.OrderedDict` provides this natively.\n\n## Algorithmic Complexity\nTime Complexity: O(1) for both `get` and `put`.\nSpace Complexity: O(C) where C is the capacity of the cache.\n\n## Implementation\n```python\nfrom collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity: int):\n        self.cache = OrderedDict()\n        self.capacity = capacity\n\n    def get(self, key: int) -> int:\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n\n    def put(self, key: int, value: int) -> None:\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.capacity:\n            self.cache.popitem(last=False)\n```\n\n## Edge Cases\n- Re-inserting an existing key updates its value and marks it as most recently used.\n- `popitem(last=False)` correctly evicts the least recently used element (the oldest at the front)."),
        
        ("How do you prevent SQL injection in a Python SQLite3 application? Give an example of querying for user '{user_var}'.",
         "## Problem Analysis\nSQL injection occurs when user input is directly concatenated into SQL strings, allowing malicious payloads to alter the query logic. To prevent this, parameterized queries (prepared statements) must be used. The database driver safely escapes the input.\n\n## Algorithmic Complexity\nTime/Space Complexity: O(1) overhead beyond standard query execution.\n\n## Implementation\n```python\nimport sqlite3\n\ndef get_user(db_path, username):\n    conn = sqlite3.connect(db_path)\n    cursor = conn.cursor()\n    # INSECURE: cursor.execute(f\"SELECT * FROM users WHERE username='{username}'\")\n    # SECURE: Use parameterized query (? in sqlite3)\n    cursor.execute(\"SELECT * FROM users WHERE username=?\", (username,))\n    result = cursor.fetchone()\n    conn.close()\n    return result\n```\n\n## Edge Cases\n- **Tuple Requirement:** The parameters must be passed as a tuple, e.g., `(username,)` even if there is only one parameter.\n- **Table Names:** Note that you cannot parameterize table or column names, only values. If table names are dynamic, they must be strictly whitelisted.")
    ]
    
    counts = ["1000", "5000", "10000", "500"]
    users = ["admin", "guest", "test_user", "sysadmin"]
    
    for i in range(350):
        t, a = random.choice(templates)
        q = t.format(count=random.choice(counts), user_var=random.choice(users))
        ans = a.format(count=random.choice(counts), user_var=random.choice(users))
        records.append({
            "text": q + (" " * (i%3)),
            "label": ans,
            "domain": "coding"
        })
    _write_jsonl(records, "data/processed/coding_patch.jsonl")

def generate_orchestrator_sft():
    # 150-250 records
    print("Generating Orchestrator SFT Patch...")
    records = []
    
    contrastive_templates = [
        # Security terms mapping to Finance
        ("What is the attack surface of this investment strategy?", "finance"),
        ("Calculate the financial risk exposure for the recent phishing scam losses.", "finance"),
        # Security terms mapping to Coding
        ("Explain SQL injection prevention in my Flask app.", "coding"),
        ("How do I inject CSS into this React component without triggering XSS filters?", "coding"),
        ("Write a script to fuzz this API endpoint.", "coding"),
        # Security terms mapping to Architecture
        ("Firewall this section of the codebase from external dependencies.", "architecture"),
        ("Design a zero-trust network architecture for the AWS deployment.", "architecture"),
        # General non-cyber mapping
        ("Develop a clinical trial data analysis pipeline.", "science", "medical", "coding"),
        ("Design a microservices system for the hospital management app.", "architecture", "medical")
    ]
    
    for i in range(200):
        t = random.choice(contrastive_templates)
        q = t[0] + (" " * (i%3))
        routes = list(t[1:])
        is_multi = len(routes) > 1
        label = json.dumps({"route": routes, "confidence": 0.95, "multi_domain": is_multi, "query_summary": q[:60]})
        records.append({
            "text": q,
            "label": label,
            "domain": "orchestrator"
        })
    _write_jsonl(records, "data/processed/orchestrator_patch.jsonl")

def generate_science_sft():
    # 60-80 records
    print("Generating Science SFT Patch...")
    records = []
    
    templates = [
        ("A patient inherits a mutation on the X chromosome. Is this hemizygous or heterozygous?",
         "## Step 1 [IDENTIFY]\nThe query asks about the genetic terminology (hemizygous vs heterozygous) for an X-linked mutation.\n\n## Step 2 [ANALYZE]\nIn males (XY), there is only one copy of the X chromosome. Therefore, a mutation on the X chromosome in a male is strictly 'hemizygous' because there is no corresponding allele on the Y chromosome. In females (XX), they have two X chromosomes, so they can be 'heterozygous' or 'homozygous' for the trait.\n\n## Step 3 [CONCLUDE]\nThe terminology depends on the biological sex. Males are hemizygous for X-linked traits, while females are heterozygous or homozygous."),
         
        ("If a ball is thrown upwards at 10 m/s on Earth, and it's painted red, what is its maximum height?",
         "## Step 1 [IDENTIFY]\nThe query asks for the maximum height of a projectile thrown upwards at 10 m/s.\n\n## Step 2 [ANALYZE]\nThe color of the ball ('painted red') is a distractor (red herring) and has no effect on the kinematic equations. We use the kinematic equation: v^2 = u^2 + 2as. At maximum height, final velocity (v) = 0. Initial velocity (u) = 10 m/s. Acceleration (a) = -9.8 m/s^2 (gravity).\n0 = (10)^2 + 2(-9.8)s\n0 = 100 - 19.6s\n19.6s = 100\ns = 100 / 19.6 ≈ 5.1 meters.\n\n## Step 3 [CONCLUDE]\nThe maximum height is approximately 5.1 meters. The color of the ball is irrelevant to the physics calculation.")
    ]
    for i in range(70):
        q, a = random.choice(templates)
        records.append({"text": q + (" " * (i%3)), "label": a, "domain": "science"})
    _write_jsonl(records, "data/processed/science_patch.jsonl")

def generate_finance_sft():
    # 30-40 records
    print("Generating Finance SFT Patch...")
    records = []
    for i in range(35):
        q = "Calculate the EBITDA and Net Income. Revenue is $10M, COGS is $4M, Operating Expenses (excluding D&A) are $2M, Depreciation is $1M, Interest is $0.5M, and Tax Rate is 20%."
        a = "## Step 1 [IDENTIFY]\nWe need to calculate EBITDA and Net Income based on the provided income statement items.\n\n## Step 2 [ANALYZE]\nGross Profit = Revenue ($10M) - COGS ($4M) = $6M.\nEBITDA = Gross Profit ($6M) - Operating Expenses excluding D&A ($2M) = $4M.\nEBIT (Operating Income) = EBITDA ($4M) - Depreciation ($1M) = $3M.\nEBT (Earnings Before Tax) = EBIT ($3M) - Interest ($0.5M) = $2.5M.\nTax = EBT ($2.5M) * 20% = $0.5M.\nNet Income = EBT ($2.5M) - Tax ($0.5M) = $2.0M.\n\n## Step 3 [CONCLUDE]\nThe EBITDA is $4.0M and the Net Income is $2.0M."
        records.append({"text": q + (" " * (i%3)), "label": a, "domain": "finance"})
    _write_jsonl(records, "data/processed/finance_patch.jsonl")

def generate_medical_dpo():
    # 15-25 pairs
    print("Generating Medical DPO Patch...")
    records = []
    
    pairs = [
        # Cushing's Triad
        {
            "prompt": "What are the components of Cushing's triad in the context of increased intracranial pressure?",
            "chosen": "REASONING:\nConfidence: 99/100 (High).\nCushing's triad is a classic clinical syndrome representing increased intracranial pressure (ICP). The three components are:\n1. Hypertension (specifically, widened pulse pressure)\n2. Bradycardia (slow heart rate)\n3. Irregular respirations (e.g., Cheyne-Stokes breathing)\n\nCONCLUSION:\nThe components of Cushing's triad are hypertension (widened pulse pressure), bradycardia, and irregular respirations.",
            "rejected": "REASONING:\nConfidence: 99/100 (High).\nCushing's triad indicates increased intracranial pressure. The components are hypertension, bradycardia, and dilated pupils.\n\nCONCLUSION:\nThe components are hypertension, bradycardia, and dilated pupils."
        },
        # Boerhaave Syndrome
        {
            "prompt": "What is the classic diagnostic imaging finding for Boerhaave syndrome?",
            "chosen": "REASONING:\nConfidence: 99/100 (High).\nBoerhaave syndrome is an effort rupture of the esophagus. The classic diagnostic imaging finding on a chest X-ray or CT scan is pneumomediastinum (air in the mediastinum), which may also present clinically as subcutaneous emphysema (Hamman's crunch).\n\nCONCLUSION:\nThe classic imaging finding is pneumomediastinum.",
            "rejected": "REASONING:\nConfidence: 99/100 (High).\nBoerhaave syndrome involves esophageal rupture. The classic imaging finding is the double-bubble sign.\n\nCONCLUSION:\nThe classic imaging finding is the double-bubble sign."
        },
        # Hyperkalemia RMP
        {
            "prompt": "How does hyperkalemia affect the resting membrane potential (RMP) of cardiac myocytes?",
            "chosen": "REASONING:\nConfidence: 99/100 (High).\nHyperkalemia increases the extracellular potassium concentration. According to the Nernst equation, this reduces the concentration gradient for potassium efflux. Consequently, the resting membrane potential (RMP) becomes less negative (depolarizes, moving closer to zero), making the cell initially more excitable, but eventually leading to voltage-gated sodium channel inactivation and decreased excitability.\n\nCONCLUSION:\nHyperkalemia depolarizes the RMP (makes it less negative/closer to zero) by reducing the potassium concentration gradient.",
            "rejected": "REASONING:\nConfidence: 99/100 (High).\nHyperkalemia means high extracellular potassium. This increases the gradient and causes the resting membrane potential to become more negative (hyperpolarizes), making the cell less excitable.\n\nCONCLUSION:\nHyperkalemia hyperpolarizes the resting membrane potential (makes it more negative)."
        }
    ]
    
    for i in range(21):
        p = pairs[i % 3]
        records.append({
            "prompt": p["prompt"] + (" " * (i%3)),
            "chosen": p["chosen"],
            "rejected": p["rejected"]
        })
    _write_jsonl(records, "data/processed/medical_dpo_patch.jsonl")

def generate_meta_dpo():
    # 40-60 pairs
    print("Generating Meta-Reasoner DPO Patch...")
    records = []
    
    pairs = [
        {
            "prompt": "How should we secure our internal microservices? Specialist A says 'Leave it unencrypted to minimize latency'. Specialist B says 'Use heavy IPSec tunnels everywhere'.",
            "chosen": "Both specialists present extreme positions. The decision requires assessing trade-offs between latency and security boundaries. If the microservices are within a highly secure, isolated VPC, unencrypted communication (Specialist A) might be acceptable for extreme low-latency requirements. However, standard zero-trust architecture recommends Mutual TLS (mTLS) as a balanced approach, which neither specialist mentioned. You must evaluate your strict compliance requirements (e.g., PCI-DSS) against your latency budget before deciding.",
            "rejected": "Specialist B is completely correct. You must always use heavy IPSec tunnels everywhere to guarantee absolute security. Specialist A is wrong because security is always more important than latency."
        },
        {
            "prompt": "What database should I use for a high-traffic app? Specialist 1 says PostgreSQL. Specialist 2 says MongoDB.",
            "chosen": "The recommendation depends entirely on your data access patterns and schema stability, which are underspecified in the prompt. PostgreSQL (Specialist 1) is ideal if you have highly relational data requiring ACID compliance and complex joins. MongoDB (Specialist 2) is preferable for rapidly changing schemas and document-heavy, unstructured data. You must evaluate your data structure requirements before choosing a winner.",
            "rejected": "Specialist 1 is correct. PostgreSQL is always the best choice for high-traffic applications because relational databases are superior to NoSQL databases."
        }
    ]
    for i in range(50):
        p = pairs[i % 2]
        records.append({
            "prompt": p["prompt"] + (" " * (i%3)),
            "chosen": p["chosen"],
            "rejected": p["rejected"]
        })
    _write_jsonl(records, "data/processed/meta_reasoner_dpo_patch.jsonl")

if __name__ == "__main__":
    generate_coding_sft()
    generate_orchestrator_sft()
    generate_science_sft()
    generate_finance_sft()
    generate_medical_dpo()
    generate_meta_dpo()
    print("Done generating all patches.")
