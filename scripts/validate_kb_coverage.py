# -*- coding: utf-8 -*-
"""scripts/validate_kb_coverage.py

Pre-flight verification script that inspects offline SQLite KBs across domains to
report passage counts, average passage lengths, and support passage coverage percentages.
"""

import os
import sqlite3
import sys

KB_DIR = "data/offline_kb"

def validate_domain_kb(domain: str):
    db_path = os.path.join(KB_DIR, f"{domain}_kb.db")
    if not os.path.exists(db_path):
        print(f"| {domain:<15} | NOT FOUND | N/A | N/A | N/A |")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*), AVG(LENGTH(support_passage)), COUNT(CASE WHEN LENGTH(support_passage) > 50 THEN 1 END) FROM passage_kb")
    total_records, avg_len, substantive_passages = cursor.fetchone()

    avg_len = round(avg_len or 0, 1)
    coverage_pct = round((substantive_passages / total_records) * 100.0, 1) if total_records else 0.0

    print(f"| {domain:<15} | {total_records:<10} | {avg_len:<15} | {coverage_pct:<14}% |")
    conn.close()

def main():
    print("=========================================================================")
    print("               SABER Offline KB Pre-Flight Audit Report                   ")
    print("=========================================================================")
    print(f"| {'Domain':<15} | {'Passages':<10} | {'Avg Length (ch)':<15} | {'Coverage %':<14} |")
    print("| :-------------- | :--------- | :-------------- | :------------- |")
    
    domains = ["cyber", "finance", "coding", "architecture", "meta_reasoner", "orchestrator"]
    for d in domains:
        validate_domain_kb(d)
    print("=========================================================================\n")

if __name__ == "__main__":
    main()
