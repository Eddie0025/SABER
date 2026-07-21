# -*- coding: utf-8 -*-
"""scripts/2_build_kb.py

Step 2: Extract reference support passages from processed datasets and compile indexed SQLite KBs
for 0ms offline Sentinel RL verification.
Output: data/offline_kb/{domain}_kb.db
"""

import os
import sys

sys.path.append(os.path.abspath('.'))
from scripts.build_offline_kb import main as build_kbs
from scripts.validate_kb_coverage import main as validate_kbs

def main():
    print("=========================================================================")
    print("           STEP 2: BUILDING & AUDITING OFFLINE KNOWLEDGE BASES            ")
    print("=========================================================================")
    build_kbs()
    print("\n--- PRE-FLIGHT KB COVERAGE AUDIT REPORT ---")
    validate_kbs()
    print("[Step 2 Complete] All offline SQLite KBs compiled and verified in data/offline_kb/\n")

if __name__ == "__main__":
    main()
