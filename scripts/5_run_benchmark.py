# -*- coding: utf-8 -*-
"""scripts/5_run_benchmark.py

Step 5: Unified 5-Mode Benchmark Evaluator.
Evaluates Base Qwen, Adaptors, Adaptor+CoT, and Sentinel 2-Pass across benchmark datasets.
Usage:
    python3 scripts/5_run_benchmark.py
"""

import os
import sys

sys.path.append(os.path.abspath('.'))
from scripts.run_final_benchmark import run_benchmark

def main():
    print("=========================================================================")
    print("           STEP 5: UNIFIED 5-MODE BENCHMARK EVALUATOR                     ")
    print("=========================================================================")
    run_benchmark()

if __name__ == "__main__":
    main()
