# -*- coding: utf-8 -*-
"""scripts/1_build_datasets.py

Step 1: Download raw domain datasets from HuggingFace and apply strict benchmark prompt/response alignment.
Output: data/processed/{domain}.jsonl
"""

import os
import sys

sys.path.append(os.path.abspath('.'))
from saber.training.dataset_loader import build_all_processed_datasets

def main():
    print("=========================================================================")
    print("           STEP 1: BUILDING & ALIGNING DOMAIN TRAINING DATASETS           ")
    print("=========================================================================")
    build_all_processed_datasets()
    print("\n[Step 1 Complete] All domain datasets formatted and saved to data/processed/\n")

if __name__ == "__main__":
    main()
