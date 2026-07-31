import logging
from datasets import load_dataset
from typing import Dict, Any

logger = logging.getLogger("SABER_DatasetLoader")

# Maps specialist domains to their HuggingFace datasets or local paths
DATASET_REGISTRY = {
    "cybersecurity": [
        {"path": "pAILabs/infosec-security-qa", "split": "train"},
        {"path": "mitre/attack-stix", "split": "train"}, # Mock path
        {"path": "cvefixes", "split": "train"} # Mock path
    ],
    "finance": [
        {"path": "finqa", "split": "train"},
        {"path": "convfinqa", "split": "train"},
        {"path": "tat-qa", "split": "train"}
    ],
    "python": [
        {"path": "sahil2801/CodeAlpaca-20k", "split": "train"},
        {"path": "ise-uiuc/Magicoder-OSS-Instruct-75K", "split": "train"}
        # Removing APPS to avoid auth/structure issues for this test run
    ],
    "javascript": [
        # The Stack is gated (requires HF Token). Using an ungated code instruct dataset.
        {"path": "TokenBender/code_instructions_122k_alpaca_style", "split": "train"}
    ],
    "sql": [
        # Spider sometimes fails on older datasets library versions, using a robust SQL dataset
        {"path": "b-mc2/sql-create-context", "split": "train"}
    ],
    "architecture_qa": [
        {"path": "synthetic-architecture-qa", "split": "train"}
    ],
    "architecture_planner": [
        {"path": "synthetic-planner-decomp", "split": "train"}
    ],
    "medical": [
        {"path": "medical-o1-reasoning", "split": "train"},
        {"path": "medqa-usmle", "split": "train"},
        {"path": "medmcqa", "split": "train"}
    ],
    "science": [
        {"path": "sciq", "split": "train"},
        {"path": "arc-challenge", "split": "train"}
    ],
    "orchestrator": [
        {"path": "synthetic-routing", "split": "train"}
    ],
    "meta_reasoner": [
        {"path": "synthetic-contradiction", "split": "train"}
    ]
}

def apply_chatml_formatting(example):
    """
    Normalizes dataset into ChatML.
    Assumes standard QA datasets have 'question' and 'answer' fields.
    For this scaffolding, we fallback to standard text if fields are missing.
    """
    q = example.get('question', example.get('instruction', ''))
    a = example.get('answer', example.get('output', ''))
    
    # Format: <|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>
    text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
    return {"text": text}

def load_specialist_dataset(specialist_name: str) -> Any:
    """
    Loads and normalizes the datasets for a given specialist into a standard ChatML schema.
    Applies standard length bounds filtering.
    """
    if specialist_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown specialist '{specialist_name}'. Available: {list(DATASET_REGISTRY.keys())}")
        
    logger.info(f"Loading datasets for {specialist_name}...")
    
    datasets_list = []
    for dset_config in DATASET_REGISTRY[specialist_name]:
        try:
            # We only load a small subset if it's a mock path, else we load the real train split
            logger.info(f"Loading {dset_config['path']}")
            dset = load_dataset(dset_config["path"], split=dset_config["split"], streaming=False)
            
            # Apply basic V1 filtering (length bounds)
            # Assuming 'answer' or 'output' exists
            dset = dset.filter(lambda x: len(x.get('answer', x.get('output', '')).split()) >= 10)
            
            # Apply formatting
            dset = dset.map(apply_chatml_formatting)
            datasets_list.append(dset)
        except Exception as e:
            logger.warning(f"Could not load {dset_config['path']}: {e}. Skipping.")
    
    if not datasets_list:
        logger.error(f"Failed to load any datasets for {specialist_name}.")
        return None
        
    from datasets import concatenate_datasets
    combined_dataset = concatenate_datasets(datasets_list)
    logger.info(f"Successfully loaded and formatted {len(combined_dataset)} examples for {specialist_name}.")
    
    return combined_dataset
