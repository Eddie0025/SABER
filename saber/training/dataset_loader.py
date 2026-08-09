import logging
from datasets import load_dataset
from typing import Dict, Any

logger = logging.getLogger("SABER_DatasetLoader")

# Maps specialist domains to their HuggingFace datasets or local paths
DATASET_REGISTRY = {
    "cybersecurity": [{"path": "pAILabs/infosec-security-qa", "split": "train"}],
    "finance": [{"path": "virattt/financial-qa-10K", "split": "train"}],
    "python": [{"path": "sahil2801/CodeAlpaca-20k", "split": "train"}],
    "javascript": [{"path": "TokenBender/code_instructions_122k_alpaca_style", "split": "train"}],
    "sql": [{"path": "b-mc2/sql-create-context", "split": "train"}],
    "architecture_qa": [{"path": "HuggingFaceH4/no_robots", "split": "train"}],
    "architecture_planner": [{"path": "HuggingFaceH4/no_robots", "split": "train"}],
    "medical": [{"path": "openlifescienceai/medmcqa", "split": "train"}],
    "science": [{"path": "allenai/sciq", "split": "train"}],
    "orchestrator": [{"path": "HuggingFaceH4/no_robots", "split": "train"}],
    "meta_reasoner": [{"path": "HuggingFaceH4/no_robots", "split": "train"}]
}

def apply_chatml_formatting(example):
    """
    Normalizes dataset into ChatML.
    Assumes standard QA datasets have 'question'/'instruction' and 'answer'/'output'/'response' fields.
    """
    # no_robots uses 'prompt' and 'messages' natively. Let's pull from standard keys first.
    q = example.get('question') or example.get('instruction') or example.get('prompt') or ''
    
    # Handle HuggingFaceH4/no_robots conversational structure natively
    if not q and 'messages' in example and isinstance(example['messages'], list) and len(example['messages']) > 0:
        q = example['messages'][0].get('content', '')
        
    # For SQL datasets with context
    context = example.get('context', '')
    if context and q:
        q = f"Context:\n{context}\n\nQuestion:\n{q}"
    
    a = example.get('answer') or example.get('output') or example.get('response') or example.get('solution') or ''
    if not a and 'messages' in example and isinstance(example['messages'], list) and len(example['messages']) > 1:
        a = example['messages'][1].get('content', '')
        
    # Format: <|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>
    text = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>"
    return {"text": text, "question": q, "answer": a}

def load_specialist_dataset(specialist_name: str, max_samples: int = 20000) -> Any:
    """
    Loads and normalizes the datasets for a given specialist into a standard ChatML schema.
    Applies standard length bounds filtering and caps dataset size to prevent catastrophic forgetting.
    """
    if specialist_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown specialist '{specialist_name}'. Available: {list(DATASET_REGISTRY.keys())}")
        
    logger.info(f"Loading datasets for {specialist_name}...")
    
    datasets_list = []
    for dset_config in DATASET_REGISTRY[specialist_name]:
        try:
            logger.info(f"Loading {dset_config['path']}")
            dset = load_dataset(dset_config["path"], split=dset_config["split"], streaming=False)
            
            # Apply formatting
            dset = dset.map(apply_chatml_formatting)
            
            # Apply basic filtering (minimum response length)
            dset = dset.filter(lambda x: len(str(x.get('answer', '')).split()) >= 5)
            
            # Cap dataset size to prevent catastrophic forgetting on massive datasets (e.g. MedMCQA 180k -> 20k)
            if len(dset) > max_samples:
                logger.info(f"Subsampling {len(dset)} -> {max_samples} samples for {specialist_name} to prevent catastrophic forgetting.")
                dset = dset.shuffle(seed=42).select(range(max_samples))
                
            if len(dset) > 0:
                datasets_list.append(dset)
        except Exception as e:
            logger.warning(f"Could not load {dset_config['path']}: {e}. Skipping.")
    
    if not datasets_list:
        logger.error(f"Failed to load any datasets for {specialist_name}. Ensure network is up and datasets are correctly formatted.")
        return None
        
    from datasets import concatenate_datasets
    combined_dataset = concatenate_datasets(datasets_list)
    
    if len(combined_dataset) == 0:
        logger.error(f"Combined dataset for {specialist_name} is empty after filtering!")
        return None
    
    # Ensure combined dataset doesn't exceed cap
    if len(combined_dataset) > max_samples:
        combined_dataset = combined_dataset.shuffle(seed=42).select(range(max_samples))
        
    logger.info(f"Successfully loaded and formatted {len(combined_dataset)} examples for {specialist_name}.")
    return combined_dataset
