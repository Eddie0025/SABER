import logging
from typing import Optional
from saber.specialists.base_spec import BaseSpecialist
from saber.coding.shared_memory import SharedMemory

logger = logging.getLogger("SABER_JavascriptSpecialist")

class JavascriptSpecialist(BaseSpecialist):
    """
    DoRA specialist for JavaScript/TypeScript generation.
    Connects to SharedMemory to contextually integrate with architecture or Python backend logic.
    """
    def __init__(self, base_model_id: str, adapter_path: str, memory: Optional[SharedMemory] = None):
        super().__init__(base_model_id, adapter_path)
        self.memory = memory
        self.system_prompt = (
            "You are an expert JavaScript/TypeScript developer. Write clean, modern, "
            "and efficient JS/TS code. Adhere to ES6+ standards."
        )

    def generate_code(self, prompt: str, context_keys: list = None) -> str:
        memory_context = ""
        if self.memory and context_keys:
            for key in context_keys:
                ctx = self.memory.get(key)
                if ctx:
                    memory_context += f"\\nContext [{key}]: {ctx}\\n"
                    
        full_prompt = f"<|im_start|>system\\n{self.system_prompt}<|im_end|>\\n"
        full_prompt += f"<|im_start|>user\\n{memory_context}{prompt}<|im_end|>\\n<|im_start|>assistant\\n"
        
        logger.info("Executing JavaScript DoRA Inference...")
        return self.generate(full_prompt)
