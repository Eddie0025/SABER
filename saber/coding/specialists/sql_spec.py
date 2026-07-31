import logging
from typing import Optional
from saber.specialists.base_spec import BaseSpecialist
from saber.coding.shared_memory import SharedMemory

logger = logging.getLogger("SABER_SQLSpecialist")

class SQLSpecialist(BaseSpecialist):
    """
    DoRA specialist for SQL and database design generation.
    """
    def __init__(self, base_model_id: str, adapter_path: str, memory: Optional[SharedMemory] = None):
        super().__init__(base_model_id, adapter_path)
        self.memory = memory
        self.system_prompt = (
            "You are an expert Database Administrator and SQL developer. "
            "Write highly optimized, secure, and robust SQL queries and schema definitions."
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
        
        logger.info("Executing SQL DoRA Inference...")
        return self.generate(full_prompt)
