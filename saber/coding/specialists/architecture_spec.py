import logging
from typing import Optional
from saber.specialists.base_spec import BaseSpecialist
from saber.coding.shared_memory import SharedMemory

logger = logging.getLogger("SABER_ArchitectureSpecialist")

class ArchitectureSpecialist(BaseSpecialist):
    """
    DoRA specialist for high-level system architecture and design patterns.
    """
    def __init__(self, base_model_id: str, adapter_path: str, memory: Optional[SharedMemory] = None):
        super().__init__(base_model_id, adapter_path)
        self.memory = memory
        self.system_prompt = (
            "You are an expert Principal Software Architect. "
            "Design scalable, resilient, and secure system architectures. "
            "Provide clear specifications and diagrams."
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
        
        logger.info("Executing Architecture DoRA Inference...")
        return self.generate(full_prompt)
