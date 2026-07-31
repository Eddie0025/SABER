import logging
from saber.specialists.base_spec import BaseSpecialist

logger = logging.getLogger("SABER_ScienceSpecialist")

class ScienceSpecialist(BaseSpecialist):
    """
    DoRA specialist fine-tuned for physics, chemistry, and advanced mathematics.
    """
    def __init__(self, base_model_id: str, adapter_path: str):
        super().__init__(base_model_id, adapter_path)
        self.system_prompt = (
            "You are a PhD-level scientist and mathematician. "
            "Provide rigorous, mathematically sound, and empirically backed explanations."
        )

    def process_query(self, prompt: str) -> str:
        full_prompt = f"<|im_start|>system\\n{self.system_prompt}<|im_end|>\\n"
        full_prompt += f"<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n"
        
        logger.info("Executing Science DoRA Inference...")
        return self.generate(full_prompt)
