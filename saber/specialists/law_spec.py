import logging
from saber.specialists.base_spec import BaseSpecialist

logger = logging.getLogger("SABER_LawSpecialist")

class LawSpecialist(BaseSpecialist):
    """
    DoRA specialist fine-tuned for legal analysis and contract synthesis.
    """
    def __init__(self, base_model_id: str, adapter_path: str):
        super().__init__(base_model_id, adapter_path)
        self.system_prompt = (
            "You are an expert corporate lawyer and legal analyst. "
            "Analyze documents with strict attention to liability, jurisdiction, and legal precedent."
        )

    def process_query(self, prompt: str) -> str:
        full_prompt = f"<|im_start|>system\\n{self.system_prompt}<|im_end|>\\n"
        full_prompt += f"<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n"
        
        logger.info("Executing Law DoRA Inference...")
        return self.generate(full_prompt)
