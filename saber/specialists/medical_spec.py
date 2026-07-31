import logging
from saber.specialists.base_spec import BaseSpecialist

logger = logging.getLogger("SABER_MedicalSpecialist")

class MedicalSpecialist(BaseSpecialist):
    """
    DoRA specialist fine-tuned on PubMed and medical diagnostics.
    """
    def __init__(self, base_model_id: str, adapter_path: str):
        super().__init__(base_model_id, adapter_path)
        self.system_prompt = (
            "You are an expert diagnostician and medical researcher. "
            "Base all advice on peer-reviewed medical literature. "
            "Always include a disclaimer that this is not professional medical advice."
        )

    def process_query(self, prompt: str) -> str:
        full_prompt = f"<|im_start|>system\\n{self.system_prompt}<|im_end|>\\n"
        full_prompt += f"<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n"
        
        logger.info("Executing Medical DoRA Inference...")
        return self.generate(full_prompt)
