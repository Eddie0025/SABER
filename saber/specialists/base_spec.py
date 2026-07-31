import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger("SABER_BaseSpecialist")

class BaseSpecialist:
    """
    Base class for executing a trained DoRA adapter for inference.
    """
    def __init__(self, base_model_id: str, adapter_path: str):
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        
    def load(self, base_model: AutoModelForCausalLM = None, tokenizer: AutoTokenizer = None):
        """
        Loads the DoRA adapter. If the base model is already in memory 
        (e.g., shared across specialists), it will inject the adapter into it dynamically.
        """
        logger.info(f"Loading specialist adapter from {self.adapter_path}...")
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        else:
            self.tokenizer = tokenizer
            
        if base_model is None:
            base_model = AutoModelForCausalLM.from_pretrained(self.base_model_id, device_map="auto")
            
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        logger.info("Adapter loaded successfully.")
        
    def generate(self, prompt: str, max_new_tokens=512) -> str:
        """Run inference through the adapter."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
