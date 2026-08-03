import logging
from typing import Optional, Dict, Any

try:
    import mlx.core as mx
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    
logger = logging.getLogger("MLXEngine")

class MLXEngine:
    """
    Context-managed MLX inference engine optimized for Apple Silicon (M-series Macs).
    Automatically clears the Metal cache on exit to prevent memory leaks.
    """
    def __init__(self, model_path: str, adapter_path: Optional[str] = None):
        if not MLX_AVAILABLE:
            raise ImportError("mlx and mlx_lm are required to run MLXEngine. Install them if on Apple Silicon.")
            
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        
    def __enter__(self):
        logger.info(f"Loading MLX model from {self.model_path}")
        if self.adapter_path:
            logger.info(f"Applying adapter from {self.adapter_path}")
            self.model, self.tokenizer = load(self.model_path, adapter_path=self.adapter_path)
        else:
            self.model, self.tokenizer = load(self.model_path)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Cleaning up MLX model and clearing Metal cache.")
        # Delete references to free memory
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        
        # Explicitly clear the Metal cache
        mx.metal.clear_cache()
        
    def generate_response(self, system_prompt: str, user_prompt: str, max_tokens: int = 512, temp: float = 0.7) -> str:
        """
        Generates a response using the ChatML template format.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Engine must be used as a context manager (with MLXEngine(...) as engine:)")
            
        # Format using ChatML
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        response = generate(
            self.model, 
            self.tokenizer, 
            prompt=prompt, 
            max_tokens=max_tokens, 
            temp=temp,
            verbose=False
        )
        return response
