import os
import gc
import logging
import torch
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from saber.config import (
    BASE_MODEL, SPECIALIST_REGISTRY, DOMAIN_SYSTEM_PROMPTS,
    INFERENCE_DTYPE, DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
)

logger = logging.getLogger("SABER_SpecialistEngine")

# Map string dtype to torch dtype
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class SpecialistEngine:
    """
    Core SABER inference engine.
    
    Keeps the Qwen2.5-7B-Instruct base model resident in VRAM and dynamically
    hot-swaps DoRA adapters for each specialist domain. Only one adapter is
    active at a time.
    
    Usage:
        engine = SpecialistEngine()
        engine.load_base_model()
        
        # Casual chat (no adapter)
        response = engine.generate_bare([{"role": "user", "content": "Hello!"}])
        
        # Specialist inference
        engine.load_adapter("cybersecurity")
        response = engine.generate([{"role": "user", "content": "Explain XSS"}])
        engine.unload_adapter()
    """

    def __init__(self):
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.current_adapter: Optional[str] = None
        self._loaded_adapters: set = set()  # Track which adapters are loaded into PEFT

    def load_base_model(self):
        """Load the base Qwen model into GPU. Called once at startup."""
        if self.model is not None:
            logger.info("Base model already loaded.")
            return

        dtype = DTYPE_MAP.get(INFERENCE_DTYPE, torch.bfloat16)
        logger.info(f"Loading base model {BASE_MODEL} in {INFERENCE_DTYPE}...")

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()
        logger.info("Base model loaded successfully.")

    def load_adapter(self, domain: str):
        """
        Hot-swap a DoRA specialist adapter onto the resident base model.
        If the adapter was previously loaded, just switch to it.
        If a different adapter is active, disable it first.
        """
        if self.model is None:
            raise RuntimeError("Base model not loaded. Call load_base_model() first.")

        # Resolve adapter path
        adapter_path = SPECIALIST_REGISTRY.get(domain)
        if not adapter_path:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(SPECIALIST_REGISTRY.keys())}")

        # Check for fallback path
        if not os.path.exists(adapter_path):
            fallback = adapter_path.replace("_grpo_final", "_grpo")
            if os.path.exists(fallback):
                adapter_path = fallback
            else:
                fallback_v2 = adapter_path.replace("_grpo_final", "_v2")
                if os.path.exists(fallback_v2):
                    adapter_path = fallback_v2
                else:
                    raise FileNotFoundError(
                        f"Adapter not found for {domain}: tried {adapter_path}, {fallback}, {fallback_v2}"
                    )

        if self.current_adapter == domain:
            logger.info(f"Adapter '{domain}' already active.")
            return

        # If this adapter was previously loaded into the PEFT model, just switch
        if domain in self._loaded_adapters:
            logger.info(f"Switching to cached adapter '{domain}'...")
            self.model.set_adapter(domain)
            self.current_adapter = domain
            return

        # Load new adapter
        logger.info(f"Loading adapter '{domain}' from {adapter_path}...")
        if not self._loaded_adapters:
            # First adapter — wrap the base model with PeftModel
            self.model = PeftModel.from_pretrained(
                self.model, adapter_path, adapter_name=domain
            )
            self.model.eval()
        else:
            # Additional adapter — load into existing PeftModel
            self.model.load_adapter(adapter_path, adapter_name=domain)
            self.model.set_adapter(domain)

        self._loaded_adapters.add(domain)
        self.current_adapter = domain
        logger.info(f"Adapter '{domain}' loaded and active.")

    def unload_adapter(self):
        """Disable the current adapter, reverting to the bare base model."""
        if self.current_adapter is None:
            return

        logger.info(f"Disabling adapter '{self.current_adapter}'...")
        self.model.disable_adapters()
        self.current_adapter = None

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a response using the currently active adapter (or bare model).
        
        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            system_prompt: Optional system prompt to prepend.
        """
        if self.model is None:
            raise RuntimeError("Base model not loaded. Call load_base_model() first.")

        # Inject system prompt if provided and not already in messages
        if system_prompt and (not messages or messages[0]["role"] != "system"):
            messages = [{"role": "system", "content": system_prompt}] + messages

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
            )

        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return response

    def generate_bare(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        """Generate using the bare base model with no adapter."""
        was_active = self.current_adapter
        if was_active:
            self.model.disable_adapters()

        response = self.generate(messages, max_new_tokens=max_new_tokens, temperature=temperature)

        if was_active:
            self.model.set_adapter(was_active)

        return response

    def generate_for_domain(
        self,
        domain: str,
        messages: List[Dict[str, str]],
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> str:
        """Convenience: load adapter, inject system prompt, generate, return."""
        self.load_adapter(domain)
        system_prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, f"You are a {domain} AI specialist.")
        return self.generate(messages, max_new_tokens=max_new_tokens, system_prompt=system_prompt)

    def shutdown(self):
        """Release all GPU memory."""
        logger.info("Shutting down SpecialistEngine...")
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.current_adapter = None
        self._loaded_adapters.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Engine shut down. GPU memory released.")
