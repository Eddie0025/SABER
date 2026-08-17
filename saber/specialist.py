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

# Check available backends
try:
    import mlx.core as mx
    from mlx_lm import load as mlx_load, generate as mlx_generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


class SpecialistEngine:
    """
    Core SABER inference engine.
    
    Supports:
    - MLX 4-bit (Apple Silicon Metal GPU acceleration - ultra-lightweight ~4.5GB VRAM)
    - PyTorch (CUDA / Apple Silicon MPS / CPU)
    
    Keeps the Qwen2.5-7B-Instruct base model resident and dynamically hot-swaps
    DoRA specialist adapters.
    """

    def __init__(self, use_4bit: bool = True):
        self.use_4bit = use_4bit
        self.model = None
        self.tokenizer = None
        self.current_adapter: Optional[str] = None
        self._loaded_adapters: set = set()
        self.backend = "torch"  # "mlx" or "torch"
        self.device = "cpu"

    def load_base_model(self):
        """Load the base Qwen model into GPU/VRAM. Called once at startup."""
        if self.model is not None:
            logger.info("Base model already loaded.")
            return

        # Check if Apple Silicon + 4-bit MLX is available
        if self.use_4bit and MLX_AVAILABLE and torch.backends.mps.is_available():
            try:
                self.backend = "mlx"
                mlx_model_id = "mlx-community/Qwen2.5-7B-Instruct-4bit"
                logger.info(f"🚀 Loading SABER in 4-bit Metal (MLX) mode from {mlx_model_id}...")
                self.model, self.tokenizer = mlx_load(mlx_model_id)
                logger.info("✅ 4-bit MLX base model loaded successfully onto Apple Silicon GPU.")
                return
            except Exception as e:
                logger.warning(f"MLX 4-bit load encountered: {e}. Falling back to PyTorch MPS/CUDA.")

        # PyTorch Backend
        self.backend = "torch"
        if torch.cuda.is_available():
            self.device = "cuda"
            dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            dtype = torch.float16
        else:
            self.device = "cpu"
            dtype = torch.float32

        logger.info(f"Loading PyTorch base model {BASE_MODEL} on {self.device} ({dtype})...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
            device_map="auto" if self.device != "mps" else None,
        )
        if self.device == "mps":
            self.model.to("mps")
        self.model.eval()
        logger.info(f"✅ PyTorch base model loaded successfully on {self.device}.")

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

        if self.backend == "mlx":
            response = mlx_generate(
                self.model,
                self.tokenizer,
                prompt=prompt_text,
                max_tokens=max_new_tokens,
                temp=temperature if temperature > 0 else 0.0,
                verbose=False
            )
            return response.strip()

        # PyTorch generation
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
