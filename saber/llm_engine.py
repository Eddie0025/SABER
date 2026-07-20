# -*- coding: utf-8 -*-
"""saber.llm_engine

Dynamic Model Swapping Engine.
Ensures only ONE model is loaded into RAM at any given time to prevent
out-of-memory errors on constrained hardware (like a 16GB Mac).

Usage:
    with LLMEngine("Qwen/Qwen2.5-7B") as engine:
        response = engine.generate("Prompt text")
    # Model is automatically unloaded and RAM cleared here.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from saber.context import SessionMemory


_MODEL_CACHE = {}

class LLMEngine:
    """Context manager for safely loading and unloading LLMs."""

    def __init__(self, model_id_or_path: str, max_new_tokens: int = 2048):
        self.model_id_or_path = model_id_or_path
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()

    def _get_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        except ImportError:
            return "cpu"

    def __enter__(self) -> "LLMEngine":
        # Check cache first (ignore token limit for weight caching)
        cache_key = self.model_id_or_path
        if cache_key in _MODEL_CACHE:
            self.model, self.tokenizer = _MODEL_CACHE[cache_key]
            return self

        # pyrefly: ignore [missing-import]
        import torch
        # pyrefly: ignore [missing-import]
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import transformers
        transformers.logging.set_verbosity_error()

        # Setup tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id_or_path, trust_remote_code=True
            )
        except Exception:
            # Fallback for PEFT adapters with corrupted tokenizer_config
            import json
            import os
            config_path = os.path.join(self.model_id_or_path, "adapter_config.json")
            with open(config_path, "r") as f:
                base_model = json.load(f)["base_model_name_or_path"]
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Define data type based on device to save memory
        dtype = torch.float16
        if self.device == "mps":
            dtype = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id_or_path,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        if self.device == "mps":
            try:
                self.model = self.model.to(self.device)
            except Exception:
                pass

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import os
        if os.getenv("SABER_KEEP_MODELS_LOADED") == "1":
            # Cache weights instead of unloading to save swap time
            cache_key = self.model_id_or_path
            _MODEL_CACHE[cache_key] = (self.model, self.tokenizer)
            self.model = None
            self.tokenizer = None
            return
            
        # Delete references
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None

        # Force Python garbage collection
        gc.collect()

        # Force PyTorch to release memory back to the OS
        try:
            import torch
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()
        except Exception as e:
            print(f"[LLMEngine] Warning: Could not clear hardware cache: {e}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response using the loaded model."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Use 'with LLMEngine(...) as engine:'")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        full_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get all possible end of sequence token IDs
        eos_ids = [self.tokenizer.eos_token_id]
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int):
            eos_ids.append(im_end_id)
            
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=eos_ids,
        )

        # Decode only the newly generated tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()

    def generate_with_history(
        self,
        history: list,
        new_user_message: Optional[str] = None,
    ) -> str:
        """Generate a response given a multi-turn conversation history.

        Parameters
        ----------
        history : list[dict]
            List of ``{"role": "system"|"user"|"assistant", "content": "..."}``
            dicts representing the conversation so far.
        new_user_message : str or None
            If provided, appended as a new user turn before generation.

        Returns
        -------
        str
            The model's generated assistant response.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Use 'with LLMEngine(...) as engine:'")

        # Copy history so we don't mutate the original
        messages = list(history)

        # Append the new user message if provided
        if new_user_message:
            messages.append({"role": "user", "content": new_user_message})

        full_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get all possible end of sequence token IDs
        eos_ids = [self.tokenizer.eos_token_id]
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int):
            eos_ids.append(im_end_id)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=eos_ids,
        )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()

    def generate_from_session(
        self,
        session_memory: "SessionMemory",
        session_id: str,
        user_message: str,
    ) -> str:
        """Convenience: add user message to session, generate, store response.

        This is the main entry point for multi-turn chat. It:
        1. Adds the user message to the session history.
        2. Retrieves the full truncated history.
        3. Generates the assistant response.
        4. Stores the assistant response back into the session.
        5. Returns the response text.

        Parameters
        ----------
        session_memory : SessionMemory
            The session memory manager (from ``saber.context``).
        session_id : str
            The session to use. Must already exist in ``session_memory``.
        user_message : str
            The new user turn.

        Returns
        -------
        str
            The generated assistant response.
        """
        # 1. Store the user message
        session_memory.add_message(session_id, "user", user_message)

        # 2. Get the truncated history
        history = session_memory.get_history(session_id)

        # 3. Generate (history already includes the new user message)
        response = self.generate_with_history(history)

        # 4. Store the assistant response
        session_memory.add_message(session_id, "assistant", response)

        return response

