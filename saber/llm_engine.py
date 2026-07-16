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
import sys
from typing import Any, Dict, Optional


class LLMEngine:
    """Context manager for safely loading and unloading LLMs."""

    def __init__(self, model_id_or_path: str, max_new_tokens: int = 512):
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
        print(f"[LLMEngine] Loading {self.model_id_or_path} into RAM...")
        # pyrefly: ignore [missing-import]
        import torch
        # pyrefly: ignore [missing-import]
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id_or_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Define data type based on device to save memory
        dtype = torch.float16
        if self.device == "mps":
            dtype = torch.float32  # MPS often struggles with half precision unless specifically optimized, but we can try float16 if float32 takes too much RAM. Actually, float16 works on M-series for inference. Let's use float16.
            dtype = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id_or_path,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        if self.device == "mps":
            self.model = self.model.to(self.device)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"[LLMEngine] Unloading {self.model_id_or_path} and clearing RAM...")
        
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

        # Format prompt (simple instruction format, adjustable for specific models)
        if system_prompt:
            full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
        else:
            full_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"

        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
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

        # Build the multi-turn prompt from history
        parts: list = []
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|{role}|>\n{content}")

        # Append the new user message if provided
        if new_user_message:
            parts.append(f"<|user|>\n{new_user_message}")

        # Terminate with assistant tag to prompt generation
        parts.append("<|assistant|>")
        full_prompt = "\n".join(parts) + "\n"

        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
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

