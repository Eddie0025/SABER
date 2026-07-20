# -*- coding: utf-8 -*-
"""saber.frontend_chatbot

Lightweight front-facing chitchat/greeting assistant using SmolLM2-360M-Instruct.
Responds instantly to chitchat and provides real-time acknowledgments while 
heavy domain pipelines run in the background.
"""

from __future__ import annotations

import os
from typing import Generator
from saber.llm_engine import LLMEngine

class FrontendChatbot:
    """Manages the lightweight SmolLM2-360M-Instruct front-facing agent."""

    def __init__(self, model_id: str = "HuggingFaceTB/SmolLM2-360M-Instruct"):
        self.model_id = model_id
        # Pre-cache or warm up on startup if needed
        self._warmed = False

    def warm_up(self) -> None:
        """Load tokenizer/model weights into memory to prevent first-hit latency."""
        if self._warmed:
            return
        try:
            print(f"[*] Pre-loading frontend chatbot ({self.model_id})...")
            # We open the engine once to download/cache the model
            with LLMEngine(self.model_id) as engine:
                pass
            self._warmed = True
            print("[+] Frontend chatbot warmed up and ready.")
        except Exception as e:
            print(f"[!] Warning: Failed to warm up frontend chatbot: {e}")

    def generate_response_stream(self, query: str, history: list[dict] = None) -> Generator[str, None, None]:
        """Stream a friendly instant response for greetings/chitchat."""
        from transformers import TextIteratorStreamer
        from threading import Thread
        import torch

        # Build prompt using chat template
        messages = []
        if history:
            # Only keep the last 5 turns to stay responsive and fast
            messages.extend(history[-10:])
        else:
            messages.append({"role": "user", "content": query})

        # Ensure system prompt is present
        if not any(m["role"] == "system" for m in messages):
            messages.insert(0, {
                "role": "system", 
                "content": "You are SABER, a helpful and friendly AI assistant. Keep your responses short, conversational, and direct."
            })

        try:
            # We load the Smol model using LLMEngine
            with LLMEngine(self.model_id) as engine:
                model = engine.model
                tokenizer = engine.tokenizer
                
                # Apply chat template
                inputs = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=True, 
                    add_generation_prompt=True, 
                    return_tensors="pt"
                ).to(engine.device)

                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                generation_kwargs = dict(
                    input_ids=inputs,
                    streamer=streamer,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )

                # Run generation in a separate thread so we can stream chunks
                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()

                for new_text in streamer:
                    yield new_text

                thread.join()
        except Exception as e:
            print(f"[FrontendChatbot] Error generating stream: {e}")
            yield "Hello! How can I help you today?"
