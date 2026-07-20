# -*- coding: utf-8 -*-
"""saber.api.main

FastAPI backend for the SABER UI.
Exposes endpoints for querying the system, listing specialists,
and managing configurations.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os

from saber.config import SaberConfig, VerificationTier
from saber.registry import SpecialistRegistry
from saber.audit import AuditLogger
from saber.orchestrator import Orchestrator
from saber.chat_history import ChatHistory

app = FastAPI(title="SABER API", version="0.1.0")

# Enable CORS for local UI development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global SABER components
from saber.frontend_chatbot import FrontendChatbot
from fastapi.responses import StreamingResponse
import json
import asyncio
import threading

# Global SABER components
config = SaberConfig.from_env()

# Dynamically plug in fine-tuned meta-reasoner if it exists
if os.path.exists("models/meta_reasoner_v2"):
    config.base_model = "models/meta_reasoner_v2"

audit = AuditLogger(log_path=config.audit_log_path)
registry = SpecialistRegistry(persist_path=config.db_path)
registry.auto_discover()

# Dynamically plug in fine-tuned specialists if they exist
for domain in ["medical", "science", "cyber", "finance", "coding", "architecture"]:
    spec = registry.get(domain)
    if spec:
        v2_path = f"models/{domain}_v2"
        if os.path.exists(v2_path):
            spec.load_model(v2_path)

orchestrator = Orchestrator(config=config, registry=registry, audit=audit)
chat_history = ChatHistory(db_path="data/chat_history.db")
chatbot = FrontendChatbot()

@app.on_event("startup")
async def startup_event():
    # Warm up frontend chatbot in a background thread to prevent blocking FastAPI startup
    threading.Thread(target=chatbot.warm_up, daemon=True).start()


class QueryRequest(BaseModel):
    query: str
    verification_tier: Optional[int] = None


@app.post("/api/query")
async def run_query(req: QueryRequest):
    """Run a query through the SABER pipeline with streaming support."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    tier = None
    if req.verification_tier is not None:
        try:
            tier = VerificationTier(req.verification_tier)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid verification tier.")

    # Save user message to history
    chat_history.add_message(role="user", content=req.query)

    async def event_generator():
        loop = asyncio.get_event_loop()
        
        # 1. Run domain classification in a background thread
        try:
            activated_domains = await loop.run_in_executor(
                None, 
                lambda: [
                    d for d, score in orchestrator.classify_domains(req.query).items()
                    if score >= orchestrator.config.activation_threshold
                ]
            )
        except Exception as e:
            print(f"[API] Domain classification failed: {e}")
            activated_domains = []

        # 2. Check if we have activated expert domains
        if not activated_domains:
            # General conversation / Greeting: stream from SmolLM frontend agent
            history = chat_history.get_messages()[:-1]  # Exclude the current query
            smol_history = [{"role": m["role"], "content": m["content"]} for m in history]
            smol_history.append({"role": "user", "content": req.query})

            full_response = ""
            try:
                for token in chatbot.generate_response_stream(req.query, history=smol_history):
                    full_response += token
                    yield json.dumps({"type": "smol_delta", "content": token}) + "\n"
                    await asyncio.sleep(0.005)
            except Exception as e:
                full_response = f"Hello! How can I help you today? (Fallback: {e})"
                yield json.dumps({"type": "smol_delta", "content": full_response}) + "\n"

            # Save Smol's response to DB history
            chat_history.add_message(
                role="system",
                content=full_response,
                metadata={"query_id": "chitchat", "status": "complete", "domains_activated": []}
            )
            return

        # 3. If real domain query: Stream a ping to Smol to tell user we are processing
        domains_list = ", ".join([d.capitalize() for d in activated_domains])
        ping_msg = f"I am activating our expert pipeline for {domains_list} to analyze this for you..."
        yield json.dumps({"type": "ping", "content": ping_msg}) + "\n"

        # 4. Run full pipeline in background thread
        try:
            result = await loop.run_in_executor(
                None,
                lambda: orchestrator.process_query(req.query, tier=tier)
            )
            
            # Stream the final response
            yield json.dumps({
                "type": "expert_answer",
                "answer": result.get("answer", ""),
                "domains_activated": result.get("domains_activated", []),
                "verification_cycles": result.get("verification_cycles", 0),
                "confidence": result.get("confidence", 1.0),
                "unresolved_flags": result.get("unresolved_flags", []),
                "status": "complete"
            }) + "\n"

            # Save response to history
            chat_history.add_message(
                role="system",
                content=result.get("answer", ""),
                metadata={
                    "query_id": result.get("query_id"),
                    "status": "complete",
                    "domains_activated": result.get("domains_activated"),
                    "verification_cycles": result.get("verification_cycles"),
                    "confidence": result.get("confidence"),
                }
            )
        except Exception as e:
            err_msg = f"Error in background pipeline: {e}"
            yield json.dumps({"type": "error", "content": err_msg}) + "\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/specialists")
async def list_specialists():
    """List all registered specialists and their health."""
    specs = registry.all()
    return [
        {
            "domain": d,
            "id": s.meta.specialist_id,
            "health": s.meta.health.value,
            "authority_score": s.meta.authority_score,
            "capabilities": s.meta.capabilities
        }
        for d, s in specs.items()
    ]


@app.get("/api/history")
async def get_history():
    """Return all chat messages in chronological order."""
    return chat_history.get_messages()


@app.delete("/api/history")
async def clear_history():
    """Clear all chat history and start a fresh conversation."""
    chat_history.clear()
    return {"status": "cleared"}


# Mount the static UI directory if it exists
ui_dir = os.path.join(os.path.dirname(__file__), "..", "ui")
if os.path.exists(ui_dir):
    app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")

