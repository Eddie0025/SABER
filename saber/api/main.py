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


class QueryRequest(BaseModel):
    query: str
    verification_tier: Optional[int] = None


@app.post("/api/query")
async def run_query(req: QueryRequest):
    """Run a query through the SABER pipeline."""
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

    result = orchestrator.process_query(req.query, tier=tier)

    # Save system response to history with metadata
    chat_history.add_message(
        role="system",
        content=result.get("answer", ""),
        metadata={
            "query_id": result.get("query_id"),
            "status": result.get("status"),
            "confidence": result.get("confidence"),
            "domains_activated": result.get("domains_activated"),
            "verification_cycles": result.get("verification_cycles"),
            "total_flags_raised": result.get("total_flags_raised", 0),
            "total_flags_resolved": result.get("total_flags_resolved", 0),
            "unresolved_flags": result.get("unresolved_flags", []),
            "latency_seconds": result.get("latency_seconds", 0),
        },
    )

    return result


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

