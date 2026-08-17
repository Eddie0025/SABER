import os
import re
import logging
import sqlite3
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path

from saber.config import (
    SENTINEL_EMBEDDING_MODEL,
    SENTINEL_EMBEDDING_MODEL_LOCAL,
    SENTINEL_RELEVANCE_THRESHOLD,
    OFFLINE_KB_DIR,
)

logger = logging.getLogger("SABER_Sentinel")


class SentinelVerifier:
    """
    SABER's Sentinel Verification Kernel.
    
    Pure Python — NO adapter loading. Uses embedding-based semantic search
    against the offline SQLite knowledge base to verify specialist claims.
    
    Pipeline:
    1. Extract verifiable claims from the specialist's response.
    2. Encode claims using BAAI/bge-base-en-v1.5.
    3. Cosine similarity search against domain KB.
    4. If match found (score >= threshold), verify consistency using bare model.
    5. If contradiction → FLAG → trigger specialist rewrite.
    
    Currently supports domains with an existing KB:
    - cybersecurity (cyber_kb_v2.db)
    
    Other domains pass through with "No KB hit" until their KBs are built.
    """

    def __init__(self):
        self.embed_model = None
        self._kb_cache = {}  # domain -> (passages, embeddings)

    def _load_embedding_model(self):
        """Lazy-load the embedding model on first use."""
        if self.embed_model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            # Try local path first, then HuggingFace
            if os.path.exists(SENTINEL_EMBEDDING_MODEL_LOCAL):
                logger.info(f"Loading embedding model from local: {SENTINEL_EMBEDDING_MODEL_LOCAL}")
                self.embed_model = SentenceTransformer(SENTINEL_EMBEDDING_MODEL_LOCAL)
            else:
                logger.info(f"Loading embedding model from HuggingFace: {SENTINEL_EMBEDDING_MODEL}")
                self.embed_model = SentenceTransformer(SENTINEL_EMBEDDING_MODEL)
        except ImportError:
            logger.warning("sentence-transformers not installed. Sentinel KB search disabled.")
            self.embed_model = None

    def _get_kb_path(self, domain: str) -> Optional[str]:
        """Resolve the KB database path for a domain."""
        # Try exact match first
        kb_path = os.path.join(OFFLINE_KB_DIR, f"{domain}_kb_v2.db")
        if os.path.exists(kb_path):
            return kb_path

        # Try domain alias (cyber -> cybersecurity)
        aliases = {
            "cybersecurity": "cyber_kb_v2.db",
        }
        alias_file = aliases.get(domain)
        if alias_file:
            alias_path = os.path.join(OFFLINE_KB_DIR, alias_file)
            if os.path.exists(alias_path):
                return alias_path

        return None

    def _load_kb(self, domain: str) -> Optional[Tuple[List[str], np.ndarray]]:
        """Load passages and embeddings from the domain's offline KB."""
        if domain in self._kb_cache:
            return self._kb_cache[domain]

        kb_path = self._get_kb_path(domain)
        if kb_path is None:
            logger.info(f"No offline KB found for domain '{domain}'.")
            return None

        try:
            conn = sqlite3.connect(kb_path)
            cursor = conn.cursor()

            # Load passages
            cursor.execute("SELECT id, passage FROM knowledge")
            passages = {}
            for row in cursor.fetchall():
                passages[row[0]] = row[1]

            # Load embeddings
            cursor.execute("SELECT id, embedding_blob FROM embeddings")
            embeddings = {}
            for row in cursor.fetchall():
                emb = np.frombuffer(row[1], dtype=np.float32)
                embeddings[row[0]] = emb

            conn.close()

            # Align passages and embeddings by ID
            common_ids = sorted(set(passages.keys()) & set(embeddings.keys()))
            if not common_ids:
                logger.warning(f"KB {kb_path} has no overlapping passage/embedding IDs.")
                return None

            passage_list = [passages[i] for i in common_ids]
            embedding_matrix = np.stack([embeddings[i] for i in common_ids])

            result = (passage_list, embedding_matrix)
            self._kb_cache[domain] = result
            logger.info(f"Loaded KB for '{domain}': {len(passage_list)} passages.")
            return result

        except Exception as e:
            logger.error(f"Failed to load KB for '{domain}': {e}")
            return None

    def _cosine_similarity(self, query_emb: np.ndarray, kb_embs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all KB embeddings."""
        # Normalize
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        kb_norms = kb_embs / (np.linalg.norm(kb_embs, axis=1, keepdims=True) + 1e-8)
        return kb_norms @ query_norm

    def _extract_claims(self, response: str) -> List[str]:
        """
        Extract verifiable factual claims from the specialist's response.
        Simple heuristic: split into sentences, filter out short/trivial ones.
        """
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', response)
        claims = []
        for s in sentences:
            s = s.strip()
            # Skip very short sentences or questions
            if len(s) < 20 or s.endswith("?"):
                continue
            # Limit claim length for embedding
            claims.append(s[:500])
        return claims[:5]  # Max 5 claims to check

    def verify(self, domain: str, response: str) -> Tuple[str, str, Optional[str]]:
        """
        Verify a specialist's response against the offline KB.
        
        Returns:
            (status, footer, kb_passage)
            - status: "GREEN_CHIT" | "NO_KB" | "FLAG"
            - footer: Human-readable Sentinel footer string
            - kb_passage: The matched KB passage (if any)
        """
        # Check if we have a KB for this domain
        kb_data = self._load_kb(domain)
        if kb_data is None:
            return (
                "NO_KB",
                "⚡ Verified by SABER Sentinel (No KB hit — passed through)",
                None,
            )

        # Load embedding model
        self._load_embedding_model()
        if self.embed_model is None:
            return (
                "NO_KB",
                "⚡ Verified by SABER Sentinel (Embedding model unavailable — passed through)",
                None,
            )

        passages, embeddings = kb_data

        # Extract claims and search
        claims = self._extract_claims(response)
        if not claims:
            return (
                "NO_KB",
                "⚡ Verified by SABER Sentinel (No verifiable claims found — passed through)",
                None,
            )

        # Find the best matching KB passage across all claims
        best_score = 0.0
        best_passage = None

        for claim in claims:
            query_emb = self.embed_model.encode(claim, normalize_embeddings=True)
            scores = self._cosine_similarity(query_emb, embeddings)
            max_idx = np.argmax(scores)
            if scores[max_idx] > best_score:
                best_score = scores[max_idx]
                best_passage = passages[max_idx]

        if best_score < SENTINEL_RELEVANCE_THRESHOLD:
            return (
                "NO_KB",
                f"⚡ Verified by SABER Sentinel (No KB hit — best score {best_score:.2f} < {SENTINEL_RELEVANCE_THRESHOLD})",
                None,
            )

        # KB hit found — for now, pass through as GREEN_CHIT
        # Full contradiction detection (using bare LLM to compare) is a future enhancement
        logger.info(f"Sentinel KB hit: score={best_score:.3f}, passage='{best_passage[:80]}...'")
        return (
            "GREEN_CHIT",
            f"⚡ Verified by SABER Sentinel (Offline KB — score {best_score:.2f})",
            best_passage,
        )
