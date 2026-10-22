"""Text -> vector.

  hashing                 No downloads, deterministic, decent for keyword-heavy policy text.
                          Good for tests and for a baseline row in your results table.
  sentence-transformers   Real semantic embeddings (pip install sentence-transformers).
                          Default model: BAAI/bge-small-en-v1.5 (384 dimensions).
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np

from ..config import get_settings

TOKEN = re.compile(r"[a-z0-9$§.]+")
STOP = {"the", "a", "an", "of", "to", "and", "or", "is", "are", "in", "on", "for", "by", "your", "you",
        "we", "our", "this", "that", "with", "any", "be", "it", "as", "at", "not", "under"}


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        words = [w.strip(".") for w in TOKEN.findall(text.lower())]
        words = [w for w in words if w and w not in STOP]
        stems = [w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words]
        return stems + [f"{a}_{b}" for a, b in zip(stems, stems[1:])]

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in self._tokens(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.name = model_name
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)


@lru_cache
def get_embedder():
    s = get_settings()
    if s.embedder == "sentence-transformers":
        return SentenceTransformerEmbedder(s.embedding_model)
    return HashingEmbedder()
