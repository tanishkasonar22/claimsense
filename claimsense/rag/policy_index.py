"""Chunk policies by section and search them.

Chunking by the policy's own structure (one chunk per numbered section) keeps citations
exact: every retrieved chunk IS a citable section like §3.1.

Backends:
  InMemoryIndex   NumPy cosine similarity. Lite mode and tests.
  QdrantIndex     Qdrant vector database (QDRANT_URL=http://localhost:6333). Full mode.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import get_settings
from .embeddings import get_embedder

SECTION = re.compile(r"^\*\*(§\d+\.\d+)([^*]*)\*\*\s*(.*)$")
PART = re.compile(r"^##\s+(Part \d+\.\s+.+)$")


@dataclass(frozen=True)
class PolicyChunk:
    policy_code: str
    section: str        # "§3.1"
    part: str           # "Part 3. Coverage for Damage to Your Auto"
    text: str           # full section text, used for citations

    @property
    def embed_text(self) -> str:
        return f"{self.part}. {self.section} {self.text}"


def chunk_policy(markdown: str, policy_code: str) -> list[PolicyChunk]:
    chunks, part = [], ""
    for line in markdown.splitlines():
        line = line.strip()
        if m := PART.match(line):
            part = m.group(1)
        elif m := SECTION.match(line):
            title = m.group(2).strip()
            body = m.group(3).strip()
            text = f"{title} {body}".strip() if title else body
            chunks.append(PolicyChunk(policy_code, m.group(1), part, text.replace("**", "")))
    return chunks


def load_all_chunks(policies_dir: Path) -> list[PolicyChunk]:
    out = []
    for path in sorted(policies_dir.glob("*.md")):
        out.extend(chunk_policy(path.read_text(encoding="utf-8"), path.stem))
    return out


class InMemoryIndex:
    def __init__(self, chunks: list[PolicyChunk]):
        self.embedder = get_embedder()
        self.chunks = chunks
        self.vectors = self.embedder.embed([c.embed_text for c in chunks]) if chunks else np.zeros((0, 1))

    def search(self, query: str, policy_code: str, k: int = 5) -> list[tuple[PolicyChunk, float]]:
        idx = [i for i, c in enumerate(self.chunks) if c.policy_code == policy_code]
        if not idx:
            return []
        q = self.embedder.embed([query])[0]
        scores = self.vectors[idx] @ q
        order = np.argsort(-scores)[:k]
        return [(self.chunks[idx[i]], float(scores[i])) for i in order]

    def get(self, policy_code: str, section: str) -> PolicyChunk | None:
        return next((c for c in self.chunks if c.policy_code == policy_code and c.section == section), None)


class QdrantIndex(InMemoryIndex):  # pragma: no cover - needs a running Qdrant
    COLLECTION = "policy_sections"

    def __init__(self, chunks: list[PolicyChunk], url: str):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
        self.embedder = get_embedder()
        self.chunks = chunks
        self.client = QdrantClient(url=url)
        vectors = self.embedder.embed([c.embed_text for c in chunks])
        if self.client.collection_exists(self.COLLECTION):
            self.client.delete_collection(self.COLLECTION)
        self.client.create_collection(self.COLLECTION, vectors_config=VectorParams(
            size=vectors.shape[1], distance=Distance.COSINE))
        self.client.upsert(self.COLLECTION, points=[
            PointStruct(id=i, vector=vectors[i].tolist(),
                        payload={"policy_code": c.policy_code, "section": c.section, "part": c.part, "text": c.text})
            for i, c in enumerate(chunks)])

    def search(self, query: str, policy_code: str, k: int = 5):
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        q = self.embedder.embed([query])[0].tolist()
        res = self.client.query_points(
            self.COLLECTION, query=q, limit=k,
            query_filter=Filter(must=[FieldCondition(key="policy_code", match=MatchValue(value=policy_code))]))
        return [(PolicyChunk(p.payload["policy_code"], p.payload["section"], p.payload["part"], p.payload["text"]),
                 float(p.score)) for p in res.points]


@lru_cache
def get_policy_index() -> InMemoryIndex:
    s = get_settings()
    chunks = load_all_chunks(s.policies_dir)
    if s.qdrant_url:
        return QdrantIndex(chunks, s.qdrant_url)
    return InMemoryIndex(chunks)
