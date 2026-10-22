"""Similar-claims search with OpenSearch hybrid retrieval (booklet Phase 2.4, study guide C7).

    docker compose --profile search up -d opensearch
    python -m training.fraud.similar_claims index          # index the claims history
    python -m training.fraud.similar_claims query G000005  # find look-alikes for a gold claim

Hybrid = keyword (BM25 on the narrative + exact matches on phone / shop / VIN) combined with
vector similarity of the narrative, merged with Reciprocal Rank Fusion (RRF).
Needs: pip install opensearch-py sentence-transformers
"""
from __future__ import annotations

import json
import sys

import pandas as pd

from claimsense.config import get_settings
from claimsense.rag.embeddings import get_embedder

INDEX = "claims_history"


def client():
    from opensearchpy import OpenSearch
    import os
    return OpenSearch(os.environ.get("OPENSEARCH_URL", "http://localhost:9200"))


def create_index(os_client, dim: int) -> None:
    if os_client.indices.exists(INDEX):
        os_client.indices.delete(INDEX)
    os_client.indices.create(INDEX, body={
        "settings": {"index": {"knn": True}},
        "mappings": {"properties": {
            "claim_id": {"type": "keyword"}, "customer_id": {"type": "keyword"},
            "repair_shop": {"type": "keyword"}, "contact_phone": {"type": "keyword"}, "vin": {"type": "keyword"},
            "incident_type": {"type": "keyword"}, "narrative": {"type": "text"},
            "report_date": {"type": "date"}, "claim_amount": {"type": "float"},
            "historical_decision": {"type": "keyword"},
            "embedding": {"type": "knn_vector", "dimension": dim},
        }}})


def index_history(batch: int = 1000) -> None:
    from opensearchpy.helpers import bulk
    s = get_settings()
    df = pd.read_parquet(s.synthetic_dir / "tabular" / "claims_history.parquet")
    emb = get_embedder()
    oc = client()
    create_index(oc, emb.dim)
    for start in range(0, len(df), batch):
        part = df.iloc[start:start + batch]
        vecs = emb.embed(part["narrative"].tolist())
        actions = [{"_index": INDEX, "_id": r.claim_id, "_source": {
            "claim_id": r.claim_id, "customer_id": r.customer_id, "repair_shop": r.repair_shop,
            "contact_phone": r.contact_phone, "vin": r.vin, "incident_type": r.incident_type,
            "narrative": r.narrative, "report_date": str(r.report_date), "claim_amount": float(r.claim_amount),
            "historical_decision": r.historical_decision, "embedding": v.tolist()}}
            for r, v in zip(part.itertuples(), vecs)]
        bulk(oc, actions)
        print(f"indexed {start + len(part):,}/{len(df):,}")


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda t: -t[1])


def similar(narrative: str, shop: str | None = None, phone: str | None = None, vin: str | None = None,
            exclude_customer: str | None = None, k: int = 5) -> list[dict]:
    oc = client()
    should = [{"match": {"narrative": narrative}}]
    should += [{"term": {f: {"value": v, "boost": 3}}} for f, v in (("repair_shop", shop), ("contact_phone", phone), ("vin", vin)) if v]
    must_not = [{"term": {"customer_id": exclude_customer}}] if exclude_customer else []
    kw = oc.search(index=INDEX, body={"size": 50, "query": {"bool": {"should": should, "must_not": must_not}}})
    vec = get_embedder().embed([narrative])[0].tolist()
    kn = oc.search(index=INDEX, body={"size": 50, "query": {"knn": {"embedding": {"vector": vec, "k": 50}}}})
    fused = rrf([[h["_id"] for h in kw["hits"]["hits"]], [h["_id"] for h in kn["hits"]["hits"]]])[:k]
    docs = {h["_id"]: h["_source"] for h in kw["hits"]["hits"] + kn["hits"]["hits"]}
    out = []
    for cid, score in fused:
        d = docs[cid]
        matches = [f for f, v in (("repair_shop", shop), ("contact_phone", phone), ("vin", vin)) if v and d.get(f) == v]
        out.append({"claim_id": cid, "score": round(score, 4), "matches": matches,
                    "incident_type": d["incident_type"], "decision": d["historical_decision"]})
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["index"]:
        index_history()
    elif sys.argv[1:2] == ["query"]:
        gold_id = sys.argv[2]
        s = get_settings()
        from claimsense.speech import voice_script
        g = next(json.loads(line) for line in (s.synthetic_dir / "gold" / "gold_claims.jsonl").open() if gold_id in line)
        print(json.dumps(similar(voice_script(gold_id) or "", shop=g["expected"]["voice_facts"]["repair_shop"],
                                 exclude_customer=g["customer_id"]), indent=2))
    else:
        print(__doc__)
