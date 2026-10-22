# ClaimSense

**Multimodal AI claims processing with a human in the loop.** Photos, voice notes and documents go in; an evidence-backed, fairness-audited recommendation comes out; a human adjuster makes every final decision.

> All data is synthetic. No real people, policies or claims. Portfolio project.

## What it does

A customer submits a car damage claim: photos, a voice note and a repair estimate. ClaimSense:

1. transcribes the voice note (Whisper) and reads the documents (PDF text layer or OCR)
2. detects damage in the photos (YOLO, optimized with ONNX/TensorRT)
3. scans every piece of customer content for prompt injection and redacts personal data
4. extracts the claim facts (rules baseline, Gemini, or a QLoRA fine-tuned open model)
5. checks the customer's policy with RAG and cites the exact sections (§3.1, §4.5...)
6. cross-checks the story against the documents and photos
7. scores fraud risk with reasons (rules baseline or LightGBM with SHAP-style contributions)
8. drafts a recommendation, then **pauses** (LangGraph interrupt) until an adjuster decides in the dashboard

## Architecture

```
Customer ─► FastAPI ─┬─(inline)─► LangGraph agent ─────────────────────────────┐
                     └─(kafka)──► claims.submitted ─► speech / documents / vision workers ─► orchestrator
                                                                                 │
   speech ─┐                                                                      ▼
documents ─┼─► analyze: guardrails → facts → coverage RAG → consistency → fraud → recommend
   vision ─┘        │                   │                                   │
                    │             Qdrant (policies)              investigate (MCP tools)
                    ▼                                                       │
              Gemini / local model (router)                     human_review (interrupt) ─► Next.js dashboard
Stores: Postgres · MinIO · Redis      Offline: Airflow + dbt + Spark · Ray · MLflow · Langfuse · CI eval gate
```

## Results so far (Tier 1, offline rules baseline)

Measured with `python -m evals.run_gold` on 200 synthetic gold claims, no API key, laptop CPU:

| Area | Metric | Value |
|---|---|---|
| Documents | Repair estimates parsed exactly (digital PDFs) | 200 / 200 |
| Extraction | Incident type, parts, date, shop, consistency | 100% |
| Coverage | Correct covered / not covered | 96% |
| Coverage | Correct deductible when covered | 100% |
| Fraud | PR-AUC (rules baseline) | 0.93 |
| Decisions | Agreement with the correct decision | 89.5% |
| Safety | Red-team attacks blocked / benign messages wrongly blocked | 70 of 70 / 0 of 6 |
| Speed | Median seconds per claim | 0.02 |

**Read these honestly:**
- Extraction is perfect because the voice scripts are clean templates. The real test is Whisper transcripts of real recordings and OCR on phone photos. Expect lower numbers, which is the point of the LLM and fine-tuning work in Tier 2.
- The 8 coverage misses are exclusions the evidence can't reveal (hidden rideshare use, intentional damage, racing). That's realistic: they belong to the fraud check and the adjuster.
- The red-team rules were written while looking at these attacks, so 100% is optimistic. Write a fresh held-out attack set and report that number.
- Photos and voice notes used fallbacks until CarDD and real recordings are added (the eval prints a warning).

On the time-split history (train on claims before July 2025, test after), a gradient-boosted model beat the rules baseline: PR-AUC 0.894 → 0.928. That figure came from scikit-learn standing in for LightGBM; rerun `make fraud` for the LightGBM number.

## Repository map (matches the booklet phases)

| Folder | What | Booklet |
|---|---|---|
| `generators/`, `generate_all.py` | Synthetic customers, policies, claims, documents, voice scripts, red-team set, fine-tuning data | 1.1 |
| `claimsense/schemas.py` | Every data shape services exchange | 1.1 |
| `claimsense/api/` | FastAPI API, agent service, Gradio demo | 1.2, 3.7 |
| `dashboard/` | Next.js + TypeScript adjuster dashboard | 1.2 |
| `claimsense/speech/`, `documents/`, `facts.py` | Whisper, PDF/OCR reading, fact extraction | 1.3 |
| `claimsense/guardrails/` | PII redaction, injection detection, output rules | 1.3, 1.6, 3.5 |
| `claimsense/llm/` | Gemini, OpenAI-compatible (Ollama/vLLM/TGI), router, prompts | 1.3, 2.6 |
| `claimsense/rag/` | Policy chunking, embeddings, Qdrant, coverage assessment | 1.4 |
| `claimsense/agent/`, `tools.py`, `mcp_server/` | LangGraph workflow, read-only tools, MCP server + client | 1.5 |
| `claimsense/pipeline.py` | All steps as plain functions (used by the agent, workers, evals) | 1.5 |
| `evals/` | Gold eval, red-team, fairness audit, CI gate | 1.6, 3.4, 3.5 |
| `claimsense/vision/`, `training/yolo/`, `training/synthetic_images/` | Damage detection, export/benchmark, Stable Diffusion data | 2.1–2.3 |
| `claimsense/fraud/`, `training/fraud/` | Features, scorer, LightGBM training, Spark features, OpenSearch hybrid search | 2.4 |
| `training/extraction_qlora/`, `training/dpo/` | QLoRA fine-tuning, DPO preference tuning | 2.5 |
| `training/serving/` | LLM server load testing | 2.6 |
| `claimsense/streaming/` | Kafka workers with retries, dead-letter topic, idempotency | 3.1 |
| `pipelines/` | Airflow DAG, dbt models | 3.2 |
| `training/batch/` | Ray backlog processing | 3.3 |
| `k8s/`, `docker-compose.yml`, `Dockerfile` | Containers and Kubernetes | 3.6 |
| `.github/workflows/ci.yml` | Lint, tests, eval gate, dashboard build, images | 1.6, 3.5 |

## Quick start

See **[GETTING_STARTED.md](GETTING_STARTED.md)** for the full walkthrough. The short version:

```bash
make setup && source .venv/bin/activate
make data          # generate synthetic data (~2 min)
make test          # tests
make eval          # baseline metrics
make api           # http://localhost:8000/docs
make dashboard     # http://localhost:3000 (second terminal)
```

## Status

| Part | State |
|---|---|
| Synthetic data, document parsing, fact extraction, policy RAG, fraud rules, guardrails, pipeline, evals, CI gate | Written and tested |
| MCP server and client | Written and tested end to end over stdio |
| LangGraph agent, API, storage, dashboard | Written; covered by `tests/test_api_and_agent.py`, run it after installing |
| Gemini / Ollama / vLLM clients, Whisper, OCR, YOLO, Qdrant | Written; need their libraries, models or keys |
| Tier 2–3 scripts (training, Kafka, Airflow, dbt, Ray, k8s) | Starting code; run and adjust in each phase |

Library APIs change often (the MCP SDK renamed its server class in v2 while this was being written). If something errors, check that library's current docs first.

## Design decisions

See [DECISIONS.md](DECISIONS.md).
