# Design decisions

For every decision: what I chose, what else I considered, why, and the evidence. Add a row each phase. This file becomes the design doc and the answers to "why did you…" interview questions.

## Results log

| Date | Change | Metric | Before | After | Notes |
|---|---|---|---|---|---|
| | Baseline (rules, no LLM) | decision agreement | – | 0.895 | 200 gold claims, fallbacks for photos/voice |
| | | fraud PR-AUC | – | 0.93 | rules scorer |
| | | coverage correct | – | 0.96 | misses = hidden exclusions |
| | | | | | |

## Decisions made so far

**One Python package, many entry points (not one repo per service).**
Considered: separate folders with separate dependencies per service. Chose one package with shared schemas so services can't drift apart; each service is a different command in the same Docker image. Revisit if services need very different dependencies.

**Pipeline steps are plain functions; LangGraph only wires them.**
Considered: putting logic inside graph nodes. Plain functions are reused by the agent, Kafka workers, evaluation and tests, and run in milliseconds without a server.

**Mostly fixed workflow, LLM inside steps.**
Considered: a free-form tool-calling agent choosing every step. A fixed route is predictable, testable and auditable, which matters for regulated decisions. The LLM is used where language understanding helps (extraction, coverage reading, summaries).

**Rules baselines everywhere.**
Every model (extraction, coverage, fraud, injection) has a deterministic baseline. It makes the system run with no API key, and it gives every improvement a number to beat.

**Chunk policies by section.**
Considered: fixed-size chunks. Section chunks make every retrieved chunk a citable section (§3.1), so citations can be checked mechanically.

**Only cite retrieved sections.**
The coverage step drops any section the model cites that wasn't retrieved, so the model can't invent policy text.

**The AI never approves.**
`needs_human` is always true, approval is blocked when injection is suspected, and summaries are scrubbed of "the claim is approved" phrasing. Learned from real industry controversies about automated claim decisions.

**Point-in-time fraud features and a time-based split.**
History statistics use only earlier claims; training uses older claims and testing newer ones. Random splits would leak the future and inflate results.

**Protected attributes are never model inputs.**
They're kept in synthetic data only to audit outcomes (Phase 3.4).

## Open questions

- Which embedding model gives the best retrieval recall on the policies? (Phase 1.4)
- Does Stable Diffusion data help the rarest damage class on real test images? (Phase 2.2)
- What share of claims can the fine-tuned local model handle without falling back to Gemini? (Phase 2.6)
