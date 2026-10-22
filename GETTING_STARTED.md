# Getting started with ClaimSense

Follow these steps in order. Each step ends with a check, so you always know it worked. Steps 1–6 take one afternoon. After that, you follow the booklet one phase at a time.

## Step 1. Install the tools (about 1 hour)

Install on your laptop:

1. **Python 3.11 or 3.12**: python.org. Check: `python --version`
2. **Git**: git-scm.com. Check: `git --version`
3. **VS Code**: with the Python extension
4. **Node.js 22 (LTS)**: nodejs.org, for the dashboard. Check: `node --version`
5. **Docker Desktop**: not needed until Step 8. Give it 8 GB of memory in its settings.

Windows users: use **WSL2 (Ubuntu)** for everything. The commands below assume a Mac, Linux or WSL terminal.

## Step 2. Put the project on GitHub (15 minutes)

1. Unzip `claimsense.zip` into a folder, for example `~/projects/claimsense`.
2. Create an empty **public** repository on GitHub called `claimsense` (GitHub Actions is free for public repos).
3. In a terminal:

```bash
cd ~/projects/claimsense
git init -b main
git add .
git commit -m "ClaimSense: initial project"
git remote add origin https://github.com/<your-username>/claimsense.git
git push -u origin main
```

Check: your repo shows the code on GitHub, and **no** `data/synthetic`, `.env` or `models` folders (the `.gitignore` keeps them out).

## Step 3. Create the Python environment (10 minutes)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -U pip
pip install -e ".[api,llm,fraud,dev]"
cp .env.example .env
```

Check: `python -c "import claimsense, fastapi, langgraph; print('ok')"` prints `ok`.

Every time you open a new terminal, run `source .venv/bin/activate` first.

## Step 4. Generate the synthetic data (2 minutes)

```bash
python generate_all.py
python scripts/validate_data.py
```

Check: the validator ends with `All checks passed.` You now have 2,000 customers, 20,000 past claims, 200 gold claims with PDFs, voice scripts, a red-team set and fine-tuning data in `data/synthetic/`.

**Keeping data off your laptop:** the same command recreates identical data anywhere, so you never need to back it up. CarDD images and trained models go in Google Drive; set `CLAIMSENSE_DATA_DIR` in `.env` to point at it.

## Step 5. Run the tests and your first evaluation (5 minutes)

```bash
pytest -q
python -m evals.run_gold
python -m evals.run_redteam
python -m evals.fairness_audit
```

Check: tests pass, and the evaluation prints a table (decision agreement around 0.9). **Copy these numbers into `DECISIONS.md`.** They're your baseline, and every later phase has to beat them.

## Step 6. Run the app (10 minutes)

Terminal 1:
```bash
source .venv/bin/activate
uvicorn claimsense.api.main:app --reload --port 8000
```
Open http://localhost:8000/docs. Try `POST /demo/gold/{gold_id}` with `G000005`, then `GET /claims`.

Terminal 2:
```bash
cd dashboard
npm install
npm run dev
```
Open http://localhost:3000, submit a sample claim, open it, and approve or deny it.

Check: the claim moves from `processing` to `ready_for_review` to `decided`, and the audit trail shows every step.

## Step 7. Turn on a real LLM (15 minutes)

1. Get a free API key at **aistudio.google.com** (Get API key).
2. In `.env`: `LLM_PROVIDER=gemini` and `GEMINI_API_KEY=<your key>`.
3. Run `python -m evals.run_gold --limit 20`.

Compare with the rules baseline. Free-tier rate limits apply, so use `--limit` and expect it to be slower. Never commit your key.

Optional local model: install **Ollama**, run `ollama pull qwen2.5:3b`, then set `LLM_PROVIDER=openai_compat`.

## Step 8. Start the full stack with Docker (when you reach Phase 1.2–1.4)

```bash
docker compose up -d --build
docker compose ps
```
Then set `DATABASE_URL`, `S3_ENDPOINT` and `QDRANT_URL` in `.env` (see `.env.example`) when you want your local runs to use the services.

## Step 9. Download the real datasets

1. **CarDD**: fill in and email the licence form from cardd-ustc.github.io. When the link arrives, put it in Google Drive and a 200-image sample in `data/public/cardd/`.
   While you wait, you can start with the Hugging Face mirror (`mouen5/CarDD`, about 2.2 GB, FiftyOne format). Download and convert it in Colab; see the instructions at the top of `training/yolo/prepare_cardd_hf.py`. It produces the same folder layout as the official download.
   Then, either way:
   ```bash
   python -m training.yolo.prepare_cardd --cardd data/public/cardd
   python scripts/assemble_gold_with_cardd.py --cardd data/public/cardd
   ```
2. **Kaggle vehicle fraud**: download from kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection into `data/public/kaggle_vehicle_fraud/`. Use it to check that your fraud features behave sensibly on real-world-shaped data.
3. **Voice notes**: record 20–30 of the scripts in `data/synthetic/voice/voice_scripts.jsonl` as `data/synthetic/voice/<claim_id>.wav`, then `pip install faster-whisper`. The eval will start reporting real speech results.

## Step 10. Follow the booklet, one phase at a time

| Week | Do this | Code to start from |
|---|---|---|
| 1–2 | Phase 1.1: read `generators/`, understand every field; hand-check 30 gold claims (set `hand_checked: true`) | `generators/`, `scripts/validate_data.py` |
| 3 | Phase 1.2: API + dashboard; add a real upload form | `claimsense/api/`, `dashboard/` |
| 4–5 | Phase 1.3: Whisper on your recordings, OCR on `*_photo.png` documents, Gemini extraction | `claimsense/speech/`, `documents/`, `facts.py` |
| 6 | Phase 1.4: switch to sentence-transformers + Qdrant; measure retrieval recall | `claimsense/rag/` |
| 7–8 | Phase 1.5: agent, MCP in Claude Desktop, human approval | `claimsense/agent/`, `mcp_server/` |
| 9 | Phase 1.6: Langfuse tracing, CI gate on GitHub | `tracing.py`, `.github/workflows/ci.yml` |
| 10–17 | Tier 2 | `training/` |
| 18–24 | Tier 3 | `claimsense/streaming/`, `pipelines/`, `k8s/` |

**How to work each phase:**
1. Read the study guide section for the phase.
2. Read the code for the phase. Run it. Break it on purpose and see what happens.
3. Improve one thing, run `python -m evals.run_gold`, and write the before/after numbers in `DECISIONS.md`.
4. Commit on a branch, open a pull request, and let CI check it.
5. Ask Claude Code to explain anything you don't understand. You must be able to explain every line in an interview.

## When something breaks

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | Activate the venv; install the extra for that phase, e.g. `pip install -e ".[speech]"` |
| `synthetic data missing` | Run `python generate_all.py` |
| Gemini 429 errors | Free-tier rate limit: use `--limit`, wait, or use the rules/local route |
| Claim stuck in `processing` | Check the API terminal for the traceback; `GET /claims/{id}` shows the error |
| `MCPServer`/`FastMCP` import error | MCP SDK version change; the server handles both, so upgrade with `pip install -U mcp` |
| Dashboard can't reach API | API must run on port 8000; check `NEXT_PUBLIC_API_URL` |
