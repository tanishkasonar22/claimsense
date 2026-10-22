# Shortcuts. Run `make help` to list them. (Windows: use the commands inside each target directly.)
.PHONY: help setup data validate test eval redteam fairness gate api dashboard mcp up down fraud

help:
	@grep -E '^[a-z]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-10s %s\n", $$1, $$2}'

setup: ## create .venv and install the Tier 1 libraries
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e ".[api,llm,fraud,dev]"
	cp -n .env.example .env || true

data: ## generate all synthetic data (about 2 minutes)
	python generate_all.py

validate: ## check the planted signals in the synthetic data
	python scripts/validate_data.py

test: ## unit + integration tests
	pytest -q

eval: ## run the pipeline on the gold set and print every metric
	python -m evals.run_gold

redteam: ## run the red-team attacks
	python -m evals.run_redteam

fairness: ## fairness audit of historical decisions
	python -m evals.fairness_audit

gate: ## the CI evaluation gate
	python -m evals.ci_gate --limit 40

fraud: ## train the LightGBM fraud model (Tier 2)
	python -m training.fraud.train_fraud

api: ## start the API on http://localhost:8000 (docs at /docs)
	uvicorn claimsense.api.main:app --reload --port 8000

dashboard: ## start the dashboard on http://localhost:3000
	cd dashboard && npm install && npm run dev

mcp: ## open the MCP Inspector against the ClaimSense MCP server
	npx @modelcontextprotocol/inspector python -m claimsense.mcp_server.server

up: ## start the Docker Compose core stack
	docker compose up -d --build

down: ## stop the Docker Compose stack
	docker compose down
