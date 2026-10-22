#!/usr/bin/env bash
# One-command setup for ClaimSense (Mac / Linux).
#   cd ~/projects/claimsense && bash setup.sh
set -e
cd "$(dirname "$0")"

echo "== 1/5 Checking the project files"
for f in generate_all.py pyproject.toml claimsense/pipeline.py generators/claims.py; do
  [ -e "$f" ] || { echo "Missing $f. Unzip claimsense-project.zip again into ~/projects."; exit 1; }
done

echo "== 2/5 Finding Python 3.11 or newer"
PY=""
for c in python3.12 python3.11 python3.13 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
    PY="$c"; break
  fi
done
[ -n "$PY" ] || { echo "Need Python 3.11+. Run: pyenv install 3.12.7 && pyenv local 3.12.7, then run this again."; exit 1; }
echo "   using $($PY --version)"

echo "== 3/5 Creating .venv and installing libraries (a few minutes)"
rm -rf .venv
"$PY" -m venv .venv
.venv/bin/python -m pip install -q -U pip
.venv/bin/python -m pip install -q -e ".[api,llm,fraud,dev]"
[ -f .env ] || cp .env.example .env

echo "== 4/5 Generating synthetic data (about 2 minutes)"
.venv/bin/python generate_all.py > /dev/null
.venv/bin/python scripts/validate_data.py | tail -1

echo "== 5/5 Running tests and the first evaluation"
.venv/bin/python -m pytest -q
.venv/bin/python -m evals.run_gold | sed -n '2,20p'

echo
echo "Done. In every new terminal run:"
echo "   cd $(pwd) && source .venv/bin/activate"
