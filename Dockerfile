# One image for the API, the MCP server and every Kafka worker (different commands).
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# tesseract = OCR for document photos; ffmpeg = audio decoding for Whisper; libgl/glib = OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Copy dependency metadata first so this layer is cached when only code changes.
COPY pyproject.toml README.md ./
RUN mkdir -p claimsense && touch claimsense/__init__.py \
    && pip install ".[api,llm,docs,speech,fraud,full]" \
    && rm -rf claimsense
COPY claimsense ./claimsense
COPY evals ./evals
COPY generators ./generators
COPY scripts ./scripts
COPY generate_all.py ./
RUN pip install --no-deps .

RUN useradd --create-home app && chown -R app /app
USER app
EXPOSE 8000
CMD ["uvicorn", "claimsense.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
