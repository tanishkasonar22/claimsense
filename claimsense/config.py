"""All settings in one place. Values come from environment variables or a .env file.

MODE
  lite  Runs on a laptop with no Docker: SQLite, local files, in-memory vector index,
        and (if no API key is set) the offline rules baseline instead of an LLM.
  full  Uses the Docker Compose services: Postgres, MinIO, Qdrant, Kafka, Langfuse.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader so we don't need python-dotenv. Real env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


@dataclass(frozen=True)
class Settings:
    mode: str = field(default_factory=lambda: _env("CLAIMSENSE_MODE", "lite"))

    # Where datasets live. Point this at a Google Drive folder if you keep data in the cloud,
    # for example /content/drive/MyDrive/claimsense/data inside Colab.
    data_dir: Path = field(default_factory=lambda: Path(_env("CLAIMSENSE_DATA_DIR", str(ROOT / "data"))))
    # CarDD can live outside the project, e.g. in Google Drive. Gold photo paths are stored as
    # "cardd://..." and resolved against this folder (see claimsense/vision/__init__.py).
    #   Colab: /content/drive/MyDrive/claimsense/data/public/cardd
    #   Mac (Google Drive for desktop): ~/Library/CloudStorage/GoogleDrive-<you>/My Drive/claimsense/data/public/cardd
    cardd_dir: Path = field(default_factory=lambda: Path(os.path.expanduser(
        _env("CARDD_DIR", str(ROOT / "data" / "public" / "cardd")))))
    model_dir: Path = field(default_factory=lambda: Path(_env("CLAIMSENSE_MODEL_DIR", str(ROOT / "models"))))
    upload_dir: Path = field(default_factory=lambda: Path(_env("CLAIMSENSE_UPLOAD_DIR", str(ROOT / "uploads"))))

    database_url: str = field(default_factory=lambda: _env("DATABASE_URL", f"sqlite:///{ROOT / 'claimsense.db'}"))
    checkpoint_db: str = field(default_factory=lambda: _env("CHECKPOINT_DB", str(ROOT / "checkpoints.sqlite")))

    # Object storage (full mode). Leave S3_ENDPOINT empty to use the local upload_dir.
    s3_endpoint: str = field(default_factory=lambda: _env("S3_ENDPOINT"))
    s3_bucket: str = field(default_factory=lambda: _env("S3_BUCKET", "claimsense"))
    s3_access_key: str = field(default_factory=lambda: _env("S3_ACCESS_KEY", "minioadmin"))
    s3_secret_key: str = field(default_factory=lambda: _env("S3_SECRET_KEY", "minioadmin"))

    # LLM: "rules" (offline baseline), "gemini", or "openai_compat" (Ollama, vLLM, TGI, llama.cpp server)
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "rules"))
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.5-flash"))
    openai_compat_url: str = field(default_factory=lambda: _env("OPENAI_COMPAT_URL", "http://localhost:11434/v1"))
    openai_compat_model: str = field(default_factory=lambda: _env("OPENAI_COMPAT_MODEL", "qwen2.5:3b"))
    # Router (Phase 2.6): try the local model first, fall back to Gemini
    router_enabled: bool = field(default_factory=lambda: _env("ROUTER_ENABLED", "false").lower() == "true")

    # Retrieval
    embedder: str = field(default_factory=lambda: _env("EMBEDDER", "hashing"))  # hashing | sentence-transformers
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL"))           # empty = in-memory index

    # Speech and vision
    whisper_model: str = field(default_factory=lambda: _env("WHISPER_MODEL", "small"))
    vision_model_path: str = field(default_factory=lambda: _env("VISION_MODEL_PATH"))  # .pt or .onnx; empty = fallback

    # Fraud
    fraud_model_path: str = field(default_factory=lambda: _env("FRAUD_MODEL_PATH", str(ROOT / "models" / "fraud_lgbm.txt")))
    fraud_investigate_threshold: float = field(default_factory=lambda: float(_env("FRAUD_THRESHOLD", "0.5")))

    # Streaming and observability (Tier 3 / Phase 1.6)
    kafka_bootstrap: str = field(default_factory=lambda: _env("KAFKA_BOOTSTRAP", "localhost:9092"))
    langfuse_host: str = field(default_factory=lambda: _env("LANGFUSE_HOST"))
    mlflow_tracking_uri: str = field(default_factory=lambda: _env("MLFLOW_TRACKING_URI"))

    @property
    def synthetic_dir(self) -> Path:
        return self.data_dir / "synthetic"

    @property
    def policies_dir(self) -> Path:
        return self.synthetic_dir / "policies"


@lru_cache
def get_settings() -> Settings:
    return Settings()
