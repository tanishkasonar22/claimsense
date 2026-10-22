"""File storage for photos, audio and documents.

  LocalFileStore  files under CLAIMSENSE_UPLOAD_DIR (lite mode)
  S3FileStore     MinIO or any S3-compatible store (full mode; pip install boto3)

The database stores only the returned key (e.g. "claims/CL123/photo_0.jpg"), never file bytes.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from ..config import get_settings

SAFE = re.compile(r"[^A-Za-z0-9._-]")
ALLOWED = {".jpg", ".jpeg", ".png", ".webp", ".pdf", ".wav", ".mp3", ".m4a", ".ogg", ".webm"}
MAX_BYTES = 25 * 1024 * 1024


class FileRejected(ValueError):
    pass


def _key(claim_id: str, filename: str, kind: str, i: int) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED:
        raise FileRejected(f"file type {ext or '(none)'} not allowed")
    return f"claims/{SAFE.sub('_', claim_id)}/{kind}_{i}{ext}"


class LocalFileStore:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def save(self, claim_id: str, filename: str, data: bytes, kind: str, i: int = 0) -> str:
        if len(data) > MAX_BYTES:
            raise FileRejected("file too large")
        key = _key(claim_id, filename, kind, i)
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)          # local path, readable by the services directly

    def open_path(self, key: str) -> Path:
        return Path(key)


class S3FileStore:  # pragma: no cover - needs MinIO
    def __init__(self, endpoint: str, bucket: str, access_key: str, secret_key: str, cache: Path):
        import boto3
        self.s3 = boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=access_key,
                               aws_secret_access_key=secret_key)
        self.bucket, self.cache = bucket, cache
        try:
            self.s3.head_bucket(Bucket=bucket)
        except Exception:
            self.s3.create_bucket(Bucket=bucket)

    def save(self, claim_id: str, filename: str, data: bytes, kind: str, i: int = 0) -> str:
        if len(data) > MAX_BYTES:
            raise FileRejected("file too large")
        key = _key(claim_id, filename, kind, i)
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data)
        return f"s3://{self.bucket}/{key}"

    def open_path(self, key: str) -> Path:
        """Download to a local cache so OCR/Whisper/YOLO can read a normal file path."""
        key = key.removeprefix(f"s3://{self.bucket}/")
        local = self.cache / key
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(self.bucket, key, str(local))
        return local


@lru_cache
def get_file_store():
    s = get_settings()
    if s.s3_endpoint:
        return S3FileStore(s.s3_endpoint, s.s3_bucket, s.s3_access_key, s.s3_secret_key, s.upload_dir / "_cache")
    return LocalFileStore(s.upload_dir)
