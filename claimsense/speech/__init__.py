"""Voice note -> Transcript (booklet Phase 1.3, study guide B6)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from ..config import get_settings
from ..schemas import Transcript


@lru_cache
def _whisper_model(size: str):
    from faster_whisper import WhisperModel  # pip install faster-whisper
    return WhisperModel(size, device="auto", compute_type="int8")


def transcribe(audio_path: str | Path | None, claim_id: str | None = None) -> Transcript:
    """Transcribe with faster-whisper. If the audio file or the library is missing, fall back
    to the voice script (the exact words) so the rest of the pipeline can still run.
    The fallback is clearly marked in `source` so evaluation never mistakes it for Whisper output.
    """
    s = get_settings()
    if audio_path:
        p = Path(audio_path)
        if not p.is_absolute():
            p = s.synthetic_dir / p
        if p.exists():
            try:
                model = _whisper_model(s.whisper_model)
                segments, info = model.transcribe(str(p), vad_filter=True, beam_size=5)
                segs = [{"start": round(x.start, 2), "end": round(x.end, 2), "text": x.text.strip()} for x in segments]
                return Transcript(text=" ".join(x["text"] for x in segs), language=info.language,
                                  source="whisper", segments=segs)
            except ImportError:
                pass
    if claim_id:
        script = voice_script(claim_id)
        if script:
            return Transcript(text=script, language="en", source="script_fallback")
    return Transcript(text="", source="none")


@lru_cache
def _scripts() -> dict[str, str]:
    path = get_settings().synthetic_dir / "voice" / "voice_scripts.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.open() if line.strip()]
    return {r["claim_id"]: r["script"] for r in rows}


def voice_script(claim_id: str) -> str | None:
    return _scripts().get(claim_id)
