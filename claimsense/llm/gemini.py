"""Google Gemini via the google-genai SDK (free tier from Google AI Studio).

Check the current model names and free-tier limits in the Gemini API docs; they change.
"""
from __future__ import annotations

import time

from . import LLMError, T
from ._json import is_parse_error, parse_json_model, schema_instruction


class GeminiClient:
    name = "gemini"

    def __init__(self, api_key: str, model: str, max_retries: int = 4):
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set. Get a free key at aistudio.google.com.")
        from google import genai  # imported here so the package works without it installed
        from google.genai import types
        self._types = types
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_retries = max_retries

    def _call(self, system: str, user: str, json_mode: bool) -> str:
        cfg = self._types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                resp = self.client.models.generate_content(model=self.model, contents=user, config=cfg)
                return resp.text or ""
            except Exception as e:  # rate limits (429) and transient errors: back off and retry
                if attempt == self.max_retries - 1:
                    raise LLMError(f"Gemini call failed: {e}") from e
                time.sleep(delay)
                delay *= 2
        raise LLMError("unreachable")

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        sys_prompt = f"{system}\n\n{schema_instruction(schema)}"
        last: Exception | None = None
        for _ in range(2):  # one retry with the error message fed back
            raw = self._call(sys_prompt, user, json_mode=True)
            try:
                return parse_json_model(raw, schema)  # type: ignore[return-value]
            except Exception as e:
                if not is_parse_error(e):
                    raise
                last = e
                user = f"{user}\n\nYour previous answer was invalid: {e}. Return valid JSON only."
        raise LLMError(f"Gemini returned invalid JSON twice: {last}")

    def text(self, system: str, user: str) -> str:
        return self._call(system, user, json_mode=False)

