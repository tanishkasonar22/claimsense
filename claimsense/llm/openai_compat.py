"""Any server that speaks the OpenAI chat-completions API.

Works with:
  Ollama      OPENAI_COMPAT_URL=http://localhost:11434/v1   OPENAI_COMPAT_MODEL=qwen2.5:3b
  vLLM        OPENAI_COMPAT_URL=http://<host>:8000/v1       OPENAI_COMPAT_MODEL=<hf model id>
  TGI         OPENAI_COMPAT_URL=http://<host>:8080/v1       OPENAI_COMPAT_MODEL=tgi
  llama.cpp   OPENAI_COMPAT_URL=http://localhost:8080/v1

Uses plain HTTP (httpx) so you can see exactly what is sent.
"""
from __future__ import annotations

import httpx

from . import LLMError, T
from ._json import is_parse_error, parse_json_model, schema_instruction


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, api_key: str = "not-needed", timeout: float = 120):
        self.name = f"openai_compat:{model}"
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout = timeout
        self.last_usage: dict = {}

    def _call(self, system: str, user: str, json_mode: bool) -> str:
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            r = httpx.post(self.url, json=body, headers=self.headers, timeout=self.timeout)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"{self.name} request failed: {e}") from e
        data = r.json()
        self.last_usage = data.get("usage", {})
        return data["choices"][0]["message"]["content"] or ""

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        sys_prompt = f"{system}\n\n{schema_instruction(schema)}"
        raw = self._call(sys_prompt, user, json_mode=True)
        try:
            return parse_json_model(raw, schema)  # type: ignore[return-value]
        except Exception as e:
            if is_parse_error(e):
                raise LLMError(f"{self.name} returned invalid JSON: {e}") from e
            raise

    def text(self, system: str, user: str) -> str:
        return self._call(system, user, json_mode=False)
