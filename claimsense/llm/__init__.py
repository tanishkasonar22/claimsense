"""One interface for every LLM ClaimSense can use.

    llm = get_llm()                      # picks the provider from settings
    facts = llm.structured(SYSTEM, user_text, ClaimFacts)

Providers:
  rules          No LLM. Callers use their deterministic baseline instead (get_llm returns None).
  gemini         Google Gemini API (free tier) - google-genai package
  openai_compat  Any OpenAI-compatible server: Ollama, vLLM, TGI, llama.cpp server
  router         Local model first, Gemini when the local answer fails validation (Phase 2.6)
"""
from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

from ..config import get_settings

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    name: str

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        """Return an instance of `schema`, validated. Raise LLMError if impossible."""

    def text(self, system: str, user: str) -> str:
        """Free-text answer."""


def get_llm() -> LLMClient | None:
    s = get_settings()
    provider = s.llm_provider.lower()
    if provider == "rules":
        return None
    if provider == "gemini":
        from .gemini import GeminiClient
        return GeminiClient(s.gemini_api_key, s.gemini_model)
    if provider == "openai_compat":
        from .openai_compat import OpenAICompatClient
        return OpenAICompatClient(s.openai_compat_url, s.openai_compat_model)
    if provider == "router":
        from .gemini import GeminiClient
        from .openai_compat import OpenAICompatClient
        from .router import RouterClient
        return RouterClient(OpenAICompatClient(s.openai_compat_url, s.openai_compat_model),
                            GeminiClient(s.gemini_api_key, s.gemini_model))
    raise ValueError(f"Unknown LLM_PROVIDER: {s.llm_provider}")
