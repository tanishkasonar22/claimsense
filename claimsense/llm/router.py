"""Model router (booklet Phase 2.6).

Try the cheap local model first. If its output fails validation, or a caller-supplied
check says the answer is not good enough, send the request to the strong model.
Counts how often each route is used so you can report "X% of claims stayed local".
"""
from __future__ import annotations

from collections import Counter
from typing import Callable

from . import LLMClient, LLMError, T


class RouterClient:
    def __init__(self, local: LLMClient, strong: LLMClient,
                 accept: Callable[[object], bool] | None = None):
        self.local, self.strong = local, strong
        self.accept = accept or (lambda _: True)
        self.name = f"router({local.name}->{strong.name})"
        self.routes: Counter[str] = Counter()

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        try:
            out = self.local.structured(system, user, schema)
            if self.accept(out):
                self.routes["local"] += 1
                return out
        except LLMError:
            pass
        self.routes["strong"] += 1
        return self.strong.structured(system, user, schema)

    def text(self, system: str, user: str) -> str:
        try:
            out = self.local.text(system, user)
            if out.strip():
                self.routes["local"] += 1
                return out
        except LLMError:
            pass
        self.routes["strong"] += 1
        return self.strong.text(system, user)

    def local_share(self) -> float:
        total = sum(self.routes.values())
        return self.routes["local"] / total if total else 0.0
