"""Helpers for getting valid JSON out of a model."""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError

FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json_model(raw: str, schema: type[BaseModel]) -> BaseModel:
    """Strip markdown fences, find the outermost JSON object, validate it."""
    text = FENCE.sub("", raw.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    data = json.loads(text[start:end + 1])
    return schema.model_validate(data)


def schema_instruction(schema: type[BaseModel]) -> str:
    return ("Return ONLY one JSON object, no markdown, matching this JSON schema:\n"
            + json.dumps(schema.model_json_schema()))


def is_parse_error(e: Exception) -> bool:
    return isinstance(e, (ValueError, ValidationError, json.JSONDecodeError))
