"""Load-test LLM servers (booklet Phase 2.6, study guide D4).

    python -m training.serving.benchmark_llm --url http://localhost:11434/v1 --model qwen2.5:3b --concurrency 1 8 32
    python -m training.serving.benchmark_llm --url http://<colab-tunnel>:8000/v1 --model <merged-dir> --concurrency 1 8 32

Works with Ollama, llama.cpp server, vLLM and TGI (all OpenAI-compatible). Measures time to
first token (streaming), tokens per second, p95 latency at each concurrency level, and
extraction accuracy on a sample of fine-tuning validation prompts.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx

from claimsense.config import get_settings


def load_prompts(n: int) -> list[tuple[list[dict], str]]:
    path = get_settings().synthetic_dir / "finetune" / "sft_val.jsonl"
    rows = [json.loads(line) for line in path.open()][:n]
    return [(r["messages"][:2], r["messages"][2]["content"]) for r in rows]


async def one(client: httpx.AsyncClient, url: str, model: str, messages: list[dict]) -> dict:
    body = {"model": model, "messages": messages, "temperature": 0, "stream": True, "max_tokens": 300}
    t0, first, text, tokens = time.perf_counter(), None, "", 0
    async with client.stream("POST", f"{url}/chat/completions", json=body, timeout=300) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
            if delta and first is None:
                first = time.perf_counter() - t0
            text += delta
            tokens += 1
    total = time.perf_counter() - t0
    return {"ttft": first or total, "total": total, "tokens": tokens, "text": text}


async def run_level(url: str, model: str, prompts, concurrency: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        async def task(p):
            async with sem:
                return await one(client, url, model, p[0])
        t0 = time.perf_counter()
        results = await asyncio.gather(*(task(p) for p in prompts))
        wall = time.perf_counter() - t0
    correct = 0
    for (_, gold), r in zip(prompts, results):
        try:
            correct += json.loads(r["text"].strip().strip("`").removeprefix("json")) == json.loads(gold)
        except json.JSONDecodeError:
            pass
    totals = sorted(r["total"] for r in results)
    return {"concurrency": concurrency, "requests": len(results),
            "ttft_median_s": round(statistics.median(r["ttft"] for r in results), 3),
            "latency_p95_s": round(totals[int(0.95 * (len(totals) - 1))], 3),
            "tokens_per_s_total": round(sum(r["tokens"] for r in results) / wall, 1),
            "exact_json_match": round(correct / len(results), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    prompts = load_prompts(args.n)
    for c in args.concurrency:
        print(json.dumps(asyncio.run(run_level(args.url.rstrip("/"), args.model, prompts, c))))


if __name__ == "__main__":
    main()
