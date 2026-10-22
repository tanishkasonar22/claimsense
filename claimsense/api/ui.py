"""A minimal Gradio demo for Hugging Face Spaces (booklet Phase 3.7).

    python -m claimsense.api.ui        # http://localhost:7860

Runs the pipeline in-process (no database, no agent pause) so it works on a free CPU Space.
Set LLM_PROVIDER=gemini and GEMINI_API_KEY as Space secrets for the LLM route.
"""
from __future__ import annotations

import json

import gradio as gr

from ..config import get_settings
from ..pipeline import Context, run_pipeline, submission_from_gold


def _gold_ids(n: int = 25) -> list[str]:
    path = get_settings().synthetic_dir / "gold" / "gold_claims.jsonl"
    return [json.loads(line)["claim_id"] for line in list(path.open())[:n]]


def analyze(gold_id: str):
    path = get_settings().synthetic_dir / "gold" / "gold_claims.jsonl"
    g = next(json.loads(line) for line in path.open() if gold_id in line)
    res, ctx = run_pipeline(submission_from_gold(g), Context.default())
    rec = res.recommendation
    reasons = "\n".join(f"- [{r.kind}] {r.ref}: {r.detail}" for r in rec.reasons)
    return (f"### Recommendation: {rec.action.upper()}\n\n{rec.summary}\n\n**Evidence**\n{reasons}\n\n"
            f"_Correct answer in the gold set: {g['expected']['decision']}_",
            res.transcript.text if res.transcript else "",
            res.facts.model_dump() if res.facts else {},
            res.fraud.model_dump() if res.fraud else {})


with gr.Blocks(title="ClaimSense") as demo:
    gr.Markdown("# ClaimSense\nAI claims analysis with a human in the loop. All data is synthetic.")
    pick = gr.Dropdown(_gold_ids(), label="Sample claim")
    btn = gr.Button("Analyze")
    out = gr.Markdown()
    with gr.Row():
        transcript = gr.Textbox(label="Voice note transcript", lines=6)
        facts = gr.JSON(label="Extracted facts")
        fraud = gr.JSON(label="Fraud assessment")
    btn.click(analyze, pick, [out, transcript, facts, fraud])

if __name__ == "__main__":
    demo.launch()
