"""Preference-tune the claim summarizer with DPO (booklet Phase 2.5, study guide B5). GPU required.

    python -m training.dpo.train_dpo --base <your merged SFT model or an instruct model> \
        --pairs data/synthetic/finetune/dpo_pairs.jsonl --out /content/drive/MyDrive/claimsense/dpo-summarizer

Evaluate: generate summaries for 50 held-out claims with the SFT model and the DPO model,
have an LLM judge pick the better one (randomize the order!), then check 30 by hand.
Report the DPO win rate. beta controls how far the model may move from the starting model.
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", default="runs/dpo-summarizer")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=float, default=2.0)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.float16))
    data = load_dataset("json", data_files=args.pairs)["train"].train_test_split(test_size=0.1, seed=0)
    cfg = DPOConfig(output_dir=args.out, beta=args.beta, num_train_epochs=args.epochs, learning_rate=5e-6,
                    per_device_train_batch_size=2, gradient_accumulation_steps=8, logging_steps=10,
                    eval_strategy="steps", eval_steps=50, save_steps=50, save_total_limit=2, report_to=[])
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=data["train"], eval_dataset=data["test"],
                         processing_class=tok,
                         peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    trainer.train()
    trainer.model.save_pretrained(f"{args.out}/adapter")
    m = trainer.evaluate()
    print({k: round(v, 4) for k, v in m.items() if "reward" in k or "loss" in k})


if __name__ == "__main__":
    main()
