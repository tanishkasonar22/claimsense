"""Fine-tune a small open model to extract ClaimFacts (booklet Phase 2.5, study guide B4).
Run in Colab/Kaggle with a GPU.

    pip install -U transformers peft trl bitsandbytes accelerate datasets
    python -m training.extraction_qlora.train_qlora \
        --base Qwen/Qwen2.5-1.5B-Instruct \
        --train data/synthetic/finetune/sft_train.jsonl --val data/synthetic/finetune/sft_val.jsonl \
        --out /content/drive/MyDrive/claimsense/qlora-extractor

Data: chat-formatted examples built ONLY from history claims (generators/finetune.py), never
from gold claims. The system prompt matches claimsense/llm/prompts.py FINETUNE_SYSTEM.

After training:
  1. merge:  python -m training.extraction_qlora.train_qlora --merge-only --out <dir>
  2. serve:  vLLM:   vllm serve <dir>/merged --port 8000
             Ollama: convert to GGUF with llama.cpp's convert_hf_to_gguf.py, then `ollama create`
  3. eval:   LLM_PROVIDER=openai_compat OPENAI_COMPAT_URL=http://localhost:8000/v1 \
             OPENAI_COMPAT_MODEL=<dir>/merged python -m evals.run_gold
             and compare with the base model and Gemini (booklet table in Phase 2.5).
Check model licenses on Hugging Face before choosing a base model; library arguments change
between versions, so check the current TRL/PEFT docs if something errors.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train", default="data/synthetic/finetune/sft_train.jsonl")
    ap.add_argument("--val", default="data/synthetic/finetune/sft_val.jsonl")
    ap.add_argument("--out", default="runs/qlora-extractor")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    out = Path(args.out)
    if args.merge_only:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16, device_map="auto")
        merged = PeftModel.from_pretrained(base, out / "adapter").merge_and_unload()
        merged.save_pretrained(out / "merged")
        AutoTokenizer.from_pretrained(args.base).save_pretrained(out / "merged")
        print(f"merged model saved to {out / 'merged'}")
        return

    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(args.base)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb, device_map="auto")
    data = load_dataset("json", data_files={"train": args.train, "validation": args.val})

    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(
        output_dir=str(out / "checkpoints"), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=4, gradient_accumulation_steps=4, gradient_checkpointing=True,
        lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=20,
        eval_strategy="steps", eval_steps=100, save_steps=100, save_total_limit=3,
        max_length=2048, assistant_only_loss=True,     # learn only the answer, not the prompt
        report_to=["wandb"] if __import__("os").environ.get("WANDB_API_KEY") else [],
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=data["train"], eval_dataset=data["validation"],
                         processing_class=tok, peft_config=lora)
    trainer.train(resume_from_checkpoint=any((out / "checkpoints").glob("checkpoint-*")) or None)
    trainer.model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    print(f"adapter saved to {out / 'adapter'}; validation loss: {trainer.evaluate()['eval_loss']:.4f}")


if __name__ == "__main__":
    main()
