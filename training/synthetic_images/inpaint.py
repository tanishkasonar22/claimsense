"""Generate synthetic damage images with Stable Diffusion inpainting (booklet Phase 2.2).

    python -m training.synthetic_images.inpaint --clean-dir data/public/clean_cars \
        --damage glass_shatter --n 300 --out data/synthetic/images/glass_shatter

Run in Colab with a GPU. For each clean car photo: pick a region, paint the damage into it,
and write a YOLO label for that region (we know where we painted, so the label is free).

Then run the ablation: train A on real only, B on real + synthetic, same settings,
and evaluate BOTH on the real test set only. Report whether it helped, even if it didn't.
Check the diffusers docs for the current recommended inpainting model id.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

PROMPTS = {
    "dent": "close-up photo of a car body panel with a deep dent, realistic, daylight",
    "scratch": "close-up photo of long scratches in car paint, realistic, daylight",
    "crack": "close-up photo of a cracked plastic car bumper, realistic",
    "glass_shatter": "photo of a shattered car window with spiderweb cracks, realistic",
    "lamp_broken": "photo of a broken car headlight with cracked lens, realistic",
    "tire_flat": "photo of a flat car tire, realistic",
}
CLASSES = ["dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-dir", required=True)
    ap.add_argument("--damage", choices=CLASSES, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-inpainting")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from diffusers import AutoPipelineForInpainting
    from PIL import Image, ImageDraw

    pipe = AutoPipelineForInpainting.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda")
    rng = random.Random(args.seed)
    clean = sorted(Path(args.clean_dir).glob("*.jpg"))
    out_img, out_lbl = Path(args.out) / "images", Path(args.out) / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    cls = CLASSES.index(args.damage)
    for i in range(args.n):
        src = Image.open(rng.choice(clean)).convert("RGB").resize((512, 512))
        w, h = rng.randint(90, 220), rng.randint(70, 180)
        x, y = rng.randint(40, 512 - w - 40), rng.randint(120, 512 - h - 40)
        mask = Image.new("L", (512, 512), 0)
        ImageDraw.Draw(mask).ellipse([x, y, x + w, y + h], fill=255)
        g = torch.Generator("cuda").manual_seed(args.seed + i)
        img = pipe(prompt=PROMPTS[args.damage], image=src, mask_image=mask, generator=g,
                   num_inference_steps=30, guidance_scale=7.5).images[0]
        name = f"syn_{args.damage}_{i:05d}"
        img.save(out_img / f"{name}.jpg", quality=92)
        cx, cy = (x + w / 2) / 512, (y + h / 2) / 512
        # bounding box label; for segmentation, convert the ellipse to a polygon instead
        (out_lbl / f"{name}.txt").write_text(f"{cls} {cx:.5f} {cy:.5f} {w / 512:.5f} {h / 512:.5f}\n")
        if i % 25 == 0:
            print(f"{i}/{args.n}")
    print(f"done: {args.n} images in {out_img}. Look at a sample before training on them!")


if __name__ == "__main__":
    main()
