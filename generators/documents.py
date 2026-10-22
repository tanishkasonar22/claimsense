"""Synthetic claim documents: repair estimates and police reports.

For every document we write three files:
  <id>.pdf          a clean, digitally generated PDF
  <id>_photo.png    the same document as a "phone photo": slightly rotated, blurred, uneven light
  <id>.json         the ground-truth fields (what a perfect extractor should return)

Some PNGs are deliberately low quality so the OCR + cleanup step (Phase 1.3) has something
to fix, and so you can measure how much OpenCV cleanup helps.
"""
from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from reportlab.lib.colors import white
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from .common import SYNTH, Settings, make_faker, make_rng

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def _font(size: int):
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


# ---------------------------------------------------------------- field builders
def repair_estimate_fields(claim: dict, fk, rng: random.Random) -> dict:
    lines = claim["estimate_lines"]
    parts = round(sum(l["parts_cost"] for l in lines), 2)
    labor = round(sum(l["labor_cost"] for l in lines), 2)
    tax = round((parts + labor) * 0.0825, 2)
    est_date = dt.date.fromisoformat(str(claim["report_date"])) + dt.timedelta(days=rng.randint(0, 4))
    total = round(parts + labor + tax, 2)
    # For inflated-amount fraud, the shop estimate can itself be padded.
    if "inflated_amount" in claim["fraud_patterns"] and rng.random() < 0.5:
        total = claim["claim_amount"]
    return {
        "doc_type": "repair_estimate",
        "claim_id": claim["claim_id"],
        "shop_name": claim["repair_shop"],
        "shop_address": fk.street_address() + ", " + fk.city(),
        "shop_phone": fk.numerify("555-###-####"),
        "estimate_number": fk.bothify("EST-#####"),
        "estimate_date": est_date.isoformat(),
        "customer_name": claim["customer_name"],
        "vehicle": f"{claim['vehicle_year']} {claim['vehicle_make']} {claim['vehicle_model']}",
        "vin": claim["vin"],
        "line_items": [{k: l[k] for k in ("part", "operation", "parts_cost", "labor_hours", "labor_cost")} for l in lines],
        "subtotal_parts": parts, "subtotal_labor": labor, "tax": tax, "total": total,
    }


def police_report_fields(claim: dict, fk, rng: random.Random) -> dict:
    other = None
    if claim["incident_type"] in ("rear_end_collision", "front_collision", "side_swipe"):
        other = {"driver": fk.name(), "plate": fk.bothify("???-####").upper(),
                 "insurer": rng.choice(["Northwind Mutual", "Bluebird Insurance", "Summit Auto Group"])}
    return {
        "doc_type": "police_report",
        "claim_id": claim["claim_id"],
        "report_number": fk.bothify("PR-2026-######"),
        "incident_date": str(claim["incident_date"]),
        "incident_time": f"{rng.randint(6, 23):02d}:{rng.choice(['05', '15', '20', '40', '55'])}",
        "location": claim["narrative_location"],
        "officer_name": "Officer " + fk.last_name(),
        "badge_number": fk.numerify("####"),
        "narrative": claim["officer_narrative"],
        "insured_vehicle": f"{claim['vehicle_year']} {claim['vehicle_make']} {claim['vehicle_model']}",
        "other_party": other,
    }


# ---------------------------------------------------------------- text layout (shared by PDF and PNG)
def estimate_text(f: dict) -> list[tuple[str, str]]:
    """List of (style, text) lines. style: title | head | body | small."""
    L = [("title", f["shop_name"]), ("small", f"{f['shop_address']}  |  {f['shop_phone']}"), ("body", ""),
         ("head", "REPAIR ESTIMATE"),
         ("body", f"Estimate #: {f['estimate_number']}        Date: {f['estimate_date']}"),
         ("body", f"Customer: {f['customer_name']}"),
         ("body", f"Vehicle: {f['vehicle']}"), ("body", f"VIN: {f['vin']}"), ("body", ""),
         ("head", "Part                     Operation   Parts $    Labor h   Labor $")]
    for l in f["line_items"]:
        L.append(("body", f"{l['part'].replace('_', ' '):<25}{l['operation']:<12}{l['parts_cost']:>8.2f}   {l['labor_hours']:>6.1f}   {l['labor_cost']:>8.2f}"))
    L += [("body", ""),
          ("body", f"Parts subtotal: ${f['subtotal_parts']:,.2f}"),
          ("body", f"Labor subtotal: ${f['subtotal_labor']:,.2f}"),
          ("body", f"Tax (8.25%): ${f['tax']:,.2f}"),
          ("head", f"TOTAL: ${f['total']:,.2f}"), ("body", ""),
          ("small", "Estimate valid for 30 days. Hidden damage may increase final cost.")]
    return L


def police_text(f: dict) -> list[tuple[str, str]]:
    L = [("title", "Metro City Police Department"), ("head", "TRAFFIC INCIDENT REPORT"), ("body", ""),
         ("body", f"Report #: {f['report_number']}"),
         ("body", f"Date: {f['incident_date']}    Time: {f['incident_time']}"),
         ("body", f"Location: {f['location']}"),
         ("body", f"Reporting officer: {f['officer_name']}  (Badge {f['badge_number']})"),
         ("body", f"Insured vehicle: {f['insured_vehicle']}")]
    if f["other_party"]:
        o = f["other_party"]
        L.append(("body", f"Other party: {o['driver']}, plate {o['plate']}, insured by {o['insurer']}"))
    L += [("body", ""), ("head", "Officer narrative")]
    words, cur = f["narrative"].split(), ""
    for w in words:  # simple word wrap
        if len(cur) + len(w) > 78:
            L.append(("body", cur))
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        L.append(("body", cur))
    return L


# ---------------------------------------------------------------- renderers
PDF_STYLE = {"title": ("Helvetica-Bold", 16), "head": ("Courier-Bold", 10.5),
             "body": ("Courier", 10), "small": ("Helvetica-Oblique", 8.5)}


def write_pdf(path: Path, lines: list[tuple[str, str]], hidden_text: str | None = None) -> None:
    c = canvas.Canvas(str(path), pagesize=letter)
    y = 740
    for style, text in lines:
        font, size = PDF_STYLE[style]
        c.setFont(font, size)
        c.drawString(54, y, text)
        y -= size + 6
    if hidden_text:  # prompt-injection test: white, tiny text a human won't see but OCR/LLMs may
        c.setFillColor(white)
        c.setFont("Helvetica", 4)
        c.drawString(54, 60, hidden_text)
    c.save()


def write_photo_png(path: Path, lines: list[tuple[str, str]], rng: random.Random,
                    quality: str, hidden_text: str | None = None) -> None:
    W, H = 1275, 1650  # letter page at 150 dpi
    img = Image.new("RGB", (W, H), (250, 249, 245))
    d = ImageDraw.Draw(img)
    sizes = {"title": 34, "head": 23, "body": 22, "small": 18}
    y = 90
    for style, text in lines:
        d.text((90, y), text, fill=(25, 25, 25), font=_font(sizes[style]))
        y += sizes[style] + 14
    if hidden_text:
        d.text((90, H - 120), hidden_text, fill=(244, 243, 239), font=_font(12))

    # Make it look like a phone photo of paper.
    angle = rng.uniform(-4, 4) if quality != "clean" else rng.uniform(-1, 1)
    img = img.rotate(angle, expand=True, fillcolor=(120, 110, 95), resample=Image.BICUBIC)
    shade = Image.linear_gradient("L").resize(img.size).rotate(rng.choice([0, 90, 180, 270]))
    img = Image.composite(img, ImageEnhance.Brightness(img).enhance(0.72), shade)
    if quality == "blurry":
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(1.6, 2.6)))
    elif quality == "normal":
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 0.9)))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.05))
    img.save(path, optimize=True)


# ---------------------------------------------------------------- entry point
def generate(s: Settings, gold_claims: list[dict]) -> dict:
    rng, fk = make_rng(s, "docs"), make_faker(s, "docs")
    out = SYNTH / "documents"
    out.mkdir(parents=True, exist_ok=True)
    n_est = n_pol = 0
    for c in gold_claims:
        q = rng.choices(["clean", "normal", "blurry"], weights=[0.3, 0.5, 0.2])[0]
        est = repair_estimate_fields(c, fk, rng)
        stem = out / f"{c['claim_id']}_estimate"
        write_pdf(stem.with_suffix(".pdf"), estimate_text(est))
        write_photo_png(Path(f"{stem}_photo.png"), estimate_text(est), rng, q)
        (stem.with_suffix(".json")).write_text(json.dumps({**est, "photo_quality": q}, indent=2))
        c["documents"] = [f"documents/{c['claim_id']}_estimate.pdf"]
        n_est += 1
        if c["police_report_filed"]:
            pol = police_report_fields(c, fk, rng)
            stem = out / f"{c['claim_id']}_police"
            write_pdf(stem.with_suffix(".pdf"), police_text(pol))
            write_photo_png(Path(f"{stem}_photo.png"), police_text(pol), rng, q)
            (stem.with_suffix(".json")).write_text(json.dumps({**pol, "photo_quality": q}, indent=2))
            c["documents"].append(f"documents/{c['claim_id']}_police.pdf")
            n_pol += 1
    return {"repair_estimates": n_est, "police_reports": n_pol}
