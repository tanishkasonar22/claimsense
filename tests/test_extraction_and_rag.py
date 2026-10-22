from claimsense.documents import parse_estimate, read_document
from claimsense.facts import extract_facts_rules, find_parts, incident_type, said_date
from claimsense.rag.coverage import assess_rules, retrieve
from claimsense.rag.policy_index import chunk_policy
from claimsense.schemas import ClaimFacts, DamageItem, DamageReport


def test_find_parts_prefers_longest_and_keeps_order():
    assert find_parts("damage to the front left door and rear left door") == ["front_left_door", "rear_left_door"]
    assert find_parts("my rear bumper, trunk and left taillamp") == ["rear_bumper", "trunk", "left_taillamp"]


def test_incident_type_keywords():
    assert incident_type("A deer ran out in front of me") == "animal_strike"
    assert incident_type("I hit a deep pothole on Main St") == "pothole"
    assert incident_type("A truck changed lanes into me") == "side_swipe"


def test_said_date_uses_reference_year():
    assert said_date("on December 4 in the evening", "2026-01-10") == "2025-12-04"
    assert said_date("on March 3", "2026-05-01") == "2026-03-03"


def test_chunk_policy_sections():
    md = "## Part 3. Coverage\n**§3.1 Collision.** We pay, minus a deductible of $500.\n"
    chunks = chunk_policy(md, "CS-TEST")
    assert chunks[0].section == "§3.1" and "$500" in chunks[0].text


def test_estimate_pdf_parsed_exactly(gold):
    import json
    from claimsense.config import get_settings
    g = gold[0]
    doc = read_document(g["inputs"]["documents"][0])
    est = parse_estimate(doc.text)
    truth = json.loads((get_settings().synthetic_dir / g["inputs"]["documents"][0].replace(".pdf", ".json")).read_text())
    assert est.total == truth["total"]
    assert [li.part for li in est.line_items] == [li["part"] for li in truth["line_items"]]


def test_coverage_hail_exclusion_on_economy_policy(gold):
    facts = ClaimFacts(incident_type="hail", estimate_parts=["hood"])
    dmg = DamageReport(items=[DamageItem(part="hood", damage_type="dent", severity="minor")], source="label_fallback")
    out = assess_rules(facts, dmg, retrieve("CS-ECON", facts, dmg))
    assert out.covered is False and "§4.5" in out.sections


def test_coverage_no_collision_on_basic_policy(gold):
    facts = ClaimFacts(incident_type="rear_end_collision", estimate_parts=["rear_bumper"])
    dmg = DamageReport(items=[DamageItem(part="rear_bumper", damage_type="dent", severity="minor")], source="label_fallback")
    out = assess_rules(facts, dmg, retrieve("CS-BASIC", facts, dmg))
    assert out.covered is False


def test_rules_extractor_on_gold_sample(gold):
    from claimsense.documents import parse_police
    from claimsense.speech import voice_script
    ok = 0
    for g in gold[:30]:
        est = parse_estimate(read_document(g["inputs"]["documents"][0]).text)
        pol = parse_police(read_document(g["inputs"]["documents"][1]).text) if len(g["inputs"]["documents"]) > 1 else None
        f = extract_facts_rules(voice_script(g["claim_id"]), est, pol)
        ok += f.incident_type == g["expected"]["voice_facts"]["incident_type"]
    assert ok >= 27
