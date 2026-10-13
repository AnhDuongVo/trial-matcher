"""Scripted synthetic model fixtures for the offline workflow demo."""

import re

PARSED = {
    "criteria": [
        {"index": 0, "kind": "inclusion", "text": "", "category": "age", "age_min": 18, "age_max": 75},
        {"index": 1, "kind": "inclusion", "text": "", "category": "condition", "terms": ["type 2 diabetes", "T2DM"]},
        {
            "index": 2,
            "kind": "inclusion",
            "text": "",
            "category": "lab",
            "lab_name": "HbA1c",
            "comparator": "between",
            "value": 7.5,
            "value_high": 10.5,
            "unit": "%",
        },
        {"index": 3, "kind": "inclusion", "text": "", "category": "medication", "terms": ["metformin"]},
        {
            "index": 4,
            "kind": "inclusion",
            "text": "",
            "category": "lab",
            "lab_name": "BMI",
            "comparator": "between",
            "value": 25,
            "value_high": 40,
            "unit": "kg/m2",
        },
        {"index": 5, "kind": "exclusion", "text": "", "category": "condition", "terms": ["type 1 diabetes"]},
        {
            "index": 6,
            "kind": "exclusion",
            "text": "",
            "category": "lab",
            "lab_name": "eGFR",
            "comparator": "<",
            "value": 45,
            "unit": "mL/min/1.73 m2",
        },
        {"index": 7, "kind": "exclusion", "text": "", "category": "condition", "terms": ["myocardial infarction"]},
        {"index": 8, "kind": "exclusion", "text": "", "category": "medication", "terms": ["insulin"]},
        {"index": 9, "kind": "exclusion", "text": "", "category": "pregnancy", "terms": ["pregnancy"]},
    ]
}


def fake_assess(messages):
    """Stand-in for the LLM: reads the criterion and the facts in the prompt, answers like a careful model."""
    msg = messages[-1]["content"]
    crit = re.search("Criterion \\((\\w+)\\): (.*)", msg).group(2).lower()
    facts = msg.split("Facts from the record:")[1]

    def ids(pat):
        return re.findall("\\[(F\\d+)\\][^\\n]*" + pat, facts, re.I)

    if "type 2 diabetes" in crit:
        return {"status": "met", "confidence": 0.95, "evidence": ids("type 2")[:1], "rationale": "T2D on record"}
    if "metformin" in crit:
        hits = [line for line in facts.splitlines() if "metformin" in line.lower() and "status active" in line]
        ok = any("1000 mg twice" in line for line in hits)
        return {
            "status": "met" if ok else "not_met",
            "confidence": 0.85,
            "evidence": re.findall("\\[(F\\d+)\\]", "\n".join(hits))[:1],
            "rationale": "dose check",
        }
    if "myocardial" in crit:
        hit = ids("myocardial infarction")
        return {
            "status": "met" if hit else "not_met",
            "confidence": 0.9 if hit else 0.6,
            "evidence": hit,
            "rationale": "MI in window" if hit else "none found",
        }
    if "insulin" in crit:
        hit = [i for i in ids("insulin") if "status active" in facts]
        return {
            "status": "met" if hit else "not_met",
            "confidence": 0.9 if hit else 0.6,
            "evidence": hit,
            "rationale": "insulin",
        }
    if "pregnan" in crit:
        return {"status": "unknown", "confidence": 0.5, "evidence": [], "rationale": "not documented"}
    return {"status": "not_met", "confidence": 0.6, "evidence": [], "rationale": "not found"}
