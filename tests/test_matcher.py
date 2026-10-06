import re
import uuid
from datetime import date
from pathlib import Path

from fhir.resources.R4B.bundle import Bundle
from fhir.resources.R4B.researchstudy import ResearchStudy
from fhir.resources.R4B.researchsubject import ResearchSubject

from trial_matcher.evaluate import calibration, evaluate, f1_scores
from trial_matcher.fhir_facts import extract_facts, load_bundle
from trial_matcher.llm import FakeLLM
from trial_matcher.matcher import aggregate, build_graph, match
from trial_matcher.rules import apply_rules
from trial_matcher.schemas import Assessment, Criterion, MatchResult
from trial_matcher.trials import criteria_list, load_trial

S = Path(__file__).resolve().parents[1] / "src" / "trial_matcher" / "samples"
AS_OF = date(2026, 10, 1)

# What a good parser would return for SYN-T2D-001 (inclusion 0-4, exclusion 5-9).
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
    crit = re.search(r"Criterion \((\w+)\): (.*)", msg).group(2).lower()
    facts = msg.split("Facts from the record:")[1]

    def ids(pat):
        return re.findall(r"\[(F\d+)\][^\n]*" + pat, facts, re.I)

    if "type 2 diabetes" in crit:
        return {"status": "met", "confidence": 0.95, "evidence": ids("type 2")[:1], "rationale": "T2D on record"}
    if "metformin" in crit:
        hits = [line for line in facts.splitlines() if "metformin" in line.lower() and "status active" in line]
        ok = any("1000 mg twice" in line for line in hits)
        return {
            "status": "met" if ok else "not_met",
            "confidence": 0.85,
            "evidence": re.findall(r"\[(F\d+)\]", "\n".join(hits))[:1],
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


def fake_llm():
    return FakeLLM({"parse_criteria": PARSED, "assess": fake_assess})


def test_extract_facts_latest_lab_and_age():
    pf = extract_facts(load_bundle(S / "patients" / "p1.json"), AS_OF)
    assert pf.age_years == 58 and pf.sex == "male"
    hba1c = [f for f in pf.facts if f.code == "4548-4"]
    assert len(hba1c) == 1 and hba1c[0].value == 8.4 and "latest of 2" in hba1c[0].text


def test_rules():
    pf3 = extract_facts(load_bundle(S / "patients" / "p3.json"), AS_OF)
    lab = Criterion(
        index=2,
        kind="inclusion",
        text="HbA1c 7.5-10.5%",
        category="lab",
        lab_name="HbA1c",
        comparator="between",
        value=7.5,
        value_high=10.5,
        unit="%",
    )
    old = apply_rules(lab, pf3)
    assert old.status == "unknown" and "older than" in old.rationale  # 2025 value is too old
    egfr = Criterion(
        index=6,
        kind="exclusion",
        text="eGFR below 45",
        category="lab",
        lab_name="eGFR",
        comparator="<",
        value=45,
        unit="mL/min/1.73 m2",
    )
    assert apply_rules(egfr, pf3).status == "not_met"  # 58 is not below 45
    wrong_unit = egfr.model_copy(update={"unit": "mL/s"})
    assert apply_rules(wrong_unit, pf3) is None  # unit mismatch goes to the LLM
    preg = Criterion(index=9, kind="exclusion", text="Pregnancy", category="pregnancy")
    assert apply_rules(preg, pf3).status == "not_met"
    test = Criterion(index=1, kind="inclusion", text="Negative pregnancy test in women", category="pregnancy")
    assert apply_rules(test, pf3).status == "met"  # not applicable to a male patient
    compound = egfr.model_copy(update={"text": "Creatinine > 1.5 mg/dL or eGFR below 45"})
    assert apply_rules(compound, pf3) is None  # compound logic goes to the LLM
    mmol = lab.model_copy(update={"unit": "mmol/mol"})
    pf1 = extract_facts(load_bundle(S / "patients" / "p1.json"), AS_OF)
    assert apply_rules(mmol, pf1) is None  # HbA1c recorded in %, criterion in mmol/mol


def test_aggregate_is_conservative():
    def mk(kind, status, conf):
        return Assessment(index=0, kind=kind, criterion="x", status=status, confidence=conf)

    assert aggregate([mk("inclusion", "met", 0.9), mk("exclusion", "not_met", 0.6)])[0] == "eligible"
    assert aggregate([mk("inclusion", "met", 0.9), mk("exclusion", "unknown", 0.5)])[0] == "needs_review"
    assert aggregate([mk("inclusion", "not_met", 0.95)])[0] == "ineligible"
    assert aggregate([mk("inclusion", "not_met", 0.5)])[0] == "needs_review"  # weak evidence is not decisive


async def test_match_p1_and_fhir_output():
    pf = extract_facts(load_bundle(S / "patients" / "p1.json"), AS_OF)
    trial = load_trial(str(S / "trials" / "SYN-T2D-001.json"))
    assert len(criteria_list(trial)[0]) == 10
    result, fhir = await match(pf, trial, fake_llm())
    assert result.verdict == "eligible"
    assert [a.method for a in result.assessments][:3] == ["rule", "llm", "rule"]
    assert result.assessments[2].evidence_refs == ["Observation/p1-o2"]  # the latest HbA1c, cited by reference
    Bundle.model_validate(fhir)
    ResearchStudy.model_validate(fhir["entry"][0]["resource"])  # nested resources are not validated by Bundle
    ResearchSubject.model_validate(fhir["entry"][1]["resource"])
    assert fhir["entry"][1]["resource"]["status"] == "candidate"  # not reviewed yet


async def test_review_override_changes_verdict():
    pf = extract_facts(load_bundle(S / "patients" / "p1.json"), AS_OF)
    trial = load_trial(str(S / "trials" / "SYN-T2D-001.json"))
    graph = build_graph(fake_llm(), human_review=True)
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    await graph.ainvoke({"facts": pf.model_dump(), "trial": trial.model_dump()}, cfg)
    await graph.aupdate_state(
        cfg, {"overrides": {"8": {"status": "met", "note": "starts insulin next week"}}, "reviewer": "Coordinator A"}
    )
    state = await graph.ainvoke(None, cfg)
    result = MatchResult.model_validate(state["result"])
    assert result.verdict == "ineligible" and result.reviewed_by == "Coordinator A"
    assert result.assessments[8].method == "reviewer"
    assert state["fhir"]["entry"][1]["resource"]["status"] == "ineligible"


async def test_evaluate_on_bundled_labels(tmp_path):
    labels = tmp_path / "labels.jsonl"
    lines = [line for line in (S / "labels.jsonl").read_text().splitlines() if "SYN-T2D-001" in line]
    labels.write_text("\n".join(lines))
    summary = await evaluate(fake_llm(), labels, base_dir=S, out_dir=tmp_path / "out")
    assert summary["pairs"] == 3 and summary["criteria"] == 30
    assert summary["criterion_accuracy"] >= 0.8
    assert summary["verdict_accuracy"] == 1.0
    assert "ece" in summary["calibration"]


def test_real_fhir_shapes():
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {"resource": r}
            for r in [
                {"resourceType": "Patient", "id": "x", "gender": "female", "birthDate": "1970"},
                {"resourceType": "Medication", "id": "m1", "code": {"text": "Insulin lispro"}},
                {
                    "resourceType": "MedicationRequest",
                    "id": "r1",
                    "status": "active",
                    "medicationReference": {"reference": "Medication/m1"},
                    "authoredOn": "2026-01",
                    "dosageInstruction": [
                        {
                            "doseAndRate": [{"doseQuantity": {"value": 5, "unit": "U"}}],
                            "timing": {"repeat": {"frequency": 3, "period": 1, "periodUnit": "d"}},
                        }
                    ],
                },
                {
                    "resourceType": "Observation",
                    "id": "bp",
                    "status": "final",
                    "effectiveDateTime": "2026-09-01",
                    "code": {"text": "Blood pressure panel"},
                    "component": [
                        {
                            "code": {
                                "coding": [
                                    {
                                        "system": "http://loinc.org",
                                        "code": "8480-6",
                                        "display": "Systolic blood pressure",
                                    }
                                ]
                            },
                            "valueQuantity": {"value": 150, "unit": "mm[Hg]"},
                        }
                    ],
                },
                {"resourceType": "Condition", "id": "c1", "code": {"text": "Full-time employment (finding)"}},
                {"resourceType": "Condition", "id": "c2", "code": {"text": "Dependence on renal dialysis (finding)"}},
            ]
        ],
    }
    pf = extract_facts(bundle, AS_OF)
    texts = [f.render() for f in pf.facts]
    assert pf.age_years == 56
    assert any("Insulin lispro" in t and "5 U 3 times per 1 d" in t for t in texts)
    assert any("Systolic blood pressure" in t and "150" in t for t in texts)
    assert any("dialysis" in t for t in texts) and not any("employment" in t for t in texts)


def test_metric_helpers():
    assert f1_scores([("met", "met"), ("not_met", "met")])["met"]["precision"] == 0.5
    assert calibration([(0.95, True), (0.95, False)])["ece"] == 0.45
