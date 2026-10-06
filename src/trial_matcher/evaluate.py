"""Evaluate on a labelled set: criterion-level accuracy and F1, trial-level verdicts, and calibration.

Label file (JSONL), one line per patient-trial pair:
  {"patient": "patients/p1.json", "trial": "trials/SYN-T2D-001.json", "as_of": "2026-10-01",
   "verdict": "eligible", "criteria": ["met", "met", ..., "not_met"]}   # in criteria order, inclusion first

Calibration matters for a pre-screen: if the system says 0.9, it should be right about 90% of the time,
otherwise coordinators cannot decide which results to double-check.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from .fhir_facts import extract_facts, load_bundle
from .llm import LLM
from .matcher import match
from .trials import load_trial

CLASSES = ("met", "not_met", "unknown")
BINS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]


def f1_scores(pairs: list[tuple[str, str]]) -> dict:
    out = {}
    for c in CLASSES:
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if g != c and p == c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[c] = {
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
            "support": tp + fn,
        }
    out["macro_f1"] = round(sum(out[c]["f1"] for c in CLASSES) / len(CLASSES), 3)
    return out


def calibration(rows: list[tuple[float, bool]]) -> dict:
    table, ece, n = [], 0.0, len(rows)
    for lo, hi in BINS:
        b = [(c, ok) for c, ok in rows if lo <= c < hi]
        if not b:
            continue
        conf = sum(c for c, _ in b) / len(b)
        acc = sum(ok for _, ok in b) / len(b)
        ece += len(b) / n * abs(conf - acc)
        table.append(
            {"bin": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": len(b), "mean_conf": round(conf, 3), "accuracy": round(acc, 3)}
        )
    return {"ece": round(ece, 3), "bins": table}


async def evaluate(
    llm: LLM, labels: str | Path, base_dir: str | Path | None = None, out_dir: str | Path = "runs/eval"
) -> dict:
    labels = Path(labels)
    base = Path(base_dir) if base_dir else labels.parent
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs, calib, verdicts, by_method = [], [], [], defaultdict(list)
    errors = []
    with (out / "predictions.jsonl").open("w") as fh:
        for line in labels.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            pf = extract_facts(load_bundle(base / row["patient"]), date.fromisoformat(row["as_of"]))
            trial = load_trial(str(base / row["trial"]))
            try:
                result, _ = await match(pf, trial, llm)
            except Exception as err:
                errors.append({"pair": [row["patient"], row["trial"]], "error": str(err)[:300]})
                continue
            gold = row["criteria"]
            if len(gold) != len(result.assessments):
                errors.append(
                    {
                        "pair": [row["patient"], row["trial"]],
                        "error": f"{len(gold)} labels vs {len(result.assessments)} criteria",
                    }
                )
            for g, a in zip(gold, result.assessments, strict=False):
                pairs.append((g, a.status))
                calib.append((a.confidence, g == a.status))
                by_method[a.method].append(g == a.status)
            verdicts.append((row["verdict"], result.verdict))
            fh.write(
                json.dumps(
                    {
                        "patient": row["patient"],
                        "trial": row["trial"],
                        "gold_verdict": row["verdict"],
                        "result": result.model_dump(),
                    }
                )
                + "\n"
            )

    decided = [(g, p) for g, p in pairs if p != "unknown"]
    summary = {
        "pairs": len(verdicts),
        "criteria": len(pairs),
        "criterion_accuracy": round(sum(g == p for g, p in pairs) / len(pairs), 3) if pairs else None,
        "accuracy_when_decided": round(sum(g == p for g, p in decided) / len(decided), 3) if decided else None,
        "coverage_decided": round(len(decided) / len(pairs), 3) if pairs else None,
        "per_class": f1_scores(pairs),
        "by_method": {m: {"n": len(v), "accuracy": round(sum(v) / len(v), 3)} for m, v in by_method.items()},
        "verdict_accuracy": round(sum(g == p for g, p in verdicts) / len(verdicts), 3) if verdicts else None,
        "verdict_confusion": dict(Counter(f"{g}->{p}" for g, p in verdicts)),
        "calibration": calibration(calib),
        "errors": errors,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
