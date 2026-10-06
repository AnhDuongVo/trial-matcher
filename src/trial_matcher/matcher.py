"""Trial matching agent (LangGraph).

    parse_criteria (LLM, one call)
      -> assess (code rules for age, sex and labs; LLM for the rest, in parallel)
      -> decide (conservative verdict) -> [coordinator review interrupt] -> finalize (FHIR ResearchSubject)

Design choices:
* Numbers are compared by code, never by the LLM; the LLM handles semantics and time windows.
* "Absence of evidence" is not evidence: a not_met that rests only on nothing being found is capped at 0.6.
* The verdict is conservative: "eligible" needs every inclusion met and every exclusion not met.
* A coordinator can override any criterion; overrides are recorded with name and time.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, date, datetime
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .fhir_facts import relevant_facts
from .llm import LLM, complete_structured
from .prompts import ASSESS, PARSE, SYSTEM
from .rules import apply_rules
from .schemas import Assessment, Criterion, LLMAssessment, MatchResult, ParsedCriteria, PatientFacts, Trial
from .trials import criteria_list

ABSENCE_CAP = 0.6
DECISIVE = 0.7

CATEGORY_KINDS = {
    "condition": {"condition", "procedure"},
    "procedure": {"procedure", "condition"},
    "medication": {"medication"},
    "lab": {"observation"},
    "pregnancy": {"condition", "observation"},
}


class MatchState(TypedDict, total=False):
    facts: dict
    trial: dict
    criteria: list[dict]
    assessments: list[dict]
    result: dict
    overrides: dict  # {"3": {"status": "met", "note": "..."}}
    reviewer: str
    fhir: dict


def aggregate(assessments: list[Assessment]) -> tuple[str, float]:
    inc = [a for a in assessments if a.kind == "inclusion"]
    exc = [a for a in assessments if a.kind == "exclusion"]
    fails = [a for a in inc if a.status == "not_met"] + [a for a in exc if a.status == "met"]
    decisive = [a for a in fails if a.confidence >= DECISIVE]
    if decisive:
        return "ineligible", round(max(a.confidence for a in decisive), 3)
    if inc and not fails and all(a.status == "met" for a in inc) and all(a.status == "not_met" for a in exc):
        return "eligible", round(min(a.confidence for a in assessments), 3)
    return "needs_review", round(min((a.confidence for a in assessments), default=0.0), 3)


def facts_for(crit: Criterion, pf: PatientFacts, cap: int = 120) -> list:
    kinds = CATEGORY_KINDS.get(crit.category)
    if crit.category in {"condition", "procedure", "medication"}:
        # Absence decisions need the complete list of that kind, not just keyword hits. Facts are ordered
        # active and newest first, so a cap drops old history, never the current state.
        pool = [f for f in pf.facts if f.kind in kinds]
        hits = relevant_facts(pf, f"{crit.text} {' '.join(crit.terms)}", kinds, k=cap)
        merged = hits + [f for f in pool if f not in hits]
        return merged[:cap]
    query = f"{crit.text} {' '.join(crit.terms)} {crit.lab_name or ''}"
    return relevant_facts(pf, query, kinds, k=25)


async def assess_llm(crit: Criterion, pf: PatientFacts, llm: LLM) -> Assessment:
    facts = facts_for(crit, pf)
    out = await complete_structured(
        llm,
        [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": ASSESS.format(
                    as_of=pf.as_of,
                    age=pf.age_years,
                    sex=pf.sex,
                    kind=crit.kind,
                    criterion=crit.text,
                    facts="\n".join(f.render() for f in facts) or "(no matching facts)",
                ),
            },
        ],
        LLMAssessment,
        role="fast",
        step="assess",
        max_tokens=512,
    )
    allowed = {f.id: f for f in facts}
    evidence = [e for e in out.evidence if e in allowed]
    status, confidence, rationale = out.status, out.confidence, out.rationale
    if status == "met" and not evidence:
        status, confidence, rationale = "unknown", 0.3, f"no evidence cited; model said met: {rationale}"
    if status == "not_met" and not evidence:
        confidence = min(confidence, ABSENCE_CAP)
    return Assessment(
        index=crit.index,
        kind=crit.kind,
        criterion=crit.text,
        status=status,
        confidence=confidence,
        evidence=evidence,
        evidence_refs=[allowed[e].ref for e in evidence],
        rationale=rationale,
        method="llm",
    )


_TOK = re.compile(r"[a-z0-9]+")


def _aligned(parsed: list[Criterion], index: int, text: str) -> Criterion | None:
    """The parsed entry for criterion `index`, verified by text so off-by-one numbering cannot shift structure."""
    words = set(_TOK.findall(text.lower()))

    def sim(c: Criterion) -> float:
        other = set(_TOK.findall(c.text.lower()))
        return len(words & other) / len(words | other) if words and other else 0.0

    same_index = next((c for c in parsed if c.index == index), None)
    if same_index is not None and (not same_index.text or sim(same_index) >= 0.5):
        return same_index
    best = max(parsed, key=sim, default=None)
    return best if best is not None and sim(best) >= 0.6 else None


def build_graph(llm: LLM, human_review: bool = False, max_parallel: int = 4, checkpointer: Any | None = None):
    async def parse(state: MatchState) -> dict:
        trial = Trial.model_validate(state["trial"])
        items, left_out = criteria_list(trial)
        listing = "\n".join(f"{i}. ({kind}) {text}" for i, (kind, text) in enumerate(items))
        parsed = await complete_structured(
            llm,
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PARSE.format(criteria=listing)}],
            ParsedCriteria,
            role="fast",
            step="parse_criteria",
        )
        criteria = []
        for i, (kind, text) in enumerate(items):
            c = _aligned(parsed.criteria, i, text) or Criterion(index=i, kind=kind, text=text)
            # The split is authoritative for index, kind and text; the LLM only adds structure.
            criteria.append(c.model_copy(update={"index": i, "kind": kind, "text": text}).model_dump())
        if left_out:
            criteria.append(
                Criterion(
                    index=len(items),
                    kind="inclusion",
                    category="other",
                    text=f"{left_out} further criteria were not assessed (list too long)",
                ).model_dump()
            )
        return {"criteria": criteria}

    async def assess(state: MatchState) -> dict:
        pf = PatientFacts.model_validate(state["facts"])
        sem = asyncio.Semaphore(max_parallel)

        async def one(cd: dict) -> Assessment:
            crit = Criterion.model_validate(cd)
            if crit.text.endswith("(list too long)"):
                return Assessment(
                    index=crit.index,
                    kind=crit.kind,
                    criterion=crit.text,
                    status="unknown",
                    confidence=0.0,
                    rationale="not assessed: review the full trial text",
                    method="rule",
                )
            try:
                ruled = apply_rules(crit, pf)
            except Exception:  # bad data in one resource must not stop the screening
                ruled = None
            if ruled is not None:
                return ruled
            async with sem:
                try:
                    return await assess_llm(crit, pf, llm)
                except Exception as err:  # one failure must not lose the other criteria
                    return Assessment(
                        index=crit.index,
                        kind=crit.kind,
                        criterion=crit.text,
                        status="unknown",
                        confidence=0.0,
                        rationale=f"assessment failed: {err}",
                        method="llm",
                    )

        results = await asyncio.gather(*(one(c) for c in state["criteria"]))
        return {"assessments": [a.model_dump() for a in sorted(results, key=lambda a: a.index)]}

    async def decide(state: MatchState) -> dict:
        assessments = [Assessment.model_validate(a) for a in state["assessments"]]
        verdict, conf = aggregate(assessments)
        result = MatchResult(
            patient_ref=state["facts"]["patient_ref"],
            trial=Trial.model_validate(state["trial"]),
            verdict=verdict,
            confidence=conf,
            assessments=assessments,
        )
        return {"result": result.model_dump()}

    async def finalize(state: MatchState) -> dict:
        from .fhir_out import screening_bundle

        result = MatchResult.model_validate(state["result"])
        overrides = state.get("overrides") or {}
        if overrides or state.get("reviewer"):
            new = []
            for a in result.assessments:
                o = overrides.get(str(a.index))
                if o:
                    a = a.model_copy(
                        update={
                            "status": o["status"],
                            "confidence": 1.0,
                            "method": "reviewer",
                            "reviewer_note": o.get("note"),
                        }
                    )
                new.append(a)
            verdict, conf = aggregate(new)
            result = result.model_copy(
                update={
                    "assessments": new,
                    "verdict": verdict,
                    "confidence": conf,
                    "reviewed_by": state.get("reviewer") or "coordinator",
                    "reviewed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                }
            )
        return {"result": result.model_dump(), "fhir": screening_bundle(result)}

    g = StateGraph(MatchState)
    g.add_node("parse_criteria", parse)
    g.add_node("assess", assess)
    g.add_node("decide", decide)
    g.add_node("finalize", finalize)
    g.add_edge(START, "parse_criteria")
    g.add_edge("parse_criteria", "assess")
    g.add_edge("assess", "decide")
    g.add_edge("decide", "finalize")
    g.add_edge("finalize", END)
    if human_review:
        return g.compile(checkpointer=checkpointer or MemorySaver(), interrupt_before=["finalize"])
    return g.compile(checkpointer=checkpointer)


async def match(pf: PatientFacts, trial: Trial, llm: LLM) -> tuple[MatchResult, dict]:
    """Run without the review pause. The FHIR ResearchSubject stays 'candidate' unless reviewed."""
    state = await build_graph(llm).ainvoke({"facts": pf.model_dump(), "trial": trial.model_dump()})
    return MatchResult.model_validate(state["result"]), state["fhir"]


def report_markdown(result: MatchResult, pf: PatientFacts) -> str:
    facts = pf.by_id()
    lines = [
        f"# Pre-screening: {result.trial.nct_id}",
        "",
        f"**{result.trial.title}**",
        "",
        f"Patient {result.patient_ref}, age {pf.age_years}, {pf.sex}, screening date {pf.as_of}",
        "",
        f"**Verdict: {result.verdict}** (confidence {result.confidence:.2f})"
        + (f", reviewed by {result.reviewed_by} at {result.reviewed_at}" if result.reviewed_by else ""),
        "",
        "| # | Type | Criterion | Status | Conf. | Method | Evidence | Rationale |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a in result.assessments:
        ev = "; ".join(f"{e}: {facts[e].text[:40]}" for e in a.evidence if e in facts) or "none"
        note = f" (reviewer: {a.reviewer_note})" if a.reviewer_note else ""
        lines.append(
            f"| {a.index} | {a.kind[:3]} | {a.criterion} | {a.status} | {a.confidence:.2f} | {a.method} | "
            f"{ev} | {a.rationale.replace('|', '/')}{note} |"
        )
    lines += ["", "_Research pre-screening support only. A study coordinator and investigator confirm eligibility._"]
    return "\n".join(lines) + "\n"


def as_of_date(value: str | None) -> date:
    return date.fromisoformat(value) if value else date.today()


__all__ = ["aggregate", "as_of_date", "build_graph", "match", "report_markdown"]
