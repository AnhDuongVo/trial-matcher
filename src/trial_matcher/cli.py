"""`ctm --help`: clinical trial pre-screening from a FHIR record."""

from __future__ import annotations

import asyncio
import json
import uuid
from importlib import resources
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown

from .config import get_settings
from .fhir_facts import extract_facts, load_bundle
from .llm import NIMClient
from .matcher import as_of_date, build_graph, report_markdown
from .schemas import MatchResult, PatientFacts
from .trials import load_trial

app = typer.Typer(add_completion=False, help="Trial matching agent: FHIR patient + trial -> per-criterion eligibility.")
console = Console()


def samples_dir() -> Path:
    return Path(str(resources.files("trial_matcher") / "samples"))


def _save(out_dir: Path, result: MatchResult, pf: PatientFacts, fhir: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(result.model_dump_json(indent=2))
    (out_dir / "report.md").write_text(report_markdown(result, pf))
    (out_dir / "facts.json").write_text(pf.model_dump_json(indent=2))
    (out_dir / "research_subject_bundle.json").write_text(json.dumps(fhir, indent=2))


async def _run(
    patient: Path, trial_src: str, as_of: str | None, review: bool, reviewer: str | None, out_dir: Path
) -> MatchResult:
    pf = extract_facts(load_bundle(patient), as_of_date(as_of))
    trial = load_trial(trial_src)
    llm = NIMClient(get_settings())
    graph = build_graph(llm, human_review=review)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = await graph.ainvoke({"facts": pf.model_dump(), "trial": trial.model_dump()}, config)
    if review:
        snapshot = await graph.aget_state(config)
        draft = MatchResult.model_validate(snapshot.values["result"])
        console.print(Markdown(report_markdown(draft, pf)))
        overrides: dict[str, dict] = {}
        console.print(
            "Override criteria? Enter `index status note` (status: met, not_met, unknown), empty line to finish."
        )
        while True:
            line = typer.prompt("override", default="", show_default=False).strip()
            if not line:
                break
            parts = line.split(maxsplit=2)
            if len(parts) < 2 or parts[1] not in {"met", "not_met", "unknown"} or not parts[0].isdigit():
                console.print("[yellow]format: 3 not_met lab is from 2024[/]")
                continue
            overrides[parts[0]] = {"status": parts[1], "note": parts[2] if len(parts) > 2 else ""}
        name = reviewer or typer.prompt("Reviewer name", default="Study coordinator")
        await graph.aupdate_state(config, {"overrides": overrides, "reviewer": name})
        state = await graph.ainvoke(None, config)
    result = MatchResult.model_validate(state["result"])
    _save(out_dir, result, pf, state["fhir"])
    return result


@app.command(name="match")
def match_cmd(
    patient: Path = typer.Argument(..., exists=True, help="FHIR R4 Bundle JSON (e.g. Synthea output)"),
    trial: str = typer.Argument(..., help="NCT id (fetched from ClinicalTrials.gov) or a study JSON file"),
    as_of: str = typer.Option(None, help="Screening date YYYY-MM-DD (default today)"),
    review: bool = typer.Option(True, help="Pause for coordinator review and overrides"),
    reviewer: str = typer.Option(None),
    out_dir: Path = typer.Option(Path("runs/latest")),
):
    """Pre-screen one patient for one trial."""
    result = asyncio.run(_run(patient, trial, as_of, review, reviewer, out_dir))
    console.print(f"[bold]Verdict:[/] {result.verdict} ({result.confidence:.2f}). Saved to {out_dir}/")


@app.command()
def demo(
    trial: str = typer.Option("SYN-T2D-001"),
    patient: str = typer.Option("p1"),
    out_dir: Path = typer.Option(Path("runs/demo")),
):
    """Bundled synthetic patient and trial, no review pause."""
    s = samples_dir()
    result = asyncio.run(
        _run(
            s / "patients" / f"{patient}.json", str(s / "trials" / f"{trial}.json"), "2026-10-01", False, None, out_dir
        )
    )
    pf = extract_facts(load_bundle(s / "patients" / f"{patient}.json"), as_of_date("2026-10-01"))
    console.print(Markdown(report_markdown(result, pf)))


@app.command()
def facts(patient: Path = typer.Argument(..., exists=True), as_of: str = typer.Option(None)):
    """Show the citable facts extracted from a FHIR bundle (no LLM)."""
    pf = extract_facts(load_bundle(patient), as_of_date(as_of))
    console.print(f"{pf.patient_ref}: age {pf.age_years}, {pf.sex}, {len(pf.facts)} facts (as of {pf.as_of})")
    for f in pf.facts:
        console.print(f.render())


@app.command(name="eval")
def eval_cmd(
    labels: Path = typer.Option(None, help="Labels JSONL (default: bundled synthetic set)"),
    out_dir: Path = typer.Option(Path("runs/eval")),
):
    """Criterion accuracy, macro-F1, verdict accuracy and calibration on a labelled set."""
    from .evaluate import evaluate

    labels = labels or samples_dir() / "labels.jsonl"
    summary = asyncio.run(evaluate(NIMClient(get_settings()), labels, out_dir=out_dir))
    console.print_json(json.dumps(summary))


@app.command()
def models():
    """List the model IDs your key can use (check these against .env)."""
    from openai import OpenAI

    s = get_settings()
    client = OpenAI(base_url=s.base_url, api_key=s.require_api_key())
    ids = sorted(m.id for m in client.models.list().data)
    for mid in ids:
        mark = " <- configured" if mid == s.model_fast else ""
        console.print(f"{mid}{mark}")
    for role, mid in (("CTM_MODEL_FAST", s.model_fast),):
        if mid not in ids:
            console.print(f"[red]{role}={mid} is not in the list. Pick another ID in .env[/]")


if __name__ == "__main__":
    app()
