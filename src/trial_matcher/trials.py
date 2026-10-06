"""Load trials from ClinicalTrials.gov (API v2) or from a local JSON file with the same shape."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx

from .schemas import Trial

CTGOV_STUDY = "https://clinicaltrials.gov/api/v2/studies/{nct}"


def parse_study(study: dict[str, Any]) -> dict[str, Any]:
    proto = study.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    elig = proto.get("eligibilityModule", {})
    nct = ident.get("nctId", "")
    return {
        "nct_id": nct,
        "title": ident.get("briefTitle", ""),
        "criteria_text": elig.get("eligibilityCriteria", ""),
        "minimum_age": elig.get("minimumAge"),
        "maximum_age": elig.get("maximumAge"),
        "sex": elig.get("sex", "ALL"),
        "url": f"https://clinicaltrials.gov/study/{nct}" if nct else "",
    }


_HEADING = re.compile(r"(?im)^\s*(?:key\s+|main\s+|major\s+)?(inclusion|exclusion)\s+criteria\b[^\n]*$")
_BULLET = re.compile(r"^(\s*)(?:(?:[-*\u2022\u25e6o]|\d+[.)]|[a-z][.)])\s+)?")


def parse_criteria(text: str) -> list[tuple[str, str]]:
    """All criteria as [(kind, text)], inclusion first.

    Handles "Key Inclusion Criteria:" style headings, and folds nested sub-bullets under a parent line that
    ends with ":" (e.g. "One of the following:") into one criterion: "One of the following: (a; b; c)".
    """
    pieces = _HEADING.split(text)
    out: list[tuple[str, str]] = []
    kind = "inclusion"
    # re.split with one group gives [before, kind1, body1, kind2, body2, ...]
    sections = [("inclusion", pieces[0])] + [(pieces[i].lower(), pieces[i + 1]) for i in range(1, len(pieces) - 1, 2)]
    for kind, body in sections:
        parent: list | None = None
        parent_indent = -1
        for line in body.splitlines():
            if not line.strip():
                continue
            m = _BULLET.match(line)
            indent = len(m.group(1).replace("\t", "    "))
            item = line[m.end() :].strip()
            if len(item) <= 3:
                continue
            if parent is not None and indent > parent_indent:
                parent[2].append(item)
                continue
            if parent is not None:
                out.append((parent[0], f"{parent[1]} ({'; '.join(parent[2])})" if parent[2] else parent[1]))
                parent = None
            if item.endswith(":"):
                parent, parent_indent = [kind, item, []], indent
            else:
                out.append((kind, item))
        if parent is not None:
            out.append((parent[0], f"{parent[1]} ({'; '.join(parent[2])})" if parent[2] else parent[1]))
    return [c for c in out if c[0] == "inclusion"] + [c for c in out if c[0] == "exclusion"]


def trial_from_study(study: dict) -> Trial:
    return Trial(**parse_study(study))


def load_trial(source: str) -> Trial:
    """`source` is a path to a CT.gov v2 study JSON, or an NCT id fetched from ClinicalTrials.gov."""
    path = Path(source)
    if path.exists():
        return trial_from_study(json.loads(path.read_text(encoding="utf-8")))
    resp = httpx.get(CTGOV_STUDY.format(nct=source), params={"format": "json"}, timeout=30)
    resp.raise_for_status()
    return trial_from_study(resp.json())


MAX_CRITERIA = 60


def criteria_list(trial: Trial, max_items: int = MAX_CRITERIA) -> tuple[list[tuple[str, str]], int]:
    """([(kind, text)] with inclusion first, then exclusion; number of criteria left out beyond max_items)."""
    items = parse_criteria(trial.criteria_text)
    return items[:max_items], max(0, len(items) - max_items)
