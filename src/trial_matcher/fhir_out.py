"""Write the screening outcome back as FHIR R4: ResearchStudy + ResearchSubject (+ the reasoning as a note)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from .schemas import MatchResult

SUBJECT_STATUS = {"eligible": "eligible", "ineligible": "ineligible", "needs_review": "candidate"}


def screening_bundle(result: MatchResult) -> dict:
    study_url, subject_url = f"urn:uuid:{uuid.uuid4()}", f"urn:uuid:{uuid.uuid4()}"
    reviewed = result.reviewed_by is not None
    # Without a human review the subject is only ever a "candidate", whatever the model concluded.
    status = SUBJECT_STATUS[result.verdict] if reviewed else "candidate"
    study = {
        "resourceType": "ResearchStudy",
        "identifier": [{"system": "https://clinicaltrials.gov", "value": result.trial.nct_id}],
        "title": result.trial.title,
        "status": "active",
    }
    lines = [
        f"{a.kind} #{a.index}: {a.status} ({a.confidence:.2f}, {a.method}) {a.criterion}"
        + (f" [evidence: {', '.join(a.evidence_refs)}]" if a.evidence_refs else "")
        for a in result.assessments
    ]
    subject = {
        "resourceType": "ResearchSubject",
        "status": status,
        "study": {"reference": study_url},
        "individual": {"reference": result.patient_ref},
        "extension": [
            {
                "url": "urn:example:prescreening-summary",
                "valueString": f"AI pre-screen verdict {result.verdict} (confidence {result.confidence:.2f})"
                + (f"; reviewed by {result.reviewed_by}" if reviewed else "; not yet reviewed")
                + "\n"
                + "\n".join(lines),
            }
        ],
    }
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "entry": [{"fullUrl": study_url, "resource": study}, {"fullUrl": subject_url, "resource": subject}],
    }
