import asyncio
import json
import uuid
from datetime import date
from importlib.resources import files
from pathlib import Path

import streamlit as st
from demo_fixture import PARSED, fake_assess

from trial_matcher.fhir_facts import extract_facts, load_bundle
from trial_matcher.llm import FakeLLM
from trial_matcher.matcher import build_graph, report_markdown
from trial_matcher.schemas import MatchResult
from trial_matcher.trials import criteria_list, load_trial

st.set_page_config(layout="wide", page_title="trial-matcher workflow")
st.markdown(
    """<style>.block-container{padding-top:1.3rem;max-width:1400px}h1{font-size:1.9rem!important}h2{font-size:1.3rem!important}h3{font-size:1.1rem!important}div[data-testid=stMetricValue]{font-size:1.5rem}textarea{font-size:13px!important}</style>""",
    unsafe_allow_html=True,
)


st.title("trial-matcher: FHIR record → reviewed screening")
st.caption(
    "OFFLINE · Synthetic patient and trial · Scripted criterion parsing and model assessments · Actual FHIR extraction, rules, LangGraph review pause and export. No live trial retrieval or model calls."
)
S = Path(str(files("trial_matcher") / "samples"))
a, b = st.columns([1, 1.3])
with a:
    st.subheader("Input · patient + trial")
    patient = st.selectbox("Synthetic FHIR patient", ["p1", "p2", "p3"])
    pf = extract_facts(load_bundle(S / "patients" / f"{patient}.json"), date(2026, 10, 1))
    trial = load_trial(str(S / "trials" / "SYN-T2D-001.json"))
    st.write(f"**{trial.nct_id} · {trial.title}**")
    st.caption(f"Screening date: {pf.as_of} · age {pf.age_years} · {pf.sex}")
    with st.expander("Patient facts extracted from FHIR", expanded=True):
        st.text("\n".join(f.render() for f in pf.facts))
    with st.expander("Trial inclusion and exclusion criteria"):
        for i, (kind, text) in enumerate(criteria_list(trial)[0]):
            st.write(f"{i} · {kind}: {text}")
    if st.button("Run pre-screening", type="primary"):

        async def screen():
            graph = build_graph(FakeLLM({"parse_criteria": PARSED, "assess": fake_assess}), human_review=True)
            cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
            state = await graph.ainvoke({"facts": pf.model_dump(), "trial": trial.model_dump()}, cfg)
            return graph, cfg, state

        graph, cfg, state = asyncio.run(screen())
        st.session_state.update(graph=graph, cfg=cfg, state=state, patient=patient)
with b:
    st.subheader("Output · evidence-linked screening")
    if st.session_state.get("patient") == patient:
        state = st.session_state.state
        result = MatchResult.model_validate(state["result"])
        st.info(f"Verdict: {result.verdict} · Review: {result.reviewed_by or 'awaiting coordinator'}")
        st.dataframe(
            [
                {
                    "#": x.index,
                    "Criterion": x.criterion,
                    "Status": x.status,
                    "Method": x.method,
                    "Evidence": ", ".join(x.evidence_refs),
                    "Reason": x.rationale,
                }
                for x in result.assessments
            ],
            hide_index=True,
            height=375,
        )
        with st.expander("Inspect criterion evidence and rationale"):
            idx = st.selectbox("Criterion index", [x.index for x in result.assessments])
            ass = next(x for x in result.assessments if x.index == idx)
            st.write(ass.criterion)
            st.write(ass.rationale)
            for eid in ass.evidence:
                if eid in pf.by_id():
                    st.code(pf.by_id()[eid].render(), language=None)
        if not result.reviewed_by:
            st.subheader("Coordinator review · synthetic exercise")
            st.caption("No patient contact. This records a reviewer decision in the local synthetic example.")
            index = st.selectbox("Criterion to override", [x.index for x in result.assessments], index=8)
            status = st.selectbox("Reviewed status", ["met", "not_met", "unknown"])
            note = st.text_input(
                "Review evidence / reason", value="Synthetic review: insulin starts next week; exclusion applies."
            )
            if st.button("Apply synthetic coordinator review"):

                async def review():
                    await st.session_state.graph.aupdate_state(
                        st.session_state.cfg,
                        {"overrides": {str(index): {"status": status, "note": note}}, "reviewer": "Demo coordinator"},
                    )
                    return await st.session_state.graph.ainvoke(None, st.session_state.cfg)

                st.session_state.state = asyncio.run(review())
                st.rerun()
        else:
            st.success(f"Recorded review by {result.reviewed_by} · {result.verdict}")
            st.write(f"FHIR ResearchSubject status: **{state['fhir']['entry'][1]['resource']['status']}**")
            st.download_button("Download screening FHIR bundle", json.dumps(state["fhir"], indent=2), "screening.json")
        st.download_button("Download evidence report", report_markdown(result, pf), "screening.md")
    else:
        st.info("Run pre-screening to extract, assess and pause for coordinator review.")
