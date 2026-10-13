import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def test_actual_workflow_and_review():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "examples"))
    sys.modules.pop("demo_fixture", None)
    a = AppTest.from_file(str(root / "examples/workflow.py"), default_timeout=30).run()
    assert not a.exception, a.exception
    next(b for b in a.button if b.label.startswith(("Run pre", "Review and", "Run simulated"))).click().run()
    assert not a.exception, a.exception
    assert a.session_state.state["result"]["verdict"] == "needs_review"
    next(b for b in a.button if b.label == "Apply synthetic coordinator review").click().run()
    assert not a.exception, a.exception
    assert a.session_state.state["fhir"]["entry"][1]["resource"]["status"] == "ineligible"

    a.selectbox[0].set_value("p3").run()
    assert any("Run pre-screening to" in info.value for info in a.info)
