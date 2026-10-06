"""NeMo Agent Toolkit plugin: `trial_prescreen` takes "patient_bundle_path | trial (NCT id or JSON path) | as_of"."""

from __future__ import annotations

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig


class TrialPrescreenConfig(FunctionBaseConfig, name="trial_prescreen"):
    llm_name: LLMRef


@register_function(config_type=TrialPrescreenConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def trial_prescreen(config: TrialPrescreenConfig, builder: Builder):
    from ..fhir_facts import extract_facts, load_bundle
    from ..llm import LangChainAdapter
    from ..matcher import as_of_date, match, report_markdown
    from ..trials import load_trial

    model = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    llm = LangChainAdapter(model, model)

    async def _run(request: str) -> str:
        """Pre-screen a patient for a trial. Input: 'path/to/patient_bundle.json | NCT01234567 | 2026-10-01'."""
        parts = [p.strip() for p in request.split("|")]
        pf = extract_facts(load_bundle(parts[0]), as_of_date(parts[2] if len(parts) > 2 else None))
        result, _ = await match(pf, load_trial(parts[1]), llm)
        return report_markdown(result, pf)

    yield FunctionInfo.from_fn(_run, description=_run.__doc__)
