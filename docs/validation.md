# Validation evidence — 9 October 2026

`pip install -e ".[dev]"` succeeded in a fresh virtual environment for this repository, independently of the other projects. Tests ran on Python 3.12.14 / Darwin arm64 CPU. Other Python versions in CI have not been executed locally here.

| Check | Result |
|---|---|
| Offline test suite | 21 passed |
| Ruff lint | Passed |
| Ruff formatting | Passed |
| Mocked/simulated integration | Tested within the scope below |
| Live NVIDIA hosted endpoint | Not executed |
| Self-hosted GPU endpoint | Not executed |
| Clinical/scientific domain validation | Not completed |

## Tested scope

Missing/incompatible units, invalid or stale/future dates, entered-in-error observations, corrected/tied conflicting measurements, protocol recency, unsupported quantitative criteria returning unknown, explicit coordinator overrides.

The GitHub workflows have been added or retained, but their remote execution has not been verified after these changes. Unit tests establish behavior on fixtures; they do not establish semantic or clinical correctness.

## NAT configuration

`nat validate --config_file configs/workflow.yml` passed with the installed toolkit. This checks configuration/schema validity and plugin discovery; it does not execute a live model workflow or verify the example model IDs.

## Small offline evaluation

The scripted parser and assessor were run on three synthetic patients against SYN-T2D-001 (30 criteria); legacy gold labels were retained.

| Metric | Result |
|---|---|
| Criterion agreement | 70.0% |
| Decided coverage | 63.3% |
| Agreement when decided | 100% (19 decisions) |
| Verdict agreement | 33.3% (1 / 3) |
| Macro F1 | 0.641 |
| Total local evaluation time | 21.103 ms on the CPU above |
| Live model / clinical calibration | Not run |

Two legacy verdicts now become `needs_review`: unsupported dose/duration and compound criteria abstain instead of relying on model arithmetic. This loss of agreement is reported, not hidden by changing the labels. “Unknown” precision is 18.2% against those legacy labels. Scripted qualitative assessments limit the usefulness of the other metrics. Calibration output describes this fixture and does not validate confidence probabilities. [Results, errors, class metrics and confusion table](offline-evaluation.json).

## Re-review follow-up, 10 October 2026

Regression coverage now includes ordinary age/laboratory comparative phrases and true compound clauses; 31 tests passed locally.
