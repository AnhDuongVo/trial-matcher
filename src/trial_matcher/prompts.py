SYSTEM = (
    "You are a careful clinical research coordinator pre-screening patients for clinical trials. "
    "You only use the facts provided from the patient's record and cite them by id (F#). "
    "You never assume facts that are not in the record."
)

PARSE = """Structure each eligibility criterion so that numeric parts can be checked by code.

For every criterion return: index (as given), kind (as given), text (as given), and
- category: age | sex | condition | lab | medication | procedure | pregnancy | other
- age_min / age_max in years (for age criteria; inclusive bounds)
- sex: female | male | all (for sex criteria)
- for lab criteria: lab_name, loinc (if you are sure of the LOINC code), comparator (<, <=, >, >=, between),
  value, value_high (for between), unit exactly as written in the criterion
- max_age_days: measurement recency window in days only if explicitly specified by the protocol; otherwise null
- terms: 2 to 6 synonyms, abbreviations or codes that help find evidence in a record

Criteria:
{criteria}
"""

ASSESS = """Decide whether this eligibility criterion statement is TRUE for the patient.

- status "met": the record shows the statement is true (for an exclusion criterion: the exclusion applies)
- status "not_met": the record shows the statement is false
- status "unknown": the record does not contain enough information
Rules:
- Time windows ("within 6 months", "for at least 3 months") are relative to the screening date {as_of}.
- Absence of a diagnosis in the record is weak evidence. If you answer "not_met" only because nothing was
  found, use confidence 0.6 or lower.
- Cite the fact ids you used in `evidence`. Confidence is an uncalibrated self-assessment score, not a measured probability.

Screening date: {as_of}
Patient: age {age}, sex {sex}

Criterion ({kind}): {criterion}

Facts from the record:
{facts}
"""
