#!/usr/bin/env bash
# Generate realistic synthetic FHIR patients with Synthea (https://github.com/synthetichealth/synthea).
# Needs Java 17+. Output: ./synthea/output/fhir/*.json (one Bundle per patient).
#
#   bash scripts/get_synthea.sh 50          # 50 patients
#   ctm facts synthea/output/fhir/<file>.json
#   ctm match synthea/output/fhir/<file>.json NCT0XXXXXXX
#
# Check `java -jar synthea-with-dependencies.jar --help` for all options (state, age range, modules).
set -euo pipefail
N="${1:-20}"
mkdir -p synthea && cd synthea
if [ ! -f synthea-with-dependencies.jar ]; then
  curl -L -o synthea-with-dependencies.jar \
    https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar
fi
# -p population size; -a age range. Diabetes is part of every adult's simulated life in Synthea (the
# metabolic syndrome modules), so a 40-80 age range gives many T2D patients. To restrict modules, use
# -m with module file names from src/main/resources/modules in the Synthea repo, e.g. -m "metabolic_syndrome*".
java -jar synthea-with-dependencies.jar -p "$N" -a 40-80 Massachusetts
ls output/fhir | head
