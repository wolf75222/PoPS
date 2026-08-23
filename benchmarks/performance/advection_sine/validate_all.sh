#!/usr/bin/env bash

set -euo pipefail

HARNESS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${POPS_PERF_VALIDATION_PYTHON:-$(command -v python3)}"
test -x "${PYTHON}"
"${PYTHON}" -c 'import sys; assert sys.version_info >= (3, 10)'

"${PYTHON}" - "${HARNESS_DIR}" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from common import validate_canonical_campaign_inventory

validate_canonical_campaign_inventory(root / "campaigns")
print("valid canonical inventory: exactly seven full campaigns")
PY

for campaign in "${HARNESS_DIR}"/campaigns/*.json; do
  "${PYTHON}" "${HARNESS_DIR}/validate_campaign.py" "${campaign}" >/dev/null
  printf 'valid campaign: %s\n' "$(basename -- "${campaign}")"
done

for script in "${HARNESS_DIR}"/validate_all.sh "${HARNESS_DIR}"/slurm/*.sh "${HARNESS_DIR}"/slurm/*.sbatch "${HARNESS_DIR}"/profiling/*.sh; do
  bash -n "${script}"
done

"${PYTHON}" - "${HARNESS_DIR}" <<'PY'
import ast
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for path in sorted((*root.glob("*.py"), *(root / "profiling").glob("*.py"))):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"valid Python syntax: {path.name}")
PY

echo "validation complete: no benchmark process or SLURM job was launched"
