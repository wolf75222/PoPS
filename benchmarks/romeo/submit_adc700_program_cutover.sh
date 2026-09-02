#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${POPS_ADC700_REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd -- "${POPS_ADC700_REPO_ROOT}" && pwd)"
else
  REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
fi
export POPS_ADC700_REPO_ROOT="${REPO_ROOT}"
exec sbatch "$@" "${SCRIPT_DIR}/adc700_program_cutover.sbatch"
