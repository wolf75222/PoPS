#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec sbatch "$@" "${SCRIPT_DIR}/adc700_program_cutover.sbatch"
