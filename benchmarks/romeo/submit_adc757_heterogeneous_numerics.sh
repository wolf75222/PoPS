#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec sbatch "$@" "${SCRIPT_DIR}/adc757_heterogeneous_numerics.sbatch"
