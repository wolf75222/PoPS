#!/usr/bin/env bash
# Attach one scheduler-selected GPU UUID to exactly one measurement process.
set -euo pipefail

device_selector="${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES is not set}"
if [[ "${device_selector}" == *,* ]]; then
  echo "ADC-700 requires one CUDA_VISIBLE_DEVICES entry per rank: ${device_selector}" >&2
  exit 4
fi
gpu_uuid="$(nvidia-smi --id="${device_selector}" --query-gpu=uuid --format=csv,noheader \
  | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
if [[ -z "${gpu_uuid}" || "${gpu_uuid}" == *$'\n'* || "${gpu_uuid}" == *$'\r'* ]]; then
  echo "ADC-700 could not authenticate one GPU UUID for ${device_selector}" >&2
  exit 4
fi
export POPS_ADC700_GPU_UUID="${gpu_uuid}"
exec "$@"
