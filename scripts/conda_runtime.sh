#!/usr/bin/env bash
# Shared, side-effect-free conda discovery for the official PoPS shell workflows.
#
# Sourcing this file defines ``pops_load_conda``. The function never starts a login shell and
# therefore cannot let user startup files replace the selected conda installation.

pops_conda_executable() {
  local candidate
  if [[ -n "${POPS_CONDA_EXE:-}" ]]; then
    if [[ ! -x "$POPS_CONDA_EXE" ]]; then
      echo "POPS_CONDA_EXE is not executable: $POPS_CONDA_EXE" >&2
      return 1
    fi
    printf '%s\n' "$POPS_CONDA_EXE"
    return 0
  fi
  if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
    printf '%s\n' "$CONDA_EXE"
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi
  for candidate in \
      "$HOME/miniforge3/bin/conda" \
      "$HOME/mambaforge/bin/conda" \
      "$HOME/anaconda3/bin/conda" \
      "$HOME/miniconda3/bin/conda"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

pops_load_conda() {
  local executable base profile
  executable="$(pops_conda_executable)" || return 1
  base="$("$executable" info --base)" || return 1
  profile="$base/etc/profile.d/conda.sh"
  if [[ ! -r "$profile" ]]; then
    echo "conda shell integration is missing: $profile" >&2
    return 1
  fi
  # shellcheck source=/dev/null
  source "$profile"
  command -v conda >/dev/null 2>&1
}
