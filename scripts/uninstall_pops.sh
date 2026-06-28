#!/usr/bin/env bash
# Uninstall the Python module `pops` and, by default, the whole `pops` conda env -- the inverse of
# scripts/setup_env.sh (creates the env, pins the toolchain, persists the POPS_* vars) and
# scripts/build_python.sh (pip-installs the module, leaves a build/cp3*/ wheel cache). A bare run is a
# FULL teardown: it removes the in-tree build artifacts AND deletes the conda env (its pinned CC/CXX and
# POPS_INCLUDE / POPS_KOKKOS_ROOT / CMAKE_PREFIX_PATH / POPS_CACHE_DIR go with it). Recreate later with
# `bash scripts/setup_env.sh && bash scripts/build_python.sh`.
#
#   bash scripts/uninstall_pops.sh             # full teardown: artifacts + module + conda env `pops`
#   bash scripts/uninstall_pops.sh --keep-env  # keep env + toolchain; only `pip uninstall pops` + artifacts
#   bash scripts/uninstall_pops.sh --ccache    # ALSO clear the shared ccache (~/.cache/adc-ccache)
#   bash scripts/uninstall_pops.sh --yes       # do not prompt before deleting the env (CI / scripted)
#   POPS_ENV_NAME=myenv bash scripts/uninstall_pops.sh   # target a non-default env name
#
# In-tree artifacts removed on EVERY run: build/cp3*/ (scikit-build wheel cache), .pops_cache (compiled
# DSL .so cache), *.egg-info. The C++ preset build/ root and ~/.cache/adc-ccache are kept unless asked.
#
# NOT `set -u`: the conda shell hook references unset variables.
set -eo pipefail

ENV_NAME="${POPS_ENV_NAME:-pops}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

# --- arguments --------------------------------------------------------------------------------------
KEEP_ENV=0
DO_CCACHE=0
ASSUME_YES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-env) KEEP_ENV=1 ;;
    --ccache)   DO_CCACHE=1 ;;
    -y|--yes)   ASSUME_YES=1 ;;
    -h|--help)
      sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown argument: $1 (use --keep-env | --ccache | --yes | --help)" >&2
       exit 2 ;;
  esac
  shift
done

# --- in-tree build artifacts (no conda needed) ------------------------------------------------------
# scikit-build-core caches under build/<wheel_tag>/ (build/cp312-...); remove ONLY those tag dirs, never
# the C++ preset build/ root (its CMakeCache.txt sits at build/).
shopt -s nullglob
removed=0
for d in "$HERE"/build/cp3*/; do rm -rf "$d"; removed=1; done
[[ $removed -eq 1 ]] && echo "removed scikit-build wheel cache (build/cp3*/)" \
                     || echo "no scikit-build wheel cache (build/cp3*/) to remove"
if [[ -d "$HERE/.pops_cache" ]]; then rm -rf "$HERE/.pops_cache"; echo "removed DSL cache (.pops_cache)"; fi
egg=0
for e in "$HERE"/*.egg-info "$HERE"/python/*.egg-info; do rm -rf "$e"; egg=1; done
[[ $egg -eq 1 ]] && echo "removed *.egg-info"

# --- optional: shared cross-worktree ccache ---------------------------------------------------------
if [[ $DO_CCACHE -eq 1 ]]; then
  if command -v ccache >/dev/null 2>&1; then
    export CCACHE_DIR="${CCACHE_DIR:-$HOME/.cache/adc-ccache}"
    ccache -C >/dev/null && echo "cleared shared ccache ($CCACHE_DIR)" \
                         || echo "ccache -C failed ($CCACHE_DIR)" >&2
  else
    echo "ccache not found; --ccache skipped." >&2
  fi
fi

# --- conda present? otherwise the in-tree clean is all we can do -------------------------------------
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found: in-tree artifacts cleaned, but the module / env were left untouched." >&2
  echo "Load conda (source <base>/etc/profile.d/conda.sh) and re-run to remove the env." >&2
  exit 0
fi
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "conda env '$ENV_NAME' already absent -- nothing more to remove."
  exit 0
fi

# --- module-only: keep the env (and its toolchain + pinned POPS_* vars) ------------------------------
if [[ $KEEP_ENV -eq 1 ]]; then
  echo "--- pip uninstall pops from '$ENV_NAME' (env kept) ---"
  conda run -n "$ENV_NAME" python -m pip uninstall -y pops \
    || echo "note: 'pops' was not pip-installed in '$ENV_NAME'."
  echo "Done. Env + toolchain + pinned POPS_* vars kept. Reinstall: bash scripts/build_python.sh"
  exit 0
fi

# --- full teardown: delete the whole env -------------------------------------------------------------
# conda refuses to delete the ACTIVE env; `conda env remove` operates on it by name without activating.
if [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]]; then
  echo "You are inside '$ENV_NAME'. Run 'conda deactivate' first, then re-run." >&2
  exit 1
fi
if [[ $ASSUME_YES -eq 0 && -t 0 ]]; then
  printf "About to DELETE conda env '%s' (toolchain + pinned vars included). Continue? [y/N] " "$ENV_NAME"
  read -r reply
  case "$reply" in
    [yY]|[yY][eE][sS]) ;;
    *) echo "aborted: env kept (use --keep-env to remove only the module)."; exit 0 ;;
  esac
fi
echo "--- conda env remove -n $ENV_NAME ---"
conda env remove -n "$ENV_NAME" -y
echo ""
echo "env '$ENV_NAME' removed (module, toolchain and pinned POPS_* vars gone)."
echo "Recreate with: bash scripts/setup_env.sh && bash scripts/build_python.sh"
