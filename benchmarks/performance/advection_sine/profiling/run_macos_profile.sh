#!/usr/bin/env bash
# macOS acquisition protocol for the one canonical public Python command.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPOSITORY_ROOT="$(cd -- "${HARNESS_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
CAMPAIGN="${HARNESS_DIR}/campaigns/strong_openmp.json"
TARGET_PYTHON=""
OUTPUT=""
SAMPLE_TIME_LIMIT_SECONDS="${POPS_MACOS_PROFILE_SAMPLE_TIME_LIMIT_SECONDS:-600}"
TRACE_TIME_LIMIT_SECONDS="${POPS_MACOS_PROFILE_TRACE_TIME_LIMIT_SECONDS:-600}"
READY_TIMEOUT_SECONDS="${POPS_MACOS_PROFILE_READY_TIMEOUT_SECONDS:-1200}"
BUILD_JOBS="${POPS_MACOS_PROFILE_BUILD_JOBS:-4}"
ACTIVE_TARGET_PID=""
ACTIVE_PROFILER_PID=""
ACTIVE_NOTIFY_PID=""

cleanup_active_processes() {
  # Failure after SIGSTOP or READY must not strand a process on the developer Mac.
  local pid
  for pid in "${ACTIVE_TARGET_PID}" "${ACTIVE_PROFILER_PID}" "${ACTIVE_NOTIFY_PID}"; do
    [[ -n "${pid}" ]] || continue
    kill -CONT "${pid}" 2>/dev/null || true
  done
  for pid in "${ACTIVE_NOTIFY_PID}" "${ACTIVE_PROFILER_PID}" "${ACTIVE_TARGET_PID}"; do
    [[ -n "${pid}" ]] || continue
    kill -TERM "${pid}" 2>/dev/null || true
  done
  for pid in "${ACTIVE_NOTIFY_PID}" "${ACTIVE_PROFILER_PID}" "${ACTIVE_TARGET_PID}"; do
    [[ -n "${pid}" ]] || continue
    wait "${pid}" 2>/dev/null || true
  done
}
trap cleanup_active_processes EXIT INT TERM

usage() {
  echo "usage: run_macos_profile.sh --python PYTHON --output NEW_DIRECTORY [--campaign JSON]" >&2
  exit 2
}

while (( $# )); do
  case "$1" in
    --python) shift; TARGET_PYTHON="${1:-}" ;;
    --output) shift; OUTPUT="${1:-}" ;;
    --campaign) shift; CAMPAIGN="${1:-}" ;;
    *) usage ;;
  esac
  shift
done
[[ -n "${TARGET_PYTHON}" && -n "${OUTPUT}" ]] || usage
[[ -x "${TARGET_PYTHON}" ]] || { echo "profile Python must be executable" >&2; exit 2; }
[[ -x /usr/bin/xctrace && -x /usr/bin/sample && -x /usr/bin/notifyutil ]] || {
  echo "macOS profiling requires xctrace, sample and notifyutil in /usr/bin" >&2; exit 2; }
[[ ! -e "${OUTPUT}" ]] || { echo "refusing existing output ${OUTPUT}" >&2; exit 2; }
[[ "${SAMPLE_TIME_LIMIT_SECONDS}" =~ ^[1-9][0-9]*$ && "${SAMPLE_TIME_LIMIT_SECONDS}" -ge 60 ]] || {
  echo "sample time limit must be an integer of at least 60 seconds" >&2; exit 2; }
[[ "${TRACE_TIME_LIMIT_SECONDS}" =~ ^[1-9][0-9]*$ && "${TRACE_TIME_LIMIT_SECONDS}" -ge 60 ]] || {
  echo "xctrace time limit must be an integer of at least 60 seconds" >&2; exit 2; }
[[ "${READY_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ && "${READY_TIMEOUT_SECONDS}" -ge 300 ]] || {
  echo "READY timeout must be an integer of at least 300 seconds for the full 128^3 case" >&2; exit 2; }
[[ "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ && "${BUILD_JOBS}" -le 32 ]] || {
  echo "native build jobs must be an integer between 1 and 32" >&2; exit 2; }

"${PYTHON}" -B -c \
  'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[2]); from profile_contract import canonical_plan; canonical_plan(Path(sys.argv[1]))' \
  "${CAMPAIGN}" "${SCRIPT_DIR}"
# The one output root is published exclusively: a concurrent creator or a
# symlink destination is refused by mkdir(2), rather than merged with evidence.
mkdir -- "${OUTPUT}"
SOURCE_ARCHIVE="${OUTPUT}/source-export.tar"
SOURCE_MANIFEST="${OUTPUT}/source.manifest.json"
EXPORTED_ROOT="${OUTPUT}/source-tree"
CAMPAIGN_REL="$(${PYTHON} -B -c \
  'import sys; from pathlib import Path; print(Path(sys.argv[1]).resolve().relative_to(Path(sys.argv[2]).resolve()).as_posix())' \
  "${CAMPAIGN}" "${REPOSITORY_ROOT}")"
"${PYTHON}" -B "${HARNESS_DIR}/prepare_export.py" create --source "${REPOSITORY_ROOT}" \
  --archive "${SOURCE_ARCHIVE}" --manifest "${SOURCE_MANIFEST}" >/dev/null
mkdir "${EXPORTED_ROOT}"
/usr/bin/tar -xf "${SOURCE_ARCHIVE}" -C "${EXPORTED_ROOT}"
"${PYTHON}" -B "${EXPORTED_ROOT}/benchmarks/performance/advection_sine/prepare_export.py" \
  verify-tree --source "${EXPORTED_ROOT}" --manifest "${SOURCE_MANIFEST}" >/dev/null
CAMPAIGN="${EXPORTED_ROOT}/${CAMPAIGN_REL}"
PROFILE_LIB_DIR="${EXPORTED_ROOT}/benchmarks/performance/advection_sine/profiling"
PROFILE_CACHE_DIR="${OUTPUT}/pops-cache"
PROFILE_XDG_CACHE_DIR="${OUTPUT}/xdg-cache"
mkdir -- "${PROFILE_CACHE_DIR}" "${PROFILE_XDG_CACHE_DIR}"
export POPS_INCLUDE="${EXPORTED_ROOT}/include"
export POPS_CACHE_DIR="${PROFILE_CACHE_DIR}"
export XDG_CACHE_HOME="${PROFILE_XDG_CACHE_DIR}"
[[ -d "${POPS_INCLUDE}" ]] || { echo "exported include directory is unavailable" >&2; exit 2; }
SOURCE_TREE_SHA256="$(${PYTHON} -B -c \
  'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[2]); from profile_contract import read_json; print(read_json(Path(sys.argv[1]), "source manifest")["tree_sha256"])' \
  "${SOURCE_MANIFEST}" "${PROFILE_LIB_DIR}")"

# Build the native OpenMP variant from this run's export.  An earlier external
# build cannot possibly name a freshly created source-tree and is refused by
# construction rather than requested from the caller.  The build receipt below
# verifies CMAKE_HOME_DIRECTORY against EXPORTED_ROOT before acquisition.
MACOS_BUILD_ROOT="${OUTPUT}/native-build"
MACOS_KOKKOS_ROOT="${POPS_MACOS_PROFILE_KOKKOS_ROOT:-}"
[[ -n "${MACOS_KOKKOS_ROOT}" && -d "${MACOS_KOKKOS_ROOT}" ]] || {
  echo "macOS profiling requires POPS_MACOS_PROFILE_KOKKOS_ROOT for an authenticated build receipt" >&2; exit 2; }
MACOS_CMAKE="${POPS_MACOS_PROFILE_CMAKE:-$(command -v cmake || true)}"
MACOS_NINJA="${POPS_MACOS_PROFILE_NINJA:-$(command -v ninja || true)}"
MACOS_CXX="${POPS_MACOS_PROFILE_CXX:-$(command -v c++ || true)}"
for tool in "${MACOS_CMAKE}" "${MACOS_NINJA}" "${MACOS_CXX}"; do
  [[ "${tool}" = /* && -x "${tool}" ]] || {
    echo "macOS profiling requires executable absolute CMake, Ninja and C++ compiler paths" >&2; exit 2; }
done
"${MACOS_CMAKE}" -S "${EXPORTED_ROOT}" -B "${MACOS_BUILD_ROOT}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM="${MACOS_NINJA}" \
  -DCMAKE_CXX_COMPILER="${MACOS_CXX}" -DPOPS_NATIVE_DIM=3 \
  -DPOPS_BUILD_PYTHON=ON -DPOPS_BUILD_TESTS=OFF \
  -DPOPS_USE_KOKKOS=ON -DPOPS_USE_MPI=OFF \
  -DPOPS_USE_HDF5=OFF -DPython_EXECUTABLE="${TARGET_PYTHON}" \
  -DKokkos_ROOT="${MACOS_KOKKOS_ROOT}"
"${MACOS_CMAKE}" --build "${MACOS_BUILD_ROOT}" --target _pops --parallel "${BUILD_JOBS}"
NATIVE_VARIANTS_ROOT="${MACOS_BUILD_ROOT}/python/pops/_native"
[[ -f "${NATIVE_VARIANTS_ROOT}/variants.json" && ! -L "${NATIVE_VARIANTS_ROOT}/variants.json" ]] || {
  echo "macOS native build did not publish an authenticated variants.json" >&2; exit 2; }
BUILD_RECEIPT="${OUTPUT}/build.receipt.json"
PYTHONPATH="${MACOS_BUILD_ROOT}/python:${EXPORTED_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}" \
  POPS_NATIVE_VARIANTS_ROOT="${NATIVE_VARIANTS_ROOT}" \
  "${PYTHON}" -B "${EXPORTED_ROOT}/benchmarks/performance/advection_sine/prepare_export.py" build-receipt \
  --source "${EXPORTED_ROOT}" --build "${MACOS_BUILD_ROOT}" --manifest "${SOURCE_MANIFEST}" \
  --campaign "${CAMPAIGN}" --python "${TARGET_PYTHON}" --kokkos-root "${MACOS_KOKKOS_ROOT}" \
  --output "${BUILD_RECEIPT}" >/dev/null
export POPS_MACOS_PROFILE_BUILD_RECEIPT="${BUILD_RECEIPT}"

unix_time() {
  "${PYTHON}" -B -c 'import time; print(time.time())'
}

wait_ready() {
  local ready="$1" timeout=0 limit=$(( READY_TIMEOUT_SECONDS * 50 ))
  while [[ ! -f "${ready}" ]]; do
    (( timeout < limit )) || { echo "target never published READY within ${READY_TIMEOUT_SECONDS}s" >&2; return 1; }
    sleep 0.02
    ((timeout += 1))
  done
}

write_canonical_command() {
  "${PYTHON}" -B -c \
    'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[6]); from profile_contract import PROFILE_SCHEMA, command_sha256, profile_command, write_json_new; campaign=Path(sys.argv[1]).resolve(); output=Path(sys.argv[3]).resolve(); argv=profile_command(campaign_path=campaign, python=Path(sys.argv[2]), output_dir=output, source_root=Path(sys.argv[5])); digest=command_sha256(argv); write_json_new(Path(sys.argv[4]), {"schema":PROFILE_SCHEMA, "phase":"canonical_command", "argv":argv, "sha256":digest, "output_dir":str(output)}); print(digest); print(*argv, sep="\n")' \
    "$1" "$2" "$3" "$4" "$5" "${PROFILE_LIB_DIR}"
}

check_ready() {
  "${PYTHON}" -B -c \
    'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[7]); from profile_contract import PROFILE_SCHEMA, command_sha256, read_json, sha256, source_manifest_receipt; ready=read_json(Path(sys.argv[1]), "READY"); command=read_json(Path(sys.argv[4]), "canonical command"); campaign=Path(sys.argv[3]).resolve(); source_root=Path(sys.argv[6]); required={"campaign","source","python_package","artifact","program_artifact","native","build","runtime","host","command"}; provenance=ready.get("provenance"); source=source_manifest_receipt(manifest_path=Path(sys.argv[5]), source_root=source_root); expected_python={"path":"python/pops/__init__.py", "sha256":sha256(source_root / "python/pops/__init__.py")}; program=provenance.get("program_artifact"); assert ready.get("schema") == PROFILE_SCHEMA and ready.get("phase") == "ready_after_bind_warmup" and ready.get("nonce") == sys.argv[2] and type(ready.get("pid")) is int; assert type(provenance) is dict and set(provenance) == required and all(type(provenance[key]) is dict for key in required); assert provenance["campaign"]["path"] == str(campaign) and provenance["campaign"]["sha256"] == sha256(campaign) and provenance["source"] == source and provenance["python_package"] == expected_python; assert type(provenance["artifact"].get("semantic_identity")) is str and provenance["artifact"]["semantic_identity"]; assert type(program.get("artifact_identity")) is str and program["artifact_identity"] and type(program.get("abi_key")) is str and program["abi_key"] and type(program.get("cache_key")) is str and program["cache_key"] and type(program.get("programs")) is list and program["programs"]; assert type(provenance["native"].get("sha256")) is str and provenance["native"].get("has_kokkos") is True and provenance["build"].get("source_tree_sha256") == source["tree_sha256"] and provenance["build"].get("native_sha256") == provenance["native"]["sha256"]; assert provenance["runtime"] and provenance["host"]; assert command.get("schema") == PROFILE_SCHEMA and command.get("phase") == "canonical_command" and type(command.get("argv")) is list and command.get("sha256") == command_sha256(command["argv"]); assert provenance["command"] == {"argv":command["argv"], "sha256":command["sha256"], "output_dir":command["output_dir"]}; print(ready["pid"])' \
    "$1" "$2" "$3" "$4" "$5" "$6" "${PROFILE_LIB_DIR}"
}

write_go() {
  "${PYTHON}" -B -c \
    'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[3]); from profile_contract import PROFILE_SCHEMA, write_json_new; write_json_new(Path(sys.argv[1]), {"schema": PROFILE_SCHEMA, "phase":"go", "nonce":sys.argv[2]})' \
    "$1" "$2" "${PROFILE_LIB_DIR}"
}

write_tool_receipt() {
  "${PYTHON}" -B -c \
    'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[10]); from profile_contract import PROFILE_SCHEMA, write_json_new; start=float(sys.argv[4]); end=float(sys.argv[5]); completed=float(sys.argv[6]); target_reaped=float(sys.argv[7]); profiler_reaped=float(sys.argv[8]); assert start <= completed <= target_reaped <= profiler_reaped == end; write_json_new(Path(sys.argv[1]), {"schema":PROFILE_SCHEMA, "phase":"acquisition_complete", "tool":sys.argv[2], "nonce":sys.argv[3], "started_unix_seconds":start, "ended_unix_seconds":end, "target_completed_unix_seconds":completed, "target_reaped_unix_seconds":target_reaped, "profiler_reaped_unix_seconds":profiler_reaped, "target_completed_during_acquisition":True, "attachment_proof":sys.argv[9]})' \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${PROFILE_LIB_DIR}"
}

write_profiler_exit() {
  "${PYTHON}" -B -c \
    'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[5]); from profile_contract import PROFILE_SCHEMA, write_json_new; write_json_new(Path(sys.argv[1]), {"schema":PROFILE_SCHEMA, "phase":"profiler_exit", "tool":sys.argv[2], "returncode":int(sys.argv[3]), "exited_unix_seconds":float(sys.argv[4])})' \
    "$1" "$2" "$3" "$4" "${PROFILE_LIB_DIR}"
}

read_profiler_exit() {
  "${PYTHON}" -B -c \
    'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[3]); from profile_contract import PROFILE_SCHEMA, read_json; row=read_json(Path(sys.argv[1]), "profiler exit"); assert row == {"schema":PROFILE_SCHEMA,"phase":"profiler_exit","tool":sys.argv[2],"returncode":0,"exited_unix_seconds":row.get("exited_unix_seconds")}; assert type(row["exited_unix_seconds"]) in (int,float); print(row["exited_unix_seconds"])' \
    "$1" "$2" "${PROFILE_LIB_DIR}"
}

wait_for_notification() {
  local notify_pid="$1" profiler_pid="$2" timeout=0
  while kill -0 "${notify_pid}" 2>/dev/null; do
    kill -0 "${profiler_pid}" 2>/dev/null || {
      echo "xctrace failed before tracing-started notification" >&2; return 1; }
    (( timeout < 3000 )) || { echo "xctrace never notified tracing start" >&2; return 1; }
    sleep 0.02
    ((timeout += 1))
  done
  wait "${notify_pid}"
}

wait_stopped() {
  local target_pid="$1" timeout=0 state
  while :; do
    state="$(/bin/ps -o state= -p "${target_pid}" 2>/dev/null || true)"
    [[ "${state}" == *T* ]] && return 0
    (( timeout < 500 )) || { echo "target did not stop before sample attach" >&2; return 1; }
    sleep 0.02
    ((timeout += 1))
  done
}

wait_sample_header() {
  local report="$1" profiler_pid="$2" timeout=0
  while :; do
    kill -0 "${profiler_pid}" 2>/dev/null || { echo "sample exited before attach proof" >&2; return 1; }
    if [[ -s "${report}" ]] && /usr/bin/grep -Eq 'Sampling process|Analysis of sampling|Call graph:' "${report}"; then
      return 0
    fi
    (( timeout < 3000 )) || { echo "sample did not publish an attach header" >&2; return 1; }
    sleep 0.02
    ((timeout += 1))
  done
}

run_one() {
  local tool="$1" rep="$2" leaf rank_output command_manifest nonce ready go receipt target_pid profiler_pid notify_pid
  local acquisition_start acquisition_end completed_time target_reaped_time profiler_reaped_time notification command_digest attachment_proof profiler_exit
  local -a command_lines command
  leaf="${OUTPUT}/${tool}/$(printf 'rep%02d' "${rep}")"
  rank_output="${leaf}/rank-output"
  mkdir -- "${leaf}"
  command_manifest="${leaf}/command.json"
  command_lines=()
  while IFS= read -r line; do
    command_lines+=("${line}")
  done < <(write_canonical_command "${CAMPAIGN}" "${TARGET_PYTHON}" "${rank_output}" "${command_manifest}" "${EXPORTED_ROOT}")
  command_digest="${command_lines[0]:-}"
  command=("${command_lines[@]:1}")
  [[ "${command_digest}" =~ ^[0-9a-f]{64}$ && "${#command[@]}" -gt 1 ]] || {
    echo "could not construct canonical public command" >&2; return 1; }
  nonce="$(${PYTHON} -B -c 'import secrets; print(secrets.token_hex(32))')"
  ready="${leaf}/ready.json"; go="${leaf}/go.json"; receipt="${leaf}/worker.receipt.json"
  profiler_exit="${leaf}/profiler.exit.json"
  POPS_MACOS_PROFILE_READY="${ready}" POPS_MACOS_PROFILE_GO="${go}" \
  POPS_MACOS_PROFILE_NONCE="${nonce}" POPS_MACOS_PROFILE_RECEIPT="${receipt}" \
  POPS_MACOS_PROFILE_CAMPAIGN_PATH="${CAMPAIGN}" POPS_MACOS_PROFILE_COMMAND_SHA256="${command_digest}" \
  POPS_MACOS_PROFILE_SOURCE_MANIFEST="${SOURCE_MANIFEST}" POPS_MACOS_PROFILE_SOURCE_ROOT="${EXPORTED_ROOT}" \
  POPS_MACOS_PROFILE_SOURCE_TREE_SHA256="${SOURCE_TREE_SHA256}" \
  POPS_NATIVE_VARIANTS_ROOT="${NATIVE_VARIANTS_ROOT}" \
  PYTHONPATH="${EXPORTED_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}" \
  OMP_PROC_BIND="spread" OMP_PLACES="cores" OMP_DYNAMIC="false" \
  "${command[@]}" &
  target_pid=$!
  ACTIVE_TARGET_PID="${target_pid}"
  wait_ready "${ready}"
  [[ "$(check_ready "${ready}" "${nonce}" "${CAMPAIGN}" "${command_manifest}" "${SOURCE_MANIFEST}" "${EXPORTED_ROOT}")" == "${target_pid}" ]] || {
    echo "READY pid is not the canonical public process" >&2; kill "${target_pid}"; return 1; }
  if [[ "${tool}" == xctrace ]]; then
    notification="pops.advection-sine.${nonce}"
    /usr/bin/notifyutil -w "${notification}" >"${leaf}/notify.stdout" 2>"${leaf}/notify.stderr" &
    notify_pid=$!
    ACTIVE_NOTIFY_PID="${notify_pid}"
    acquisition_start="$(unix_time)"
    ( set +e; /usr/bin/xctrace record --template 'Time Profiler' --attach "${target_pid}" \
      --output "${leaf}/time-profiler.trace" --quiet --no-prompt \
      --time-limit "${TRACE_TIME_LIMIT_SECONDS}s" --notify-tracing-started "${notification}" \
      >"${leaf}/xctrace.stdout" 2>"${leaf}/xctrace.stderr"; profiler_rc=$?; write_profiler_exit "${profiler_exit}" xctrace "${profiler_rc}" "$(unix_time)"; exit "${profiler_rc}" ) &
    profiler_pid=$!
    ACTIVE_PROFILER_PID="${profiler_pid}"
    wait_for_notification "${notify_pid}" "${profiler_pid}"
    ACTIVE_NOTIFY_PID=""
    write_go "${go}" "${nonce}"
    attachment_proof="xctrace_notify_before_go"
    wait "${target_pid}"
    ACTIVE_TARGET_PID=""
    target_reaped_time="$(unix_time)"
    wait "${profiler_pid}"
    ACTIVE_PROFILER_PID=""
    profiler_reaped_time="$(read_profiler_exit "${profiler_exit}" xctrace)"
    acquisition_end="${profiler_reaped_time}"
    /usr/bin/xctrace export --input "${leaf}/time-profiler.trace" --toc > "${leaf}/toc.txt"
  else
    kill -STOP "${target_pid}"
    if ! wait_stopped "${target_pid}"; then kill -CONT "${target_pid}"; return 1; fi
    acquisition_start="$(unix_time)"
    ( set +e; /usr/bin/sample "${target_pid}" "${SAMPLE_TIME_LIMIT_SECONDS}" -mayDie -fullPaths \
      -file "${leaf}/sample.txt" >"${leaf}/sample.stdout" 2>"${leaf}/sample.stderr"; profiler_rc=$?; write_profiler_exit "${profiler_exit}" sample "${profiler_rc}" "$(unix_time)"; exit "${profiler_rc}" ) &
    profiler_pid=$!
    ACTIVE_PROFILER_PID="${profiler_pid}"
    if ! wait_sample_header "${leaf}/sample.txt" "${profiler_pid}"; then
      kill -CONT "${target_pid}"
      return 1
    fi
    if ! write_go "${go}" "${nonce}"; then kill -CONT "${target_pid}"; return 1; fi
    kill -CONT "${target_pid}"
    attachment_proof="sample_header_before_go_after_stop"
    wait "${target_pid}"
    ACTIVE_TARGET_PID=""
    target_reaped_time="$(unix_time)"
    wait "${profiler_pid}"
    ACTIVE_PROFILER_PID=""
    profiler_reaped_time="$(read_profiler_exit "${profiler_exit}" sample)"
    acquisition_end="${profiler_reaped_time}"
  fi
  [[ -s "${receipt}" ]] || { echo "target did not publish completed receipt" >&2; return 1; }
  completed_time="$(${PYTHON} -B -c \
    'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[2]); from profile_contract import read_json; row=read_json(Path(sys.argv[1]), "worker receipt"); assert row.get("phase") == "completed_public_lifecycle" and row.get("nonce") == sys.argv[3] and row.get("returncode") == 0; print(row["completed_unix_seconds"])' \
    "${receipt}" "${PROFILE_LIB_DIR}" "${nonce}")"
  write_tool_receipt "${leaf}/tool.receipt.json" "${tool}" "${nonce}" \
    "${acquisition_start}" "${acquisition_end}" "${completed_time}" "${target_reaped_time}" "${profiler_reaped_time}" "${attachment_proof}"
}

for tool in sample xctrace; do
  for repetition in 1 2 3 4 5; do
    run_one "${tool}" "${repetition}"
  done
done

"${PYTHON}" -B "${SCRIPT_DIR}/collect_profiles.py" --input "${OUTPUT}" --output "${OUTPUT}/summary.json"
"${PYTHON}" -B -c \
  'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[2]); from profile_contract import create_profile_complete_receipt; create_profile_complete_receipt(Path(sys.argv[1]))' \
  "${OUTPUT}" "${PROFILE_LIB_DIR}"
FIGURES_OUTPUT="$("${PYTHON}" -B -c \
  'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[2]); from profile_contract import profile_figure_publication_path; print(profile_figure_publication_path(Path(sys.argv[1])))' \
  "${OUTPUT}" "${PROFILE_LIB_DIR}")"
"${PYTHON}" -B "${SCRIPT_DIR}/plot_profiles.py" "${OUTPUT}/summary.json" --output "${FIGURES_OUTPUT}"
"${PYTHON}" -B -c \
  'import sys; from pathlib import Path; sys.path.insert(0, sys.argv[2]); from profile_contract import verify_profile_complete_receipt; verify_profile_complete_receipt(Path(sys.argv[1]))' \
  "${OUTPUT}" "${PROFILE_LIB_DIR}"
