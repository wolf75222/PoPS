#!/usr/bin/env bash
# Shared Linux CI watchdog and progress reporting. Source before run_with_heartbeat.
# Keep per-lane time budgets and parallelism at each call site.

run_with_heartbeat() {
  local label="$1"
  local limit="$2"
  shift 2
  local started=$SECONDS
  echo "::notice::${label}: watchdog=${limit}"
  timeout --signal=TERM --kill-after=30s "$limit" "$@" &
  local command_pid=$!
  (
    sleep_pid=""
    trap 'test -z "$sleep_pid" || kill "$sleep_pid" 2>/dev/null || true' EXIT
    trap 'exit 0' TERM INT
    while kill -0 "$command_pid" 2>/dev/null; do
      sleep 60 &
      sleep_pid=$!
      wait "$sleep_pid" || exit 0
      sleep_pid=""
      kill -0 "$command_pid" 2>/dev/null || exit 0
      mem_available_mib=$(awk '/MemAvailable:/ {printf "%.0f", $2 / 1024}' /proc/meminfo)
      active_compilers=$(pgrep -c -x cc1plus || true)
      echo "::notice::${label}: alive elapsed=$((SECONDS - started))s mem_available=${mem_available_mib}MiB cc1plus=${active_compilers} load=$(cut -d' ' -f1-3 /proc/loadavg)"
    done
  ) &
  local heartbeat_pid=$!
  local previous_term previous_int cancellation=0
  previous_term=$(trap -p TERM)
  previous_int=$(trap -p INT)
  # A cancelled step shell must stop its watchdog group and heartbeat before returning.
  # GNU timeout forwards TERM to the command and retains its kill-after deadline.
  trap 'cancellation=143; kill "$command_pid" 2>/dev/null || true' TERM
  trap 'cancellation=130; kill "$command_pid" 2>/dev/null || true' INT
  local status=0
  wait "$command_pid" || status=$?
  if [ "$cancellation" -ne 0 ]; then
    wait "$command_pid" 2>/dev/null || true
    status=$cancellation
  fi
  kill "$heartbeat_pid" 2>/dev/null || true
  wait "$heartbeat_pid" 2>/dev/null || true
  trap - TERM INT
  # trap -p returns shell-quoted shell code, preserving any caller handlers exactly.
  eval "$previous_term"
  eval "$previous_int"
  if [ "$status" -eq 124 ]; then
    echo "::error::${label} exceeded its ${limit} watchdog"
  elif [ "$status" -eq 130 ] || [ "$status" -eq 137 ] || [ "$status" -eq 143 ]; then
    echo "::error::${label} was terminated by signal (status=${status})"
  fi
  return "$status"
}
