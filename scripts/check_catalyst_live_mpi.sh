#!/usr/bin/env bash
# End-to-end two-rank Catalyst Live acceptance probe with a real ParaView client.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
PARAVIEW_APP="${POPS_PARAVIEW_ROOT:-${PARAVIEW_ROOT:-}}"
if [[ -z "$PARAVIEW_APP" ]]; then
  shopt -s nullglob
  PARAVIEW_APPS=(/Applications/ParaView*.app)
  shopt -u nullglob
  if [[ ${#PARAVIEW_APPS[@]} -eq 1 ]]; then
    PARAVIEW_APP="${PARAVIEW_APPS[0]}"
  fi
fi
[[ -n "$PARAVIEW_APP" ]] || {
  echo "set POPS_PARAVIEW_ROOT to a ParaView installation" >&2
  exit 2
}
if [[ -d "$PARAVIEW_APP/Contents" ]]; then
  PARAVIEW_CONTENTS="$PARAVIEW_APP/Contents"
else
  PARAVIEW_CONTENTS="$PARAVIEW_APP"
fi
PARAVIEW_CONTENTS="$(cd "$PARAVIEW_CONTENTS" && pwd)"
PVPYTHON="$PARAVIEW_CONTENTS/bin/pvpython"
[[ -x "$PVPYTHON" ]] || { echo "ParaView pvpython is missing: $PVPYTHON" >&2; exit 2; }
[[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]] || {
  echo "activate the PoPS conda environment before this probe" >&2
  exit 2
}

PROBE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pops-catalyst-live.XXXXXX")"
CLIENT_PID=""
SERVER_PID=""
SERVER_PGID=""
SUCCESS=0

server_group_signal() {
  local signal_name="$1"
  [[ -n "$SERVER_PGID" ]] || return 0
  "$CONDA_PREFIX/bin/python" -c '
import os
import signal
import sys

pgid = int(sys.argv[1])
signum = getattr(signal, "SIG" + sys.argv[2])
try:
    os.killpg(pgid, signum)
except ProcessLookupError:
    pass
' "$SERVER_PGID" "$signal_name"
}

wait_server_group_exit() {
  local timeout_seconds="$1"
  [[ -n "$SERVER_PGID" ]] || return 0
  "$CONDA_PREFIX/bin/python" -c '
import os
import sys
import time

pgid = int(sys.argv[1])
deadline = time.monotonic() + float(sys.argv[2])
while True:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        raise SystemExit(0)
    if time.monotonic() >= deadline:
        raise SystemExit(1)
    time.sleep(0.05)
' "$SERVER_PGID" "$timeout_seconds"
}

terminate_server_group() {
  local group_stopped=0
  touch "$PROBE_DIR/abort"
  server_group_signal TERM || true
  if [[ -n "$SERVER_PID" ]]; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
  fi
  if wait_server_group_exit 5; then
    group_stopped=1
  else
    server_group_signal KILL || true
    if [[ -n "$SERVER_PID" ]]; then
      kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
  fi
  if [[ -n "$SERVER_PID" ]]; then
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
  if [[ "$group_stopped" != 1 ]] && wait_server_group_exit 5; then
    group_stopped=1
  fi
  if [[ "$group_stopped" != 1 ]]; then
    echo "Catalyst Live MPI process group $SERVER_PGID did not terminate" >&2
    return 1
  fi
  SERVER_PGID=""
}

cleanup() {
  if [[ -n "$SERVER_PID" || -n "$SERVER_PGID" ]]; then
    terminate_server_group || true
  fi
  if [[ -n "$CLIENT_PID" ]] && kill -0 "$CLIENT_PID" 2>/dev/null; then
    touch "$PROBE_DIR/abort"
    kill "$CLIENT_PID" 2>/dev/null || true
    wait "$CLIENT_PID" 2>/dev/null || true
  fi
  if [[ "$SUCCESS" == 1 ]]; then
    rm -rf "$PROBE_DIR"
  else
    echo "Catalyst Live probe evidence retained in $PROBE_DIR" >&2
  fi
}
trap cleanup EXIT

PORT="$("$CONDA_PREFIX/bin/python" -c \
  'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
"$PVPYTHON" --no-mpi \
  "$HERE/tests/python/integration/mpi/probe_catalyst_live_client.py" \
  --host 127.0.0.1 --port "$PORT" --handshake "$PROBE_DIR" \
  >"$PROBE_DIR/client.log" 2>&1 &
CLIENT_PID=$!

READY=0
for _ in $(seq 1 200); do
  if [[ -f "$PROBE_DIR/client-ready.json" ]]; then
    READY=1
    break
  fi
  if [[ -f "$PROBE_DIR/client-failed.json" ]] || ! kill -0 "$CLIENT_PID" 2>/dev/null; then
    break
  fi
  sleep 0.05
done
if [[ "$READY" != 1 ]]; then
  sed -n '1,240p' "$PROBE_DIR/client.log" >&2
  echo "ParaView Catalyst Live client did not become ready" >&2
  exit 1
fi

export POPS_CATALYST_LIVE_PROBE_DIR="$PROBE_DIR"
export POPS_CATALYST_LIVE_URL="127.0.0.1:$PORT"
SERVER_SESSION_MARKER="$PROBE_DIR/server-session.ready"
SERVER_SESSION_CODE='import os
import pathlib
import sys

marker = pathlib.Path(sys.argv[1])
executable = sys.argv[2]
arguments = sys.argv[2:]
if not hasattr(os, "setsid"):
    print("POSIX session isolation is unavailable", file=sys.stderr)
    raise SystemExit(125)
try:
    os.setsid()
except OSError as error:
    print(f"cannot isolate Catalyst Live MPI session: {error}", file=sys.stderr)
    raise SystemExit(125) from error
pid = os.getpid()
pgid = os.getpgrp()
sid = os.getsid(0)
if pid != pgid or pid != sid:
    print(f"invalid Catalyst Live MPI session: pid={pid} pgid={pgid} sid={sid}", file=sys.stderr)
    raise SystemExit(125)
temporary = marker.with_name(marker.name + f".{pid}.tmp")
temporary.write_text(f"{pid} {pgid} {sid}\n", encoding="ascii")
os.replace(temporary, marker)
os.execv(executable, arguments)
'
"$CONDA_PREFIX/bin/python" -c "$SERVER_SESSION_CODE" \
  "$SERVER_SESSION_MARKER" "$HERE/scripts/paraview_python.sh" \
  --paraview-root "$PARAVIEW_CONTENTS" --mpi 2 \
  "$HERE/tests/python/integration/mpi/probe_catalyst_live_mpi.py" \
  >"$PROBE_DIR/server.log" 2>&1 &
SERVER_PID=$!
SERVER_PGID="$SERVER_PID"

SERVER_SESSION_READY=0
for _ in $(seq 1 200); do
  if [[ -f "$SERVER_SESSION_MARKER" ]]; then
    read -r SESSION_PID SESSION_PGID SESSION_SID <"$SERVER_SESSION_MARKER"
    if [[ "$SESSION_PID" == "$SERVER_PID" \
      && "$SESSION_PGID" == "$SERVER_PID" \
      && "$SESSION_SID" == "$SERVER_PID" ]]; then
      SERVER_SESSION_READY=1
    fi
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  sleep 0.05
done
if [[ "$SERVER_SESSION_READY" != 1 ]]; then
  terminate_server_group || true
  sed -n '1,240p' "$PROBE_DIR/server.log" >&2
  echo "Catalyst Live MPI server could not create a dedicated process session" >&2
  exit 1
fi

SERVER_STATUS=124
for _ in $(seq 1 1200); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    set +e
    wait "$SERVER_PID"
    SERVER_STATUS=$?
    set -e
    SERVER_PID=""
    break
  fi
  sleep 0.1
done
if [[ -n "$SERVER_PID" ]]; then
  terminate_server_group || true
elif ! wait_server_group_exit 5; then
  echo "Catalyst Live MPI launcher exited while its process group remained active" >&2
  SERVER_STATUS=125
  terminate_server_group || true
else
  SERVER_PGID=""
fi

for _ in $(seq 1 200); do
  if ! kill -0 "$CLIENT_PID" 2>/dev/null; then
    break
  fi
  sleep 0.05
done
if kill -0 "$CLIENT_PID" 2>/dev/null; then
  touch "$PROBE_DIR/abort"
  sleep 0.2
  kill "$CLIENT_PID" 2>/dev/null || true
  wait "$CLIENT_PID" 2>/dev/null || true
  CLIENT_STATUS=124
else
  set +e
  wait "$CLIENT_PID"
  CLIENT_STATUS=$?
  set -e
fi
CLIENT_PID=""

if [[ "$SERVER_STATUS" != 0 || "$CLIENT_STATUS" != 0 \
  || ! -f "$PROBE_DIR/client-ready.json" \
  || ! -f "$PROBE_DIR/client-extract-requested.json" \
  || ! -f "$PROBE_DIR/client-frame.json" \
  || ! -f "$PROBE_DIR/client-closed.json" \
  || ! -s "$PROBE_DIR/live-client.png" ]]; then
  sed -n '1,300p' "$PROBE_DIR/server.log" >&2
  sed -n '1,300p' "$PROBE_DIR/client.log" >&2
  echo "Catalyst Live end-to-end probe failed (server=$SERVER_STATUS client=$CLIENT_STATUS)" >&2
  exit 1
fi

sed -n '/^PASS /p' "$PROBE_DIR/server.log"
sed -n '/^PASS /p' "$PROBE_DIR/client.log"
SUCCESS=1
