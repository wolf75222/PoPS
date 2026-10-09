#!/bin/bash
set -euo pipefail
umask 077
: "${FUTURE_GPU_ROOT:?Root supplies admitted private namespace}"
ROOT="$FUTURE_GPU_ROOT"
finish_r5_receipts() {
 local body_status=$?
 set +e
 /usr/bin/python3 "$ROOT/registry_capture.py" "$ROOT" after-r5-probe-finally
 local registry_status=$?
 /usr/bin/python3 "$ROOT/namespace_footprint.py" "$ROOT" r5-probe-finally
 local footprint_status=$?
 /usr/bin/python3 "$ROOT/quota_guard_resume.py" "$ROOT" after-r5-probe-finally 0
 local quota_status=$?
 /usr/bin/python3 - "$ROOT" "$body_status" "$registry_status" "$footprint_status" "$quota_status" <<'PY_STATUS'
import json,sys
from pathlib import Path
with(Path(sys.argv[1])/'results/r5-probe-finally-statuses.json').open('x')as f:
 json.dump(dict(zip(('body','registry','footprint','quota'),map(int,sys.argv[2:]))),f,indent=2,sort_keys=True);f.write(chr(10))
PY_STATUS
 local receipt_status=$?
 if ((body_status != 0)); then exit "$body_status"; fi
 if ((registry_status != 0)); then exit "$registry_status"; fi
 if ((footprint_status != 0)); then exit "$footprint_status"; fi
 if ((quota_status != 0)); then exit "$quota_status"; fi
 exit "$receipt_status"
}
trap finish_r5_receipts EXIT

source "$FUTURE_GPU_ROOT/runtime_context_r5.sh"
PROBE="$ROOT/profile-probe-build/managed_cuda_probe"
/usr/bin/python3 "$ROOT/read-r5-pins.py" > "$ROOT/results/r5-cuda-libpath-admitted.txt"


/usr/bin/python3 "$ROOT/phase_quota_r5.py" "$ROOT" before-probe
ldd "$PROBE" > "$ROOT/results/r5-probe-ldd.txt" 2>&1
/usr/bin/python3 - "$ROOT" <<'PY_LDD'
import json,re,sys
from pathlib import Path
root=Path(sys.argv[1]);a=json.loads((root/'r5-admission.json').read_text());text=(root/'results/r5-probe-ldd.txt').read_text()
assert 'not found'not in text
m=re.search(r'libcudart\.so\.12[^\n]*=>\s+(\S+)',text)
assert m and str(Path(m.group(1)).resolve())==a['cuda_cudart']['resolved']
PY_LDD
"$PROBE" > "$ROOT/results/managed-pointer-launch-probe-r5.json"
"$CONDA_PREFIX/bin/python" "$ROOT/capture_kokkos_profile_r5.py"
/usr/bin/python3 "$ROOT/namespace_footprint.py" "$ROOT" r5-probe
