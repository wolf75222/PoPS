#!/bin/bash
set -euo pipefail
umask 077
: "${FUTURE_GPU_ROOT:?Root admitted namespace}"
: "${SLURM_JOB_ID:?Actual allocation required}"
: "${FUTURE_GPU_ROOT:?Root supplies admitted private namespace}"
ROOT="$FUTURE_GPU_ROOT"
finish_r5_receipts() {
 local body_status=$?
 set +e
 /usr/bin/python3 "$ROOT/registry_capture.py" "$ROOT" after-r5-runtime-finally
 local registry_status=$?
 /usr/bin/python3 "$ROOT/namespace_footprint.py" "$ROOT" r5-runtime-finally
 local footprint_status=$?
 /usr/bin/python3 "$ROOT/quota_guard_resume.py" "$ROOT" after-r5-runtime-finally 0
 local quota_status=$?
 /usr/bin/python3 - "$ROOT" "$body_status" "$registry_status" "$footprint_status" "$quota_status" <<'PY_STATUS'
import json,sys
from pathlib import Path
with(Path(sys.argv[1])/'results/r5-runtime-finally-statuses.json').open('x')as f:
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
/usr/bin/python3 "$ROOT/freeze_admission.py"
/usr/bin/python3 "$ROOT/phase_quota_r5.py" "$ROOT" before-runtime
source "$ROOT/miniforge/etc/profile.d/conda.sh"
conda activate "$ROOT/envs/pops_final_cuda_dim2"
[[ "$CONDA_PREFIX" == "$ROOT/envs/pops_final_cuda_dim2" ]]
export POPS_REQUIRE_NATIVE_TESTS=1 POPS_NATIVE_DIM=2 OMP_NUM_THREADS=2 POPS_THREADS=2 OMP_PROC_BIND=false


# A failed phase parser must stop the body, rather than disappear inside process substitution.
/usr/bin/python3 - "$ROOT" <<'PY_PHASES'
import json,re,sys
from pathlib import Path
root=Path(sys.argv[1]);phases=json.loads((root/'plan.json').read_text())['runtime']['phases']
assert isinstance(phases,list) and phases, 'Nonempty authenticated phase list required'
rows=[];seen=set()
for phase in phases:
 identity=phase['id'];world=phase['world']
 assert isinstance(identity,str) and re.fullmatch(r'[A-Za-z0-9_.-]+',identity) and identity not in seen
 assert type(world)is int and world>0
 seen.add(identity);rows.append(identity+'|'+str(world))
with(root/'results/r5-runtime-phases.tsv').open('x')as f:f.write('\n'.join(rows)+'\n')
PY_PHASES
while IFS='|' read -r phase world; do
 export FUTURE_GPU_PHASE="$phase"
 /usr/bin/python3 "$ROOT/quota_guard_resume.py" "$ROOT" "r5-before-$phase" 0
 env -u PYTHONPATH -u PYTHONOPTIMIZE -u PYTEST_ADDOPTS "$CONDA_PREFIX/bin/python" "$ROOT/launch_phase_r5.py"
 /usr/bin/python3 "$ROOT/namespace_footprint.py" "$ROOT" "r5-$phase"
done < "$ROOT/results/r5-runtime-phases.tsv"
