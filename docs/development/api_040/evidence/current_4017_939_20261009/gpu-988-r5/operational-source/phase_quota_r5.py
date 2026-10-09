import json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);phase=sys.argv[2];a=json.loads((root/'r5-admission.json').read_text())
key={'before-probe':'remaining_growth_bytes_before_probe','before-native':'remaining_growth_bytes_before_native','before-runtime':'remaining_growth_bytes_before_runtime'}[phase]
remaining=a[key];assert type(remaining)is int and remaining>=0
raise SystemExit(subprocess.call([sys.executable,str(root/'quota_guard_resume.py'),str(root),'r5-'+phase,str(remaining)]))
