"""Compare authentic source packages in separate processes; no native/JIT execution."""
from pathlib import Path
import argparse
import hashlib
import json
import runpy
import sys

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("--rate-first", action="store_true")
args = parser.parse_args()
sys.path.insert(0, str(args.source / "python"))
helpers = runpy.run_path(str(Path(__file__).with_name("test_sol61_t5_public_capture.py")))
result = {}
for rate_first in (args.rate_first,):
    authored = helpers["authored"](typed=False, rate_first=rate_first)
    program = authored[2]
    ir = json.dumps(program._serialize(), sort_keys=True, separators=(",", ":")).encode()
    cpp = helpers["emitted"](authored).encode()
    result[str(rate_first)] = {"ir_hash": program._ir_hash(),
        "ir_bytes_sha256": hashlib.sha256(ir).hexdigest(),
        "cpp_sha256": hashlib.sha256(cpp).hexdigest(),
        "legacy_integral_identity": authored[3].identity}
print(json.dumps(result, indent=2, sort_keys=True))
