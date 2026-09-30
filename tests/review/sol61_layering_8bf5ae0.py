"""Receive literal-vector parity fix against the immutable pre-layering parent."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SCAFFOLD = "e0062b10048dffc7b2df3b1fb3ff86eb6f89bd26"
FIX = "8bf5ae0c3e028aefcf0b57f70a195017a896edfd"
RECEIVER = "ff07a7ce"
probe = subprocess.check_output([
    "git", "show", SCAFFOLD + ":tests/review/sol61_layering_1c84af0.py"
], cwd=ROOT, text=True)


def replace(before, after):
    global probe
    assert probe.count(before) == 1
    probe = probe.replace(before, after)


replace('FIX = "1c84af06bad8f86680427e1066d8ec64481ad68c"', 'FIX = "' + FIX + '"')
replace('RECEIVER = "7eecb6de24c628e03a0697a503bc5c09586abdde"', 'RECEIVER = "' + RECEIVER + '"')
replace('"tests/python/unit/numerics/test_symbolic_policy_layering.py"):',
        '"tests/python/unit/numerics/test_symbolic_policy_layering.py", "python/pops/linalg/finite.py"):')
replace('OUT = ROOT / "outputs/sol61-layering-1c84af0"', 'OUT = ROOT / "outputs/sol61-layering-8bf5ae0"')
replace('assert before["programs"]["arith"]["hash"] != after["programs"]["arith"]["hash"]',
        'assert before["programs"]["arith"] == after["programs"]["arith"]\n'
        'assert before["cases"]["finite_vector_arithmetic"] == after["cases"]["finite_vector_arithmetic"]')
replace('result["residual_regression"] = "finite vector scalar multiplication changes sharing and Program hash"',
        'result["received_literal_fix"] = "finite vector literal arithmetic encoding and Program hash match parent"')
replace('print("CONFIRMED RESIDUAL: finite-vector arithmetic Program hash differs from exact parent")',
        'print("RECEIVED: finite-vector arithmetic encoding and Program hash match exact parent")')
namespace = {"__file__": __file__, "__name__": "__main__"}
exec(compile(probe, __file__, "exec"), namespace)
result = namespace["result"]
result["second_scaffold_commit"] = SCAFFOLD
result["second_scaffold_sha256"] = sha256(probe.encode()).hexdigest()
(namespace["namespace"]["OUT"] / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
