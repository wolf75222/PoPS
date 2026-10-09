"""Independent re-reception of sharing fix, exact-parent hashes and weak lifetime.

The integrated source consists of 908508a plus only 1c84af0, without unrelated
diagnostic commits. Earlier raw receipts remain untouched.
"""
from hashlib import sha256
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
FIX = "1c84af06bad8f86680427e1066d8ec64481ad68c"
RECEIVER = "7eecb6de24c628e03a0697a503bc5c09586abdde"
BASE = "7c03184d7aeda71a32e1c3e9ea5eddc0ff3d7810"
probe = subprocess.check_output([
    "git", "show", BASE + ":tests/review/sol61_layering_908508a.py"
], cwd=ROOT, text=True)
for path in ("python/pops/_ir/finite_linear.py",
             "tests/python/unit/numerics/test_symbolic_policy_layering.py"):
    assert subprocess.check_output(["git", "show", FIX + ":" + path], cwd=ROOT) == \
           subprocess.check_output(["git", "show", RECEIVER + ":" + path], cwd=ROOT)


def replace(before, after):
    global probe
    assert probe.count(before) == 1, before[:120]
    probe = probe.replace(before, after)


replace('CANDIDATE = "908508a7ce8ccc05c6241f6239818380fd0760e3"',
        'CANDIDATE = "' + RECEIVER + '"')
replace('OUT = ROOT / "outputs/sol61-layering-908508a"',
        'OUT = ROOT / "outputs/sol61-layering-1c84af0"')
replace('    if mixed:\n        value=p.value',
        '    if mixed == "arith":\n'
        '        value=(vec+vec*Fraction(2,3)).materialize(p,"mapped",template=u.n,at=u.next.point)\n'
        '    elif mixed:\n        value=p.value')
replace('receipt["programs"]={"plain":program_image(False),"mixed":program_image(True)}',
        'receipt["programs"]={"plain":program_image(False),"mixed":program_image(True),'
        '"arith":program_image("arith")}')
replace('assert after["cases"]["scalar_first_mixed"]["joint_count"] == 2',
        'assert after["cases"]["scalar_first_mixed"]["joint_count"] == 1')
replace('assert after["programs"]["mixed"]["joint_count"] == 2',
        'assert after["programs"]["mixed"]["joint_count"] == 1')
replace('assert before["programs"]["mixed"]["hash"] != after["programs"]["mixed"]["hash"]',
        'assert before["programs"]["mixed"]["hash"] == after["programs"]["mixed"]["hash"]')
replace('"confirmed_regression": "mixed Expr/finite projections duplicate application and change Program hash"',
        '"received_fix": "mixed application sharing and Program hash match exact parent"')
replace('print("CONFIRMED REGRESSION: mixed Program application count 1 -> 2; Program hash differs")',
        'print("RECEIVED FIX: mixed Program count=1 and hash matches exact parent")')

namespace = {"__file__": __file__, "__name__": "__main__"}
exec(compile(probe, __file__, "exec"), namespace)
before, after = namespace["before"], namespace["after"]
assert before["cases"]["scalar_first_mixed"] == after["cases"]["scalar_first_mixed"]
assert before["programs"]["mixed"] == after["programs"]["mixed"]
assert before["cases"]["finite_vector_arithmetic"]["canonical"] == \
       after["cases"]["finite_vector_arithmetic"]["canonical"]
assert before["programs"]["arith"]["hash"] != after["programs"]["arith"]["hash"]
result = namespace["result"]
result["source_fix"] = FIX
result["scaffold_commit"] = BASE
result["scaffold_sha256"] = sha256(probe.encode()).hexdigest()
result["residual_regression"] = "finite vector scalar multiplication changes sharing and Program hash"
(namespace["OUT"] / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print("CONFIRMED RESIDUAL: finite-vector arithmetic Program hash differs from exact parent")
