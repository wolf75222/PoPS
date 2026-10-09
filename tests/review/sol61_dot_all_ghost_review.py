"""Replay the frozen independent a77 scaffold on the exact AMR ghost fix.

Only candidate identity/output directory and the previously accepted mismatch
assertion change. Production functions are still extracted from git objects.
"""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = "def8c752fde989734cfc4aace76a447ce3d0d63d"
PROBE_FREEZE = "98e6246c3daab41df8a846168ffa9f3a7da4dd32"
PROBE_PATH = "tests/review/sol61_dot_all_a77_review.py"
source = subprocess.check_output(
    ("git", "show", PROBE_FREEZE + ":" + PROBE_PATH), cwd=ROOT, text=True)

replacements = {
    'SHA = "a77e1b1ce09b84171fd3f63ffc64f6995614b8e5"': f'SHA = "{CANDIDATE}"',
    'OUT = ROOT / "outputs/sol61-dot-all-a77-independent"':
        'OUT = ROOT / "outputs/sol61-dot-all-ghost-independent"',
    'direct.ghosts_id=99;ok(a.dot_all(7,f.levels[1],direct)==13);':
        'direct.ghosts_id=99;refuse_ghost([&]{a.dot_all(7,f.levels[1],direct);});',
    'int main(){':
        'template<class F>void refuse_ghost(F f){events.clear();bool r=false;'
        'try{f();}catch(const std::invalid_argument&e){'
        'r=std::string(e.what())=="AMR Program dot_all ghost count differs";}'
        'ok(r&&events=="V");}\nint main(){',
    ' std::cout<<checks<<" independent actual-kernel/provider/identity checks PASS\\n";':
        '// Mismatched ghosts must also refuse on a noncontributing replica.\n'
        ' Field ghost_right=f.levels[1];ghost_right.ghosts_id=99;\n'
        ' a.lane={1,2};refuse_ghost([&]{a.dot_all(7,f.levels[1],ghost_right);});\n'
        ' std::cout<<checks<<" independent actual-kernel/provider/identity checks PASS\\n";',
}
for previous, replacement in replacements.items():
    assert source.count(previous) == 1, previous
    source = source.replace(previous, replacement)

exec(compile(source, str(ROOT / PROBE_PATH), "exec"), {"__file__": str(ROOT / PROBE_PATH)})
