"""Exact 69d4ce3 source provenance and small actual-header host reception."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SHA = "69d4ce3e98b6b17d6b81b2690413ed28837be487"
OUT = ROOT / "outputs/sol61-amr69-independent"
OUT.mkdir(parents=True, exist_ok=True)
paths = ("include/pops/amr/transfer/transfer_provider.hpp", "src/runtime/amr/amr_system.cpp")
for path in paths:
    assert (ROOT / path).read_bytes() == subprocess.check_output(("git", "show", SHA + ":" + path), cwd=ROOT)
source = (ROOT / paths[1]).read_text()


def calls(name):
    at = 0
    result = []
    while True:
        try:
            start = source.index(name + "(", at)
        except ValueError:
            return result
        opening = start + len(name)
        end, depth = opening + 1, 1
        while depth:
            depth += (source[end] == "(") - (source[end] == ")")
            end += 1
        result.append(source[start:end])
        at = end


prepare = calls("prepare_regridded_state_transfer")
transfer = calls("transfer_regridded_state")
assert len(prepare) == 3 and len(transfer) == 4
assert "periodic_axes" in prepare[0] and "periodic_axes" in prepare[1]
assert "p_->cfg.periodicity" in prepare[2]  # Bootstrap action.
assert "periodic_axes" in transfer[0]
assert "cfg.periodicity" in transfer[1]  # History remapping.
assert "cfg.periodicity" in transfer[2]  # Regridded live state.
assert "ConstantInjection, p_->cfg.periodicity" in transfer[3]  # Auxiliary route unchanged.
assert "physical_boundary.lower[axis] = physical_boundary.upper[axis] = !periodic_axes[axis]" in source
assert "mapping.coarse_origin = parent_layout.domain().lo" in source
assert "mapping.fine_origin = child_layout.domain().lo" in source
assert "interpolation_source_box(\n        child_patch, ratio, mapping, provider.capabilities().source_stencil_radius,\n        physical_boundary)" in source
assert "if (prepared.populated[offset(cell, dense_box)] == 0)" in source
compiler = shutil.which("clang++") or shutil.which("c++")
assert compiler
cpp = ROOT / "tests/review/sol61_amr69_transfer_review.cpp"
binary = OUT / "probe"
command = (compiler, "-std=c++20", "-O0", "-I" + str(ROOT / "include"),
           "-I/Users/romaindespoulain/miniforge3/envs/pops-api040/include", str(cpp), "-o", str(binary))
built = subprocess.run(command, capture_output=True, text=True)
assert built.returncode == 0, built.stderr
run = subprocess.run((str(binary),), capture_output=True, text=True)
assert run.returncode == 0, run.stderr
receipt = {
    "candidate": SHA, "compiler_command": command,
    "source_file_sha256": {path: sha256((ROOT / path).read_bytes()).hexdigest() for path in paths},
    "include_archive_sha256": sha256(subprocess.check_output(("git", "archive", SHA, "include"), cwd=ROOT)).hexdigest(),
    "review_cpp_sha256": sha256(cpp.read_bytes()).hexdigest(), "result": run.stdout.strip(),
    "route_checks": "bootstrap/live regrid/history/aux actual call signatures carry actual periodicity; source proof and kernel share physical-boundary clipping",
    "scope": "actual production headers + detached host FieldViews; no runtime MPI/bootstrap/regrid execution, JIT/install/GPU/PDE qualification",
}
(OUT / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
