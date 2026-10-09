# ruff: noqa: E402 -- select and authenticate the source package before importing it
"""Bounded source/lowering coherence probe for diffusion-v6 and Program ComputedDt."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

source = Path(sys.argv[1]).resolve()
manifest = json.loads((source / "source_manifest.json").read_text())
for name, digest in manifest["files"].items():
    assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name
sys.path.insert(0, str(source / "python"))
import pops
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.time import FixedDt

assert Path(pops.__file__).resolve() == source / "python/pops/__init__.py"
fixture = Path(__file__).resolve().parents[1] / "python/support/integral_diffusion_case.py"
spec = importlib.util.spec_from_file_location("coherence_public_fixture", fixture)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


results = []
for kind, target in (("uniform", "system"), ("amr2", "amr_system")):
    case, layout, right, left, _ = module.build_diffusion_integral_case(kind)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    program = resolved.time
    generated = emit_cpp_program(
        program, model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks), target=target
    )
    transfers = [
        line.strip() for line in generated.splitlines() if "ctx.consume_external_trace(" in line
    ]
    assert program._serialize()["version"] == 6 and len(transfers) == 2
    assert right.identity in transfers[0] and left.identity in transfers[1]
    assert all(
        "diffusive-operation" in line and "/occurrence:" in line and "/evaluation:1" in line
        for line in transfers
    )
    assert generated.count(".stage_accepted_exchanges(") == 1
    results.append(
        {
            "kind": kind,
            "version": 6,
            "ir_hash": program._ir_hash(),
            "semantic_sha256": digest(program._semantic_serialize()),
            "cpp_sha256": hashlib.sha256(generated.encode()).hexdigest(),
            "trace_lines": transfers,
        }
    )

# Independent branch mutation of the public fixture: each returned RHS stays region-owned.
fixture_text = fixture.read_text()
assert fixture_text.count("    selected = rate(temporal.n)") == 1
scoped_text = fixture_text.replace(
    "    selected = rate(temporal.n)",
    """    initial_value = temporal.n
    captured_rates = []
    def capture_rate(P):
        value = rate(initial_value)
        captured_rates.append(value)
        return value
    selected = program.branch(program.dot(initial_value, initial_value) > 0,
                              capture_rate, capture_rate)""",
)
scoped_text = scoped_text.replace(
    "rate=rate(temporal.n) if diagnostic else selected,", "rate=captured_rates[0],"
)
scoped_text = scoped_text.replace(
    "rate=selected, axis=0, side=0, component=0)",
    "rate=captured_rates[1], axis=0, side=0, component=0)",
)
scoped_namespace = {}
exec(compile(scoped_text, str(fixture) + "/independent-scoped-probe", "exec"), scoped_namespace)
phase = "public authoring"
try:
    scoped_case, scoped_layout, *_ = scoped_namespace["build_diffusion_integral_case"]()
    phase = "validation and resolution"
    scoped_resolved = pops.resolve(pops.validate(scoped_case), layout=scoped_layout)
    scoped_version = scoped_resolved.time._serialize()["version"]
    phase = "emission"
    emit_cpp_program(
        scoped_resolved.time,
        model_graph=ProgramModelGraph.from_resolved_blocks(scoped_resolved.blocks),
        target="system",
    )
except (ValueError, NotImplementedError) as error:
    scoped_result = {
        "refused": True,
        "phase": phase,
        "diagnostic": str(error),
        "provisional_version": scoped_version if phase == "emission" else None,
    }
else:
    raise AssertionError("scoped diffusion trace escaped exact accepted-face authorization")

original_trace = pops.Program.accept_external_trace


def select_branch_result(self, state, **kwargs):
    kwargs["rate"] = next(value for value in self._values if value.op == "branch")
    return original_trace(self, state, **kwargs)


with patch.object(pops.Program, "accept_external_trace", select_branch_result):
    try:
        scoped_namespace["build_diffusion_integral_case"]()
    except ValueError as error:
        assert "conservative RHS" in str(error)
        scoped_result["branch_result_authoring_refusal"] = str(error)
    else:
        raise AssertionError("branch result accepted as an exact RHS trace selector")

if hasattr(pops.time, "ComputedDt"):
    from pops.time import ComputedDt

    original = pops.Program.step_strategy

    def computed(self, policy):
        assert isinstance(policy, FixedDt)
        self.reached_duration(0.8 * self.requested_dt())
        return original(self, ComputedDt(module.DT))

    with patch.object(pops.Program, "step_strategy", computed):
        case, layout, *_ = module.build_diffusion_integral_case()
    resolved = pops.resolve(pops.validate(case), layout=layout)
    assert resolved.time._serialize()["version"] == 6
    try:
        emit_cpp_program(
            resolved.time,
            model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks),
            target="system",
        )
    except NotImplementedError as error:
        assert "spatial interval" in str(error)
    else:
        raise AssertionError("ComputedDt diffusion-v6 admitted without exchange-duration remapping")
    computed_result = "refused before emission: spatial interval remapping unavailable in v1"
else:
    computed_result = "absent in diffusion-only baseline"
receipt = {
    "source_object": manifest["base_commit"],
    "source_tree_sha256": manifest["source_tree_sha256"],
    "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
    "scope": "source and public lowering only",
    "diffusion": results,
    "computed_diffusion": computed_result,
    "scoped_diffusion": scoped_result,
}
if len(sys.argv) > 2:
    destination = Path(sys.argv[2])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
