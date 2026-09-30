"""Source/protocol checks only. No fabricated positive ALE physical receipt."""
import ast
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from tests.review import sol61_moving_interval_offline_oracle as oracle

ROOT = Path(__file__).resolve().parents[2]


def word(value, signed=False):
    return int(value).to_bytes(8, "little", signed=signed)


def test_contract_requires_external_full_campaign_without_claiming_execution():
    contract = oracle.contract()
    assert contract["scientific_status"] == "pending_receipts"
    assert contract["consumes_no_pops_package"]
    assert len(contract["expected_cases"]) == 14
    assert sum(len(case["phases"]) * 3 for case in contract["expected_cases"]) == 234
    assert "outputs" in contract["case_pins"]
    assert {case["cells"] for case in contract["expected_cases"]} == {16, 32}


def test_cli_isolated_and_pops_import_forbidden():
    code = '''import importlib.abc,runpy,sys
class RefusePoPS(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=="pops" or fullname.startswith("pops."): raise AssertionError("PoPS import forbidden")
sys.meta_path.insert(0,RefusePoPS())
sys.argv=[sys.argv[1],"--describe-contract"]
runpy.run_path(sys.argv[0],run_name="__main__")
'''
    result = subprocess.run([sys.executable, "-I", "-c", code, str(Path(oracle.__file__))],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["scientific_status"] == "pending_receipts"


def test_cli_cannot_receive_without_pins():
    result = subprocess.run([sys.executable, "-I", str(Path(oracle.__file__))], capture_output=True, text=True, timeout=20)
    assert result.returncode != 0 and "external owner pins are required" in result.stderr


@pytest.mark.parametrize("raw", [b"", b"POPSEX03", b"POPSEX03" + word(1 << 63), b"POPSEX02" + word(0)])
def test_truncated_foreign_or_unbounded_wire_is_refused(raw):
    with pytest.raises(ValueError):
        oracle.moving_image(raw, dict(cells=16, components=["a"]), 0, 1)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_wire_nonfinite_values_refused_before_numeric_allocation(value):
    with pytest.raises(ValueError, match="nonfinite"):
        oracle.Reader(struct.pack("<d", value)).real()


def point_bytes(**changes):
    values = dict(clock="clock", tick=0, level=0, substep=0, stage=0, numerator=0, denominator=1,
                  dt=.001, time=0., graph="", rate="", application="")
    values.update(changes)
    def text(value):
        return word(len(value.encode())) + value.encode()
    return (text(values["clock"]) + b"".join(word(values[key], True) for key in (
        "tick", "level", "substep", "stage", "numerator", "denominator"))
        + struct.pack("<dd", values["dt"], values["time"]) + b"".join(text(values[key]) for key in ("graph", "rate", "application")))


@pytest.mark.parametrize("change", [dict(tick=-1), dict(level=1), dict(substep=1), dict(stage=1),
                                    dict(numerator=1), dict(denominator=2), dict(dt=0.), dict(clock=""), dict(graph="foreign")])
def test_inadmissible_point_authorities_refused(change):
    with pytest.raises(ValueError, match="ordinary interval"):
        oracle.point(oracle.Reader(point_bytes(**change)))


def test_context_authenticates_every_binary64_bit_and_clock_byte():
    point = oracle.point(oracle.Reader(point_bytes(clock="clôck")))
    context = oracle.context(point, "physical-state")
    assert context.startswith("pops.exchange.frame.v1/6:clôck/0/0/0/0/0/1/")
    assert context.endswith("/0/14:physical-state")
    different = dict(point, dt=float.fromhex("0x1.0624dd2f1a9fdp-10"))
    assert oracle.context(different, "physical-state") != context


def test_fab_declared_allocation_needs_actual_payload():
    with pytest.raises(ValueError, match="allocation"):
        oracle.fab(oracle.Reader(word(32) + word(3, True)), 32, 3)


def test_wire_topology_foreign_rank_and_missing_ownership_refused():
    # A topology descriptor is a structural primitive, not a saved physical state.
    raw = b"".join(word(value, True) for value in (1, 3, 1, 0, 0, 2, 1, 0, 15, 0, 1, 0))
    with pytest.raises(ValueError, match="local ownership"):
        oracle.topology(raw, 16, 3, 1, 2)
    with pytest.raises(ValueError, match="rank-space"):
        oracle.topology(raw, 16, 3, 0, 2)


def test_native_fixture_saves_all_phases_and_converges_calls():
    path = ROOT / "tests/python/integration/runtime/test_public_moving_interval.py"
    source = path.read_text()
    ast.parse(source)
    for phase in ("initial", "step1", "accepted", "continuous", "restored", "replayed", "before", "rejected", "retried"):
        assert 'directory,"' + phase + '"' in source
    assert 'velocity=-.3 if components==("a","b","c") else .7' in source
    assert 'collective_attempt(world,lambda:pops.run' in source
    assert 'moving shared face or fixed-domain boundary is inconsistent' in source
    assert 'expected_type="ValueError"' in source and 'expected_type="RuntimeError"' in source
    assert 'failure[1]==expected_message' in source
    assert 'assert len(publications)==1' in source
    assert source.count("assert_exact_moving_snapshot(") == 3
    assert "compile_resolved_plan_once" in source and "collective_directory" in source
    assert 'sorted(publications,key=lambda row:row[0])' in source


def test_refusal_control_is_unstable_at_actual_periodic_endpoints():
    # Algebraic admission counterexample; no positive physical states or receipts.
    for cells in (16, 32):
        assert .1 * abs(.7 - 0.) > 1 / cells
        assert .001 * abs(.7 - 0.) < 1 / cells
    source = (ROOT / "include/pops/runtime/program/program_context_moving_interval.inc").read_text()
    assert source.index('"moving shared face or fixed-domain boundary is inconsistent"') < source.index('"moving-GCL"')


@pytest.mark.parametrize("components,velocity", [(("a",), .7), (("a", "b", "c"), -.3), (("c", "a", "b"), .7)])
@pytest.mark.parametrize("with_source", [False, True])
def test_default_authoring_emission_is_byte_exact_against_original_fixture(components, velocity, with_source):
    # Read the original authoring function, then feed both through the same real
    # current compiler. Only the optional attempted-dt fixture seam was added.
    import pops
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from tests.python.unit.codegen.test_moving_interval_codegen import declared_case
    baseline = subprocess.run(["git", "show", oracle.SOURCE_CONTRACT + ":tests/python/unit/codegen/test_moving_interval_codegen.py"],
                              cwd=ROOT, capture_output=True, text=True, timeout=10)
    assert baseline.returncode == 0, baseline.stderr
    tree = ast.parse(baseline.stdout)
    definition = ast.Module(body=[node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
                                 or isinstance(node, ast.FunctionDef) and node.name == "declared_case"], type_ignores=[])
    namespace = {}
    exec(compile(definition, "frozen-original-fixture", "exec"), namespace)
    emitted = []
    for constructor in (namespace["declared_case"], declared_case):
        case, layout = constructor(components=components, velocity=velocity, with_source=with_source)
        resolved = pops.resolve(pops.validate(case), layout=layout)
        emitted.append(emit_cpp_program(resolved.time, model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks)))
    assert emitted[0] == emitted[1]
