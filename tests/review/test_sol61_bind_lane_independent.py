"""Independent SOURCE_ONLY receipt; native engine/module/lane are named host substitutes."""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE = "dbc9e1cd725c894f615b76a459d47986a8febd03"
EXECUTOR = "python/pops/runtime/_runtime_executor.py"
INSTALL = "python/pops/runtime/_amr_system_install.py"


def function(path, name):
    source = (ROOT / path).read_text()
    return source, next(n for n in ast.walk(ast.parse(source))
                       if isinstance(n, ast.FunctionDef) and n.name == name)


def run(nodes, namespace):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "<actual-independent-install-statements>", "exec"), namespace)


def calls(node, name):
    return any(isinstance(n, ast.Call) and (
        (isinstance(n.func, ast.Attribute) and n.func.attr == name) or
        (isinstance(n.func, ast.Name) and n.func.id == name)) for n in ast.walk(node))


class NativeLifecycleSubstitute:
    """Only records sequencing; does not emulate a native compiled provider or communicator."""
    def __init__(self):
        self.events = []
        self.providers = []
        self.lane = None
        self.hierarchy = False

    def add_equation(self, name, model, **kw):
        assert self.lane is not None
        self.events.append(("provider", name))
        self.providers.append(model)

    def geometry(self, layout):
        if self.lane is None:
            raise RuntimeError("no prepared lane")
        if len(self.providers) != 2:
            raise RuntimeError("providers incomplete")
        if self.hierarchy:
            raise RuntimeError("hierarchy already materialized")
        self.events.append(("geometry", layout))

    def _finish_program_install(self, *args):
        self.hierarchy = True
        self.events.append(("materialize", args))


def installation_sequence():
    _, body = function(INSTALL, "_install_compiled")
    return [n for n in body.body if calls(n, "add_equation")
            or calls(n, "install_embedded_boundary") or calls(n, "_finish_program_install")]


@pytest.mark.parametrize("mutation", (None, "early", "late", "missing_provider"))
def test_actual_install_statements_discriminate_two_illegal_boundaries(mutation):
    nodes = installation_sequence()
    assert len(nodes) == 3
    if mutation == "early":
        nodes = [nodes[1], nodes[0], nodes[2]]
    elif mutation == "late":
        nodes = [nodes[0], nodes[2], nodes[1]]
    engine = NativeLifecycleSubstitute()
    lane, layout, first, second = object(), object(), object(), object()
    engine.lane = lane
    plan = NS(artifact=NS(layout_plan=NS(layouts=(layout,))))
    models = {"left": (first, "sx", "tx"), "right": (second, "sy", "ty")}
    if mutation == "missing_provider":
        models.pop("right")
    namespace = dict(self=engine, install_plan=plan, lowered_instances=models,
                     per_block_params={}, compiled=object(), so_path="host-only", bind_schema=object(), params={})
    module = NS(install_embedded_boundary=lambda e, l: e.geometry(l))
    with patch.dict(sys.modules, {"pops.runtime._runtime_mesh_lowering": module}):
        if mutation is not None:
            with pytest.raises(RuntimeError):
                run(nodes, namespace)
            assert not any(event[0] == "geometry" for event in engine.events)
        else:
            run(nodes, namespace)
            assert [e[0] for e in engine.events] == ["provider", "provider", "geometry", "materialize"]
            assert engine.providers == [first, second]
            assert engine.events[2][1] is layout
            assert engine.lane is lane


def test_lowlevel_without_installplan_keeps_no_geometry_route():
    engine = NativeLifecycleSubstitute()
    engine.lane = object()
    run(installation_sequence(), dict(self=engine, install_plan=None,
        lowered_instances={"a": (object(), None, None), "b": (object(), None, None)},
        per_block_params={}, compiled=None, so_path=None, bind_schema=None, params={}))
    assert [row[0] for row in engine.events] == ["provider", "provider", "materialize"]


def test_actual_executor_and_install_authentication_precede_all_consumers():
    _, executor = function(EXECUTOR, "_install_adaptive_native_engine")
    assert not calls(executor, "install_embedded_boundary")
    positions = {name: next(i for i, n in enumerate(executor.body) if calls(n, name))
                 for name in ("require_install_authority", "_require_native_geometry",
                              "AmrSystem", "install_runtime_authorities", "_install_compiled")}
    assert list(positions.values()) == sorted(positions.values())
    source, install = function(INSTALL, "_install_compiled")
    names = ("_require_exact_install_inputs", "validate_install_arguments", "add_equation",
             "install_embedded_boundary", "_finish_program_install")
    sequence = [next(i for i, n in enumerate(install.body) if calls(n, name)) for name in names]
    assert sequence == sorted(sequence)
    # Public InitialCondition bindings still own initial_rows; the EB move does not synthesize data.
    text = ast.get_source_segment((ROOT / EXECUTOR).read_text(), executor)
    assert "for binding in plan.initial_condition_plan.bindings" in text
    assert "binding.source.options.to_data()" in text
    assert "initial_values=tuple(initial_rows)" in text
    assert "bootstrap_plan is not install_plan.bootstrap_plan" in source
    assert "amr_transfer is not install_plan.amr_transfer" in source


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_public_normalized_geometry_lowered_after_providers(dimension):
    from pops.analytic import coordinates
    from pops.boundary import ZeroFlux
    from pops.domain import CartesianDomain
    from pops.layouts import AMR
    from pops.mesh import CartesianGrid, normalize_layout_plan
    from pops.mesh.geometry import EmbeddedBoundary, LevelSet
    from pops.mesh.masks import CutCell
    from pops.model import OwnerPath
    from pops.runtime._runtime_mesh_lowering import install_embedded_boundary
    from tests.python.support.layout_plan import final_amr_layout

    frame = CartesianDomain("independent-eb", (0.,)*dimension, (1.,)*dimension).frame()
    grid = CartesianGrid(frame=frame, cells=(8,)*dimension)
    base = final_amr_layout(grid, max_levels=2, ratio=2)
    geometry = EmbeddedBoundary(LevelSet(coordinates(frame)[-1]-.375),
        CutCell(kappa_min=.03, face_open_eps=.02, cut_theta_min=.04), ZeroFlux())
    layout = AMR(grid=grid, hierarchy=base.hierarchy, tagging=base.tagging, regrid=base.regrid,
        transfer=base.transfer, execution=base.execution, embedded_boundary=geometry,
        patch_layout=base.patch_layout, load_balance=base.load_balance, tagger=base.tagger,
        clustering=base.clustering, reflux=base.reflux)
    normalized, = normalize_layout_plan(layout, owner=OwnerPath.case("independent-eb")).layouts
    invocations = []
    probe = NS(_s=NS(_set_analytic_level_set=lambda *args: invocations.append(args)))
    install_embedded_boundary(probe, normalized)
    assert len(invocations) == 1
    opcodes, literals, mode, kappa, eps, theta = invocations[0]
    assert opcodes == ["xyz"[dimension-1], "constant", "sub"]
    assert literals == [0., .375, 0.]
    assert (mode, kappa, eps, theta) == ("cutcell", .03, .02, .04)
    with pytest.raises(TypeError, match="normalized layout"):
        install_embedded_boundary(probe, object())
    assert len(invocations) == 1


def test_uniform_native_authority_and_guards_are_byte_unchanged():
    old = subprocess.check_output(["git", "show", f"{BASE}:{EXECUTOR}"], cwd=ROOT, text=True)
    current = (ROOT / EXECUTOR).read_text()
    def provider(source):
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef)
                    and n.name == "_UniformNativeProvider")
        return ast.get_source_segment(source, node)
    assert provider(old) == provider(current)
    for path in ("python/pops/runtime/_runtime_authorities.py",
                 "python/pops/runtime/_runtime_mesh_lowering.py"):
        assert subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT) == (ROOT/path).read_bytes()
    # Other AMR changes (including history capability @2) are independent of EB
    # assembly. Keep the complete native EB setter and its guards byte-pinned.
    path = "src/runtime/amr/amr_system.cpp"
    old_cpp = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT, text=True)
    current_cpp = (ROOT / path).read_text()
    def eb_setter(source):
        start = source.index("void AmrSystem<Dim>::set_analytic_level_set(")
        return source[start:source.index("\ntemplate <int Dim>", start)]
    assert eb_setter(old_cpp) == eb_setter(current_cpp)


@pytest.mark.parametrize("changed", (None, "artifact", "instances", "params", "aux", "field_plans", "context"))
def test_actual_exact_input_guard_rejects_equivalent_aliases(changed):
    _, guard = function("python/pops/runtime/_bound_snapshot.py", "_require_exact_install_inputs")
    plan = NS(artifact=NS(plan=NS(field_plans={})), instances={}, params={}, aux={}, execution_context=object())
    engine = NS(_execution_context=plan.execution_context)
    args = dict(engine=engine, compiled=plan.artifact, instances=plan.instances, field_plans=plan.artifact.plan.field_plans,
                aux=plan.aux, params=plan.params, install_plan=plan)
    changed_key = {"artifact": "compiled", "context": "context"}.get(changed, changed)
    if changed_key == "context":
        engine._execution_context = object()
    elif changed_key is not None:
        args[changed_key] = {} if changed_key != "compiled" else NS(plan=plan.artifact.plan)
    namespace = {"Any": object}
    run([guard], namespace)
    # Named authentication seam: receives argument identity, not a fabricated InstallPlan authority.
    def require_exact(p):
        assert p is plan
        return p
    with patch.dict(sys.modules, {"pops.runtime._layout_install_projection": NS(require_install_authority=require_exact)}):
        if changed is None:
            assert namespace[guard.name](**args) is plan
        else:
            with pytest.raises(ValueError, match="must be the exact value from the InstallPlan"):
                namespace[guard.name](**args)
