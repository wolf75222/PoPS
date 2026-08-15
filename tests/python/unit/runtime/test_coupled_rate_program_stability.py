"""Public coupled-rate stability from authoring through native AMR execution.

The pure resolution check proves the public AMR contract.  The native check separately exercises
the complete root lifecycle and its numerical result; neither test recreates the retired
``CoupledSource.frequency(expr)`` registration seam or reaches a private runtime engine.
"""

from __future__ import annotations

import numpy as np
import pytest

import pops
from pops.amr import (
    AMRExecution,
    AMRHierarchy,
    AMRRegrid,
    AMRTagging,
    AMRTransfer,
    Buffer,
    ConflictPolicy,
    EqualityPolicy,
    Hysteresis,
    Tag,
)
from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.lib.initial import Constant
from pops.math import Const, ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import AdaptiveCFL, every
from tests.python.support.amr_snapshots import composite_active_block_state
from tests.python.support.requirements import repo_include


GRID_CELLS = 8
REFINEMENT_RATIO = 2
COLLISION_FREQUENCY = 4.0
PROGRAM_CFL = 0.25
PROGRAM_DT_BOUND = PROGRAM_CFL / COLLISION_FREQUENCY
INCLUDE = repo_include()


def _coupled_rate_case() -> tuple[pops.Case, AMR]:
    frame = Rectangle("coupled_rate_domain", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(
        Cartesian2D()
    )
    model = pops.Model("two_species_relaxation", frame=frame)
    electrons = model.species("electrons", state=("ne",))
    ions = model.species("ions", state=("ni",))
    x_axis, y_axis = frame.axes
    electron_flux = model.flux(
        "stationary_electrons",
        frame=frame,
        state=electrons,
        components={x_axis: (0.0 * electrons["ne"],), y_axis: (0.0 * electrons["ne"],)},
        waves={x_axis: (Const(0.0),), y_axis: (Const(0.0),)},
    )
    ion_flux = model.flux(
        "stationary_ions",
        frame=frame,
        state=ions,
        components={x_axis: (0.0 * ions["ni"],), y_axis: (0.0 * ions["ni"],)},
        waves={x_axis: (Const(0.0),), y_axis: (Const(0.0),)},
    )
    electron_transport = model.rate(
        "electron_transport", equation=ddt(electrons) == -div(electron_flux)
    )
    ion_transport = model.rate("ion_transport", equation=ddt(ions) == -div(ion_flux))
    collision = model.coupled_rate(
        "density_exchange",
        inputs=(electrons, ions),
        outputs={
            electrons: (Const(COLLISION_FREQUENCY) * (ions["ni"] - electrons["ne"]),),
            ions: (Const(COLLISION_FREQUENCY) * (electrons["ne"] - ions["ni"]),),
        },
    )

    case = pops.Case("coupled_rate_amr")
    electron_block = case.block("electrons", model, states=(electrons,))
    ion_block = case.block("ions", model, states=(ions,))
    electron_state = electron_block[electrons]
    ion_state = ion_block[ions]
    refine_threshold = case.param(RuntimeParam("refine_threshold", default=0.75))
    electron_numerics = DiscretizationPlan()
    electron_numerics.rates.add(
        electron_transport,
        FiniteVolume(
            flux=electron_flux,
            variables=variables.Conservative(electrons),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )
    ion_numerics = DiscretizationPlan()
    ion_numerics.rates.add(
        ion_transport,
        FiniteVolume(
            flux=ion_flux,
            variables=variables.Conservative(ions),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )
    case.numerics(electron_numerics, block=electron_block)
    case.numerics(ion_numerics, block=ion_block)

    program = pops.Program("explicit_coupled_rate")
    electron_time = program.state(electron_state)
    ion_time = program.state(ion_state)
    exchange = collision(electron_time.n, ion_time.n)
    program.commit_many(
        {
            electron_time.next: program.value(
                "electrons_next",
                electron_time.n
                + program.dt
                * (
                    electron_transport(electron_time.n)
                    + exchange[electron_block]
                ),
                at=electron_time.next.point,
            ),
            ion_time.next: program.value(
                "ions_next",
                ion_time.n
                + program.dt * (ion_transport(ion_time.n) + exchange[ion_block]),
                at=ion_time.next.point,
            ),
        }
    )
    program.set_dt_bound(lambda _program, cfl: cfl / COLLISION_FREQUENCY)
    program.step_strategy(AdaptiveCFL(PROGRAM_CFL))
    case.program(program)

    case.initials.add(
        InitialCondition(
            state=electron_state,
            value=Constant((1.0,)),
            projection=ConservativeCellAverage(),
        )
    )
    case.initials.add(
        InitialCondition(
            state=ion_state,
            value=Constant((0.5,)),
            projection=ConservativeCellAverage(),
        )
    )

    transfer = AMRTransfer()
    transfer.state(electron_state, StateTransfer())
    transfer.state(ion_state, StateTransfer())
    layout = AMR(
        grid=CartesianGrid(
            frame=frame,
            cells=(GRID_CELLS, GRID_CELLS),
            periodic=PeriodicAxes(frame.axes),
        ),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(
                Tag(ValueExpr(electron_state) > case.value(refine_threshold)),
                Buffer(cells=1),
            ),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(100, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.synchronous(),
    )
    return case, layout


def _resolve_coupled_rate_case(*, cxx: str | None = None):
    case, layout = _coupled_rate_case()
    compile_options = None if cxx is None else {"include": INCLUDE, "cxx": cxx}
    return pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options=compile_options,
    )


def test_two_block_repeated_snapshots_keep_authenticated_block_handles() -> None:
    from pops.problem._snapshot import build_authoring_snapshot, build_problem_snapshot
    from pops.time.references import block_name

    case, layout = _coupled_rate_case()
    electron = case.blocks()["electrons"]
    ion = case.blocks()["ions"]
    program = case._time_registry.program

    first = build_problem_snapshot(case)
    second = build_problem_snapshot(case)
    compile_first = build_authoring_snapshot(case, layout=layout, time=program, libraries=())
    compile_second = build_authoring_snapshot(case, layout=layout, time=program, libraries=())

    assert first.hash == second.hash
    assert first.artifact_hash == second.artifact_hash
    assert compile_first.hash == compile_second.hash
    assert compile_first.artifact_hash == compile_second.artifact_hash

    first_blocks = first.to_dict()["blocks"]
    second_blocks = second.to_dict()["blocks"]
    electron_identity = first_blocks["electrons"]["handle"]["$handle"]
    ion_identity = first_blocks["ions"]["handle"]["$handle"]
    assert second_blocks["electrons"]["handle"]["$handle"] == electron_identity
    assert second_blocks["ions"]["handle"]["$handle"] == ion_identity
    assert electron_identity != ion_identity

    resolved_electron = case.resolve(electron)
    resolved_ion = case.resolve(ion)
    assert resolved_electron.canonical_identity() == electron_identity
    assert resolved_ion.canonical_identity() == ion_identity
    assert case.resolve(resolved_electron).canonical_identity() == electron_identity
    assert case.resolve(resolved_ion).canonical_identity() == ion_identity

    coupled_blocks = [
        value.attrs["out_block"]
        for value in program._values
        if value.op == "coupled_rate_out"
    ]
    assert {block_name(block) for block in coupled_blocks} == {"electrons", "ions"}
    coupled_identities = {
        block_name(block): case.resolve(block).canonical_identity()
        for block in coupled_blocks
    }
    assert coupled_identities == {
        "electrons": electron_identity,
        "ions": ion_identity,
    }
    assert first.semantic_identity == second.semantic_identity
    assert compile_first.semantic_identity == compile_second.semantic_identity


def _single_module_source_case(coeff: float) -> pops.Case:
    frame = Rectangle("single_module_domain", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(
        Cartesian2D()
    )
    model = pops.Model("scalar_source", frame=frame)
    state = model.state(
        "U",
        components=("u",),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    model.source("forcing", on=state, value=(Const(coeff),))
    case = pops.Case("single_module_source")
    case.block("fluid", model)
    return case


def test_single_module_scientific_identity_is_formula_sensitive() -> None:
    from pops.problem._snapshot import build_problem_snapshot

    first = build_problem_snapshot(_single_module_source_case(1.0))
    same = build_problem_snapshot(_single_module_source_case(1.0))
    other = build_problem_snapshot(_single_module_source_case(2.0))
    first_hash = first.to_dict()["blocks"]["fluid"]["model"]["scientific_hash"]
    same_hash = same.to_dict()["blocks"]["fluid"]["model"]["scientific_hash"]
    other_hash = other.to_dict()["blocks"]["fluid"]["model"]["scientific_hash"]

    assert first_hash == same_hash
    assert first.hash == same.hash
    assert first_hash != other_hash
    assert first.hash != other.hash
    assert first.semantic_identity == same.semantic_identity


def test_empty_facade_module_cache_does_not_materialize_a_competing_module() -> None:
    from pops.problem._block_registry import stabilize_model_definition
    from pops.problem._snapshot_payload import _same_module_facade, _scientific_model_hash

    case = _single_module_source_case(1.0)
    model = case._block_registry.spec("fluid")["model"]
    selected = model.module
    facade = model._dsl
    fingerprint = model.owner_path.definition_fingerprint
    facade._module_cache = None

    assert _same_module_facade(model, selected) is None
    fallback = _scientific_model_hash(model, selected)
    assert facade._module_cache is None
    assert fallback == selected.module_hash()
    assert model.owner_path.definition_fingerprint == fingerprint

    stabilize_model_definition(model)
    assert facade._module_cache is selected
    assert _same_module_facade(model, selected) is facade
    assert _scientific_model_hash(model, selected) == facade._model_hash()
    assert _scientific_model_hash(model, selected) != selected.module_hash()


def test_concurrent_repeated_snapshots_keep_authenticated_block_handles() -> None:
    from concurrent.futures import ThreadPoolExecutor

    from pops.problem._snapshot import build_problem_snapshot

    case, _layout = _coupled_rate_case()
    electron = case.blocks()["electrons"]
    ion = case.blocks()["ions"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        snapshots = list(pool.map(lambda _index: build_problem_snapshot(case), range(16)))

    hashes = {snapshot.hash for snapshot in snapshots}
    semantic = {snapshot.semantic_identity for snapshot in snapshots}
    assert hashes == {snapshots[0].hash}
    assert semantic == {snapshots[0].semantic_identity}
    electron_identity = case.resolve(electron).canonical_identity()
    ion_identity = case.resolve(ion).canonical_identity()
    assert electron_identity != ion_identity
    assert all(
        snapshot.to_dict()["blocks"]["electrons"]["handle"]["$handle"] == electron_identity
        and snapshot.to_dict()["blocks"]["ions"]["handle"]["$handle"] == ion_identity
        for snapshot in snapshots
    )


def test_fresh_process_reconstructs_two_block_snapshot_identity() -> None:
    import os
    import subprocess
    import sys
    from pathlib import Path

    from pops.problem._snapshot import build_problem_snapshot

    case, _layout = _coupled_rate_case()
    local = build_problem_snapshot(case)
    repo = Path(__file__).resolve().parents[4]
    script = (
        "from tests.python.unit.runtime.test_coupled_rate_program_stability "
        "import _coupled_rate_case\n"
        "from pops.problem._snapshot import build_problem_snapshot\n"
        "case, _layout = _coupled_rate_case()\n"
        "snapshot = build_problem_snapshot(case)\n"
        "print(snapshot.hash)\n"
        "print(snapshot.semantic_identity.hexdigest)\n"
        "print(snapshot.artifact_hash)\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "python") + os.pathsep + str(repo) + os.pathsep + env.get(
        "PYTHONPATH", ""
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    remote_hash, remote_semantic, remote_artifact = completed.stdout.strip().splitlines()
    assert remote_hash == local.hash
    assert remote_semantic == local.semantic_identity.hexdigest
    assert remote_artifact == local.artifact_hash


def test_coupled_rate_program_resolves_with_explicit_amr_stability_bound() -> None:
    resolved = _resolve_coupled_rate_case()

    assert resolved.target == "amr_system"
    assert resolved.time.has_dt_bound()
    nodes = resolved.time.ir_nodes()
    assert [node["op"] for node in nodes].count("coupled_rate") == 1
    transport_rates = [node for node in nodes if node["op"] == "rhs"]
    assert len(transport_rates) == 2
    assert all(node["attrs"]["flux"] is True for node in transport_rates)
    assert all(node["attrs"]["fluxes"] is None for node in transport_rates)
    assert resolved.capabilities["resolution"]["amr_program"]["status"] == "proven"


@pytest.mark.compiler
@pytest.mark.native_loader
def test_coupled_rate_bound_drives_a_conservative_native_amr_step(
    isolated_native_cache, native_cxx, kokkos_root
) -> None:
    del isolated_native_cache, kokkos_root
    artifact = pops.compile(_resolve_coupled_rate_case(cxx=native_cxx))
    simulation = pops.bind(artifact)
    level_count = simulation.n_levels()
    assert level_count == 2
    assert simulation.program_report().level_relations == [
        {
            "parent_level": 0,
            "child_level": 1,
            "temporal_ratio": {"numerator": 1, "denominator": 1},
            "remainder_policy": "integral_only",
        }
    ]
    before = {
        block: tuple(
            composite_active_block_state(
                simulation,
                block,
                level,
                refinement_ratio=REFINEMENT_RATIO,
            ).copy()
            for level in range(level_count)
        )
        for block in ("electrons", "ions")
    }

    requested_end = 2.0 * PROGRAM_DT_BOUND
    assert requested_end > PROGRAM_DT_BOUND
    # The Program bound intentionally prevents reaching requested_end in one step. ``pops.run``
    # reports that incomplete run as an error; the accepted native transaction remains observable
    # and must have advanced by exactly the constrained macro-step.
    with pytest.raises(RuntimeError, match="max_steps exhausted before t_end"):
        pops.run(simulation, t_end=requested_end, max_steps=1)

    assert simulation.macro_step() == 1
    step = simulation.time()
    assert 0.0 < step <= PROGRAM_DT_BOUND
    assert simulation.time() == step
    assert simulation.n_levels() == level_count
    after = {
        block: tuple(
            composite_active_block_state(
                simulation,
                block,
                level,
                refinement_ratio=REFINEMENT_RATIO,
            ).copy()
            for level in range(level_count)
        )
        for block in ("electrons", "ions")
    }

    for level in range(level_count):
        electron_before = before["electrons"][level]
        ion_before = before["ions"][level]
        electron_after = after["electrons"][level]
        ion_after = after["ions"][level]
        electron_expected = electron_before + step * COLLISION_FREQUENCY * (
            ion_before - electron_before
        )
        ion_expected = ion_before + step * COLLISION_FREQUENCY * (electron_before - ion_before)

        assert np.all(electron_after < electron_before)
        assert np.all(ion_after > ion_before)
        np.testing.assert_array_equal(electron_after, electron_expected)
        np.testing.assert_array_equal(ion_after, ion_expected)
        np.testing.assert_array_equal(
            electron_after + ion_after,
            electron_before + ion_before,
        )
