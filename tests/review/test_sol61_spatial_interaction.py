"""Public source contracts plus actual-header host quadrature; no native DSO/JIT."""
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest
import pops
from pops.fields import SpatialInteractionKernel, CellVolumeMeasure, CellMidpoint, DirectSpatialInteraction
from pops.model.spaces import FieldSpace
from pops.time import Program
from pops.time._program.spatial_interaction import interaction_contract
from pops.codegen.program_emit_spatial_interaction import emit_spatial_interaction


ROOT = Path(__file__).resolve().parents[2]


def build(*, dimension=2, budget=2**64-1, components=(2, 0)):
    model = pops.Model("physical_density")
    declared = model.state("rho", components=("north", "east", "third"))
    case = pops.Case("spatial_interaction")
    block = case.block("density", model, states=(declared,))
    program = Program("direct_map")
    source = program.state(block[declared]).n
    output = FieldSpace("potential", components=("third", "north"), sampling="cell_center")
    kernel = SpatialInteractionKernel(dimension, lambda x, y: 1 + x[0] * y[0] - 2 * y[-1])
    result = program.spatial_interaction(source, kernel, output_space=output,
        measure=CellVolumeMeasure(), quadrature=CellMidpoint(),
        realization=DirectSpatialInteraction(budget), components=components)
    return program, source, result


def test_source_kernel_measure_quadrature_method_and_publication_are_distinct():
    program, source, result = build()
    assert result.space.kind == "field" and result.block == source.block and result.point == source.point
    assert not program._commits
    assert program._serialize()["version"] == 17
    assert result.attrs["max_workspace_bytes"]["uint64_hex"] == "ffffffffffffffff"
    detached = program.detached_copy() if hasattr(program, "detached_copy") else program
    json.dumps(detached._serialize())
    _, dimension, cpp, limit, components = interaction_contract(result)
    assert dimension == 2 and limit == 2**64-1 and components == (2, 0)
    assert "x[0]" in cpp and "y[1]" in cpp
    for target in ("system", "amr_system"):
        variables = {source.id: "rho"}
        lines = []
        emit_spatial_interaction(result, variables, lines, block_indices=program._block_indices(), target=target)
        emitted = "\n".join(lines)
        assert "ctx.spatial_interaction" in emitted and "18446744073709551615ULL" in emitted
        assert "commit" not in emitted and "finite_linear_apply<12" not in emitted


@pytest.mark.parametrize("budget", (0, -1, True, 2**64, 1.0))
def test_budget_is_exact_unsigned_without_global_cbor_changes(budget):
    with pytest.raises(ValueError, match="uint64"):
        DirectSpatialInteraction(budget)


def test_legacy_program_is_not_upgraded():
    program, _, _ = build()
    legacy = Program("legacy")
    before = legacy._serialize()
    assert before["version"] < 17
    assert program._serialize()["version"] == 17
    assert legacy._serialize() == before


def test_kernel_refuses_foreign_state_and_opaque_callback_result():
    from pops._ir.expr import Var
    with pytest.raises(ValueError, match="foreign"):
        SpatialInteractionKernel(2, lambda x, y: Var("rho", "cons"))
    with pytest.raises(TypeError, match="inspectable"):
        SpatialInteractionKernel(2, lambda x, y: object())
    with pytest.raises(ValueError, match="finite"):
        SpatialInteractionKernel(2, lambda x, y: float("inf"))


def test_wrong_output_frame_is_rejected_without_ssa_publication():
    program, source, _ = build()
    before = program._serialize()
    with pytest.raises(ValueError, match="frame"):
        program.spatial_interaction(source, SpatialInteractionKernel(2, lambda x, y: 1),
            output_space=FieldSpace("foreign", components=("a",), frame="foreign", sampling="cell_center"),
            components=(0,), measure=CellVolumeMeasure(), quadrature=CellMidpoint(),
            realization=DirectSpatialInteraction(1024))
    assert before == program._serialize()


def test_dimensional_contract_is_W_times_density_times_volume():
    from pops.model import PhysicalDimension
    from fractions import Fraction
    length = PhysicalDimension((("length", Fraction(1)),))
    density = PhysicalDimension((("mass", Fraction(1)), ("length", Fraction(-2))))
    mass = PhysicalDimension((("mass", Fraction(1)),))
    model = pops.Model("dimensioned")
    declared = model.state("rho", components=("a",), units=(density,))
    case = pops.Case("units")
    block = case.block("rho", model, states=(declared,))
    program = Program("units")
    source = program.state(block[declared]).n
    kernel = SpatialInteractionKernel(2, lambda x, y: 1, units=PhysicalDimension())
    value = program.spatial_interaction(source, kernel,
        output_space=FieldSpace("integral", components=("a",), units=(mass,), sampling="cell_center"),
        measure=CellVolumeMeasure((length, length)), quadrature=CellMidpoint(),
        realization=DirectSpatialInteraction(4096))
    interaction_contract(value)
    with pytest.raises(ValueError, match=r"W\*rho\*dmu"):
        program.spatial_interaction(source, kernel,
            output_space=FieldSpace("wrong", components=("a",), units=(density,), sampling="cell_center"),
            measure=CellVolumeMeasure((length, length)), quadrature=CellMidpoint(),
            realization=DirectSpatialInteraction(4096))


@pytest.mark.parametrize("target", ("system", "amr_system"))
def test_true_public_case_resolve_and_program_emission(tmp_path, target):
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian2D
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.layouts import Uniform
    from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
    from pops.math import ddt, div
    from pops._ir.expr import Const
    from pops.numerics.terms import Flux
    from pops.time import FixedDt
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.projection import ConservativeCellAverage
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_codegen import emit_cpp_program
    frame = CartesianDomain("physical", lower=(1., -2.), upper=(3., 2.)).frame(Cartesian2D())
    model = pops.Model("spatial_density", frame=frame)
    rho = model.state("rho", components=("a", "b", "c"), sampling="cell_average")
    flux = model.flux("zero_flux", frame=frame, state=rho,
        components={axis: tuple(0 * rho[i] for i in range(3)) for axis in frame.axes},
        waves={axis: (Const(0),) * 3 for axis in frame.axes})
    rate = model.rate("retain", equation=ddt(rho) == -div(flux))
    case = pops.Case("public_spatial_interaction")
    block = case.block("rho", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(rho),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case.numerics(numerics, block=block)
    program = Program("public_direct")
    u = program.state(block[rho])
    field = program.spatial_interaction(u.n, SpatialInteractionKernel(2, lambda x, y: 1 + x[0] * y[1]),
        output_space=FieldSpace("potential", components=("c", "a"), frame=u.n.space.frame,
            support=u.n.space.support, sampling="cell_center"), components=(2, 0),
        measure=CellVolumeMeasure(), quadrature=CellMidpoint(), realization=DirectSpatialInteraction(1 << 20),
        source_scope="accepted" if target == "amr_system" else "issued")
    program.record_scalar("nonlocal_observation", program.sum_component(field, 0))
    candidate = program.value("candidate", u.n + program.dt * program.rhs(state=u.n, terms=[Flux()]), at=u.next.point)
    program.commit(u.next, candidate)
    program.step_strategy(FixedDt(.01))
    case.program(program)
    case.initials.add(InitialCondition(state=block[rho], value=BindArray(), projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(4, 3), periodic=PeriodicAxes(frame.axes)))
    if target == "amr_system":
        from pops.amr import AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer, Tag, Hysteresis, EqualityPolicy, ConflictPolicy, Buffer
        from pops.layouts import AMR
        from pops.lib.amr import StateTransfer
        from pops.math import ValueExpr
        from pops.time import every
        from pops.params import RuntimeParam
        transfer = AMRTransfer()
        transfer.state(block[rho], StateTransfer())
        threshold = case.param(RuntimeParam("threshold", default=.5))
        layout = AMR(grid=CartesianGrid(frame=frame, cells=(4, 3), periodic=PeriodicAxes(frame.axes)), hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
            tagging=AMRTagging(rules=(Tag(ValueExpr(block[rho])["a"] > case.value(threshold)), Buffer(cells=1)),
                hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
            regrid=AMRRegrid(schedule=every(1000, clock=program.clock)), transfer=transfer,
            execution=AMRExecution.synchronous())
    resolved = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    source = emit_cpp_program(resolved.time, model_graph=graph, target=target)
    assert "ctx.spatial_interaction" in source and "static_assert(pops::kNativeDimension == 2" in source
    assert "ctx.commit_many(" in source
    (tmp_path / "public.cpp").write_text(source)
    prefix = Path(sys.prefix)
    flags = ["/usr/bin/clang++", "-std=c++20", "-fsyntax-only", "-fno-fast-math", "-DPOPS_HAS_KOKKOS",
        "-DPOPS_NATIVE_DIM=2", "-DPOPS_HAS_MPI", "-DPOPS_RUNTIME_SHARED_EXCEPTION_ABI",
        "-I" + str(ROOT / "include"), "-I" + str(prefix / "include"), "-Xpreprocessor", "-fopenmp",
        "-I/opt/homebrew/opt/libomp/include", str(tmp_path / "public.cpp")]
    compiled = subprocess.run(flags, capture_output=True, text=True, timeout=60)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr


def test_actual_header_syntax_and_host_quadrature(tmp_path):
    prefix = Path(sys.prefix)
    source = ROOT / "tests/review/spatial_direct_interaction_host.cpp"
    binary = tmp_path / "direct_interaction"
    flags = ["/usr/bin/clang++", "-std=c++20", "-O0", "-fno-fast-math",
             "-DPOPS_HAS_KOKKOS", "-DPOPS_NATIVE_DIM=2", "-I" + str(ROOT / "include"),
             "-I" + str(prefix / "include"), "-Xpreprocessor", "-fopenmp",
             "-I/opt/homebrew/opt/libomp/include", str(source), "-L" + str(prefix / "lib"),
             "-lkokkoscore", "-lomp", "-Wl,-rpath," + str(prefix / "lib"), "-o", str(binary)]
    result = subprocess.run(flags, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    env = dict(os.environ, OMP_NUM_THREADS="2")
    result = subprocess.run([str(binary)], capture_output=True, text=True, env=env, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "spatial direct host assertions=59" in result.stdout


@pytest.mark.parametrize("mutation", ("version", "kernel_axis", "kernel_opaque", "measure", "quadrature", "realization", "duplicate_component", "bool_width", "budget_upper", "scope"))
def test_detached_contract_corruption_refuses_before_emission(mutation):
    from types import SimpleNamespace
    from pops.time.canonical_data import _json_ready
    program, source, value = build()
    attrs = _json_ready(value.attrs)
    if mutation == "version":
        attrs["contract"] = "pops.spatial-interaction@2"
    elif mutation == "kernel_axis":
        attrs["kernel"]["tree"] = ["x", 2]
    elif mutation == "kernel_opaque":
        attrs["kernel"]["tree"] = ["callback", "arbitrary"]
    elif mutation == "measure":
        attrs["measure"]["contract"] = "pops.unweighted@1"
    elif mutation == "quadrature":
        attrs["quadrature"] = "pops.unspecified@1"
    elif mutation == "realization":
        attrs["realization"] = "pops.implicit-poisson@1"
    elif mutation == "duplicate_component":
        attrs["components"] = [0, 0]
    elif mutation == "bool_width":
        attrs["ncomp"] = True
    elif mutation == "budget_upper":
        attrs["max_workspace_bytes"]["uint64_hex"] = "FFFFFFFFFFFFFFFF"
    elif mutation == "scope":
        attrs["source_scope"] = "assumed-candidate-barrier"
    corrupt = SimpleNamespace(op=value.op, vtype=value.vtype, inputs=value.inputs,
        attrs=attrs, block=value.block, point=value.point, space=value.space, id=value.id)
    lines = []
    with pytest.raises((ValueError, TypeError)):
        emit_spatial_interaction(corrupt, {source.id: "rho"}, lines,
            block_indices=program._block_indices(), target="amr_system")
    assert lines == []


def test_accepted_composite_scope_is_explicit_and_cannot_relabel_a_candidate():
    program, source, _ = build()
    accepted = program.spatial_interaction(source, SpatialInteractionKernel(2, lambda x, y: 1),
        output_space=FieldSpace("I", components=("a",), sampling="cell_center"),
        components=(0,), source_scope="accepted", measure=CellVolumeMeasure(), quadrature=CellMidpoint(),
        realization=DirectSpatialInteraction(4096))
    assert accepted.attrs["source_scope"] == "accepted"
    candidate = program.value("candidate", source + source)
    before = program._serialize()
    with pytest.raises(ValueError, match="State.n"):
        program.spatial_interaction(candidate, SpatialInteractionKernel(2, lambda x, y: 1),
            output_space=accepted.space, components=(0,), source_scope="accepted", measure=CellVolumeMeasure(),
            quadrature=CellMidpoint(), realization=DirectSpatialInteraction(4096))
    assert program._serialize() == before


def test_authored_history_preserves_exact_lag_clock_and_owner():
    model = pops.Model("history_density")
    declared = model.state("rho", components=("a",))
    case = pops.Case("history_convolution")
    block = case.block("density", model, states=(declared,))
    program = Program("history_convolution")
    u = program.state(block[declared])
    program.keep_history(u, depth=2)
    value = program.spatial_interaction(u.prev, SpatialInteractionKernel(1, lambda x, y: x[0] - y[0]),
        output_space=FieldSpace("I", components=("a",), sampling="cell_center"),
        measure=CellVolumeMeasure(), quadrature=CellMidpoint(), realization=DirectSpatialInteraction(4096))
    assert value.inputs[0].op == "history" and "history_contract" in value.inputs[0].attrs
    assert value.point == u.prev.point and value.block == u.block and value.clock == u.clock
    interaction_contract(value)


def test_accepted_composite_routes_to_preserved_carriers_and_active_commit_is_detached():
    facade = (ROOT / "src/runtime/amr/amr_system.cpp").read_text()
    getter = facade[facade.index("const MultiFab<Dim>& AmrSystem<Dim>::prepared_amr_block_state("):]
    getter = getter[:getter.index("\ntemplate", 1)]
    assert "p_->block_state(" in getter and "attempt_state" not in getter
    registry = facade[facade.index("  field_type& block_state(std::size_t block, std::size_t level) const {"):]
    registry = registry[:registry.index("\n  }", 1)]
    assert "return multiblock_hierarchy->state(block, level);" in registry
    hierarchy = (ROOT / "include/pops/runtime/amr/prepared_multiblock_hierarchy.hpp").read_text()
    assert "primary_->hierarchy().state(level) : additional_[block - 1].levels[level]" in hierarchy
    interaction = (ROOT / "include/pops/runtime/program/amr_program_context_spatial_interaction.inc").read_text()
    assert "accepted_composite ? facade_->prepared_amr_block_state(owner, level)" in interaction
    commit = (ROOT / "include/pops/runtime/program/amr_program_context_flux_expression_public.inc").read_text()
    active = commit[commit.index("  if (!active_attempt_states_.empty()) {"):]
    active = active[:active.index("    return;")]
    assert "active AMR Program commit cannot target an accepted block carrier" in active
    assert "*targets[candidate] = std::move(snapshots[candidate]);" in active
    engine = (ROOT / "include/pops/numerics/time/amr/levels/amr_subcycling_engine.hpp").read_text()
    recursive = engine[engine.index("    auto attempt = prepare_attempt_(root, \"begin-recursive\");"):]
    recursive = recursive[:recursive.index("\n private:")]
    assert recursive.index("advance_level_recursive_(") < recursive.index("publish_attempt_(")
    prepare = engine[engine.index("attempt->accepted_snapshot.emplace(hierarchy_->snapshot());"):]
    assert "attempt->candidates[block].emplace_back(hierarchy_->state(block, level));" in prepare


@pytest.mark.parametrize("scope", ("issued", "accepted"))
def test_state_n_cannot_be_relabelled_but_calculated_candidate_can(scope):
    from pops.time.points import TimePoint
    program, source, value = build()
    future = TimePoint(source.clock, 1)
    relabelled = program._replace_value(source, point=future)
    with pytest.raises(ValueError, match="State.n cannot be relabelled"):
        program.spatial_interaction(relabelled, SpatialInteractionKernel(2, lambda x, y: 1),
            output_space=value.space, measure=CellVolumeMeasure(), quadrature=CellMidpoint(),
            realization=DirectSpatialInteraction(4096), components=(2, 0), source_scope=scope)
    # The already authored map cannot retain a stale canonical source either.
    with pytest.raises(ValueError, match="State.n cannot be relabelled"):
        program._serialize()
    other, original, valid = build()
    computed = other.value("computed", original + original)
    computed = other._replace_value(computed, point=future)
    result = other.spatial_interaction(computed, SpatialInteractionKernel(2, lambda x, y: 1),
        output_space=valid.space, measure=CellVolumeMeasure(), quadrature=CellMidpoint(),
        realization=DirectSpatialInteraction(4096), components=(2, 0))
    assert result.point == future
    interaction_contract(result)


def test_serialization_revalidates_the_kernel_before_minting_ir17():
    from pops.time.canonical_data import _json_ready
    program, _, value = build()
    attrs = _json_ready(value.attrs)
    attrs["kernel"]["tree"] = ["x", 99]
    program._replace_value(value, attrs=attrs)
    with pytest.raises(ValueError):
        program._serialize()


def test_scratch_registry_node_is_allocated_and_voted_before_publication():
    source = (ROOT / "include/pops/runtime/program/amr_program_context_spatial_interaction.inc").read_text()
    assert source.index("staged.emplace(key, std::move(result))") < source.index("scratches_.find(key)")
    assert "interaction_phase(lane, [&] { staged.emplace" in source
    assert "std::is_nothrow_move_assignable_v<field_type>" in source
    assert "scratches_.insert(staged.extract(staged.begin()))" in source
    assert "scratches_.insert_or_assign" not in source
