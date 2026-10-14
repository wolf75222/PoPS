"""Typed spatial sampling ports and actual PDE Program construction; no native claim."""
import pytest

import pops
from pops.fields import CellMidpoint
from pops.time._program.spatial_interaction import validate_cell_field, validate_cell_mean
from examples.api_v02.models_M15_M28 import build_m26, build_m27


def _state(components=("q",)):
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D
    frame = CartesianDomain("sampling", lower=(0.,), upper=(1.,)).frame(Cartesian1D())
    model = pops.Model("sampling", frame=frame)
    q = model.state("q", components=components, sampling="cell_average")
    case = pops.Case("sampling")
    block = case.block("density", model, states=(q,))
    p = pops.Program("sampling")
    return p, p.state(block[q])


def test_midpoint_projection_retains_owner_point_and_exact_source():
    p, q = _state()
    field = p.cell_field(q.n, quadrature=CellMidpoint())
    mean = p.cell_mean(field, template=q.n, quadrature=CellMidpoint())
    assert validate_cell_field(field) is q.n
    assert validate_cell_mean(mean) is q.n
    assert mean.inputs == (field, q.n)
    assert mean.point == field.point == q.n.point
    assert mean.block == field.block == q.n.block
    assert field.space.sampling == "cell_center"
    assert mean.space.sampling == "cell_average"
    assert field.state_ref is None and mean.state_ref == q.n.state_ref


def test_sampling_is_explicit_and_rejects_temporal_or_owner_coercion():
    p, q = _state()
    with pytest.raises(TypeError):
        p.cell_field(q.n)
    with pytest.raises(TypeError, match="CellMidpoint"):
        p.cell_field(q.n, quadrature="midpoint")
    field = p.cell_field(q.n, quadrature=CellMidpoint())
    later = p.value("later", q.n, at=q.next.point)
    with pytest.raises(ValueError, match="evaluation point"):
        p.cell_mean(field, template=later, quadrature=CellMidpoint())
    other, r = _state()
    with pytest.raises(ValueError):
        p.cell_mean(field, template=r.n, quadrature=CellMidpoint())


def test_loaded_projection_cannot_change_the_sampled_formula():
    p, q = _state()
    field = p.cell_field(q.n, quadrature=CellMidpoint())
    mean = p.cell_mean(field, template=q.n, quadrature=CellMidpoint())
    attrs = dict(mean.attrs)
    attrs["expressions"] = ()
    changed = p._replace_value(mean, attrs=attrs)
    with pytest.raises(ValueError, match="quadrature evaluations"):
        validate_cell_mean(changed)


def test_differential_port_preserves_stage_and_cannot_mix_stages():
    p, q = _state()
    field = p.cell_field(q.n, quadrature=CellMidpoint())
    scratch = p.scalar_field("lap", like=field)
    lap = p.laplacian(scratch, field)
    assert lap.space == field.space and lap.point == field.point and lap.block == field.block
    later = p.value("later", q.n, at=q.next.point)
    later_field = p.cell_field(later, quadrature=CellMidpoint())
    with pytest.raises(ValueError, match="owner/point"):
        p.laplacian(scratch, later_field)


def test_computed_scratch_infers_full_width_and_rejects_explicit_coercion():
    p, q = _state(("a", "b"))
    field = p.cell_field(q.n, quadrature=CellMidpoint())
    scratch = p.scalar_field(like=field)
    assert scratch.attrs["ncomp"] == 2
    with pytest.raises(ValueError, match="component shape"):
        p.scalar_field(like=field, ncomp=1)


def test_region_midpoint_port_has_exact_scope_and_checked_stencil_effect():
    from pops.codegen.program_emit_structured_control import _check_region_effects, emit_region_laplacian
    from pops.time._program.spatial_interaction import validate_computed_laplacian
    p, q = _state(("a", "b"))
    escaped = []
    def step(builder, candidate):
        field = builder.cell_field(candidate, quadrature=CellMidpoint())
        output = builder.scalar_field(like=field)
        lap = builder.laplacian(output, field)
        escaped.append((field, lap))
        return builder.cell_mean(lap, template=candidate, quadrature=CellMidpoint())
    result = p.iterate(q.n, step, lambda builder, value: builder.boolean(False), max_steps=2)
    _check_region_effects(result.attrs["body_block"])
    field, lap = escaped[0]
    assert validate_computed_laplacian(lap) == lap.inputs
    with pytest.raises(ValueError, match="region"):
        p.cell_mean(field, template=q.n, quadrature=CellMidpoint())
    with pytest.raises(NotImplementedError, match="Uniform periodic"):
        emit_region_laplacian(lap, {}, [], owner=0, target="amr_system")
    lines = []
    emit_region_laplacian(lap, {lap.inputs[0].id: "output", field.id: "input"}, lines,
                          owner=0, target="system")
    source = "\n".join(lines)
    assert source.index("region Laplacian preflight") < source.index("prepare_mesh_boundary_session")
    assert "topology().is_periodic" in source
    assert "component < components" in source and "nonfinite provisional stencil output" in source
    assert source.index("all_reduce_max") < source.index("StepAttemptRejected")


def test_region_laplacian_does_not_admit_unqualified_scratch():
    from pops.codegen.program_emit_structured_control import _check_region_effects
    p, q = _state()
    first, second = p.scalar_field(), p.scalar_field()
    unqualified = p.laplacian(first, second)
    with pytest.raises(NotImplementedError, match="lane-uniform failure contract"):
        _check_region_effects((unqualified,))


@pytest.mark.parametrize("builder", (build_m26, build_m27))
def test_complete_multicell_pde_resolves_and_emits_original_spatial_operations(builder):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    spec = builder(cells=16, dimension=2, steps=4)
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    source = emit_cpp_program(resolved.time, model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert resolved.resolved_dimension == 2
    initial = next(iter(spec.initial_values.values()))
    assert initial.shape == (1, 1, 16), "one scalar at sixteen actual spatial cells"
    assert "ctx.laplacian(" in source
    assert "ctx.commit_many(" in source
    assert "finite_linear_apply<12, 12>" not in source
    if builder is build_m26:
        assert "ctx.spatial_interaction(" in source and "Kokkos::cos" in source
    else:
        assert source.count("ctx.laplacian(") == 2
    assert not spec.invariants({"density": initial})["passed"], "unchanged data cannot pass evolution reception"
