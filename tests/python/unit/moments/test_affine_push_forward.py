"""Independent discrete-measure oracle for the public polynomial body."""
import itertools
import math

import numpy as np
import pytest

from pops.moments import affine_push_forward


def basis(dimension, order):
    return tuple(index for index in itertools.product(range(order + 1), repeat=dimension)
                 if sum(index) <= order)


def particle_moments(points, weights, indices):
    return np.array([np.dot(weights, np.prod(points ** np.asarray(index), axis=1))
                     for index in indices])


@pytest.mark.parametrize("dimension,order", [(1, 6), (2, 6), (3, 3)])
@pytest.mark.parametrize("permuted", [False, True])
def test_push_forward_matches_independent_transformed_particles(dimension, order, permuted):
    indices = basis(dimension, order)
    if permuted:
        indices = indices[1::2] + indices[::2][::-1]
    points = np.random.default_rng(119).uniform(-.7, .8, (11, dimension))
    weights = np.arange(1., 12.) / 31.
    if dimension == 1:
        matrix = np.array([[-.8]])
    elif dimension == 2:
        angle = .37
        matrix = np.array([[math.cos(angle), math.sin(angle)],
                           [-math.sin(angle), math.cos(angle)]])
    else:
        matrix = np.array([[1., .2, 0.], [0., -.7, .1], [.3, 0., .8]])
    offset = np.linspace(-.2, .3, dimension)
    raw = tuple(particle_moments(points, weights, indices))
    result = affine_push_forward(raw, indices=indices, matrix=matrix, offset=offset)
    expected = particle_moments(points @ matrix.T + offset, weights, indices)
    np.testing.assert_allclose(result, expected, atol=8e-15, rtol=5e-14)
    zero = indices.index((0,) * dimension)
    assert result[zero] is raw[zero]


def test_density_identity_preserves_signed_zero_without_reading_unused_map():
    rho = -0.
    result = affine_push_forward((rho,), indices=((0,),), matrix=((float("nan"),),),
                                 offset=(float("inf"),))
    assert result[0] is rho
    assert math.copysign(1., result[0]) == -1.


@pytest.mark.parametrize("indices,matrix,offset,match", [
    (((0,), (2,)), ((1.,),), (0.,), "missing raw moments"),
    (((0, 0), (1, 0)), ((1., 0.), (0., 1.)), (0., 0.), "missing raw moments"),
    (((0,), (0,)), ((1.,),), (0.,), "distinct"),
    (((0,), (True,)), ((1.,),), (0.,), "integer"),
    (((0,), (-1,)), ((1.,),), (0.,), "integer"),
    (((0,), (1, 0)), ((1.,),), (0.,), "rank"),
    (((0,),), ((1., 0.),), (0.,), "square"),
    (((0,),), ((1.,),), (), "offset"),
])
def test_invalid_basis_and_dimensions_are_explicit(indices, matrix, offset, match):
    with pytest.raises(ValueError, match=match):
        affine_push_forward((1.,) * len(indices), indices=indices, matrix=matrix, offset=offset)


def test_equal_arity_for_explicit_sequences():
    for moments in ((1., 2.), np.array([1., 2.])):
        with pytest.raises(ValueError, match="equal length"):
            affine_push_forward(moments, indices=((0,),), matrix=((1.,),), offset=(0.,))


@pytest.mark.parametrize("dimension,order", [(1, 6), (2, 5)])
def test_same_body_on_declarations_stage_and_local_unknown(dimension, order):
    import pops
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.frames import Cartesian2D
    from pops.solvers.nonlinear import LocalNewton
    from pops.time import FailRun, LocalResidual

    indices = basis(dimension, order)[::-1]
    model = pops.Model("affine_library", frame=Cartesian2D())
    state = model.state("U", components=tuple("m" + "_".join(map(str, i)) for i in indices))
    matrix = tuple(tuple(.8 if i == j else .1 for j in range(dimension))
                   for i in range(dimension))

    def body(value, shift):
        return affine_push_forward(value, indices=indices, matrix=matrix,
                                   offset=(shift,) * dimension)

    model.source("push", on=state, value=body(state, .2))
    case = pops.Case("affine_reuse")
    block = case.block("matter", model)
    program = pops.Program("affine_composition")
    q = program.state(block[state])
    first = program.value("first", body(q.n, program.dt), at=q.n.point)
    second = program.value("second", body(first, -.1), at=q.next.point)

    def residual(p, candidate, target):
        mapped = body(candidate, .2)
        return tuple(mapped[i] - target[i] for i in range(len(indices)))

    solved = program.solve(LocalResidual(residual, second, captures={"target": second}),
                           solver=LocalNewton()).consume(action=FailRun())
    program.commit(q.next, solved)
    assert program.validate()
    lowered, _ = lower_and_validate(model, facade=model)
    source = emit_cpp_program(program, model=lowered)
    assert "pointwise" in source
    assert "affine_velocity_push_forward" not in source
    assert "PreparedLocal" in source


def test_migrated_consumer_emits_complete_loader_and_program_without_recipe():
    import pops
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from tests.python.integration.runtime.test_conditional_consumers_runtime import consumer_case
    case, layout, _ = consumer_case("affine_library")
    resolved = pops.resolve(pops.validate(case), layout=layout)
    block = resolved.blocks[0]
    lowered, _ = lower_and_validate(block.model, state_space=block.state_spaces[0],
        resolved_operations=block.resolved_operations, numerics=block.numerics)
    source = lowered._m.emit_cpp_native_loader(name="AffineLibrary", target="system",
        consumer_owner_qid=block.instance_owner_qid)
    assert "void pops_install_native(" in source
    assert "affine_velocity_push_forward" not in source
    program_source = emit_cpp_program(resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert "affine_velocity_push_forward" not in program_source
    assert "?" in program_source  # lazy coefficient/domain branches remain native


@pytest.mark.parametrize("dimension,order", [(1, 6), (2, 5)])
def test_native_reception_fixture_is_a_closed_public_program(dimension, order):
    import pops
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from tests.python.integration.runtime.test_affine_push_forward_runtime import affine_case
    case, layout, *_ = affine_case(dimension, order, True)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    source = emit_cpp_program(resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert "affine_velocity_push_forward" not in source
    assert "solve_prepared_local_nonlinear" in source
