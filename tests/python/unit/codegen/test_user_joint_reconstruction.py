"""Cross-state reconstruction: two outputs read three independent neighbor components."""

import pops
import pytest
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.numerics import reconstruction


def states():
    frame = Rectangle("joint_domain", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
    model = pops.Model("joint_law", frame=frame)
    a = model.species("a", state=("a0", "a1"))
    b = model.species("b", state=("b0", "b1", "b2"))
    return model, a, b


def test_cross_two_plus_three_body_is_a_real_vector_not_componentwise():
    _, a, b = states()
    policy = reconstruction.User(
        lambda sample: (
            sample(0)[0] + 0.25 * (sample(1, b)[2] - sample(-1, b)[0]),
            sample(0)[1] + 0.125 * (sample(1, b)[1] - sample(-1, a)[0]),
        ),
        state=a,
        sampling=(b,),
        formal_order=1,
    )
    from pops.numerics.reconstruction.user import authenticated_user_reconstruction

    assert authenticated_user_reconstruction(policy) is policy
    assert len(policy.expression) == 2
    assert policy.options["sample_offsets"] == (-1, 0, 1)
    assert policy.options["component_counts"] == (2, 3)


def joint_case(*, reverse=False, foreign=False, mixed=False, cross_offset=1):
    from pops.layouts import Uniform
    from pops.math import ddt, div
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan, FiniteVolume, riemann, variables
    from pops.params import RuntimeParam
    from pops.time import FixedDt

    model, a, b = states()
    frame = model.frame
    coefficient = model.value(model.param(RuntimeParam("mix", default=0.125)))
    if foreign:
        other = pops.Model("other", frame=frame)
        coefficient = other.value(other.param(RuntimeParam("mix", default=0.125)))
    owned = (a, b)
    total = sum(q for state in owned for q in state)
    fluxes = tuple(
        model.flux(
            "flux%d" % i,
            state=state,
            frame=frame,
            components={
                axis: tuple((1 if axis == frame.x else -0.5) * total + q for q in state)
                for axis in frame.axes
            },
            waves={axis: (6.0,) * 2 for axis in frame.axes} if i == 0 else None,
        )
        for i, state in enumerate(owned)
    )
    rates = tuple(
        model.rate("rate%d" % i, equation=ddt(state) == -div(flux))
        for i, (state, flux) in enumerate(zip(owned, fluxes, strict=True))
    )
    policies = (
        reconstruction.User(
            lambda s: (
                s(0)[0] + coefficient * (s(cross_offset, b)[2] - s(-1, b)[0]),
                s(0)[1] + coefficient * (s(1, b)[1] - s(-1)[0]),
            ),
            state=a,
            sampling=(b,),
            formal_order=1,
        ),
        reconstruction.User(
            lambda s: tuple(
                s(0)[j] + coefficient * (s(2, a)[j % 2] - s(-1, a)[(j + 1) % 2]) for j in range(3)
            ),
            state=b,
            sampling=(a,),
            formal_order=1,
        ),
    )
    if mixed:
        policies = (policies[0], reconstruction.User(lambda s: s(0), formal_order=1))
    case = pops.Case("joint_reconstruction")
    order = (1, 0) if reverse else (0, 1)
    blocks = {i: case.block("block%d" % i, model, states=(owned[i],)) for i in order}
    for i in order:
        plan = DiscretizationPlan()
        plan.rates.add(
            rates[i],
            FiniteVolume(
                flux=fluxes[i],
                variables=variables.Conservative(owned[i]),
                reconstruction=policies[i],
                riemann=riemann.Rusanov(),
                sampling=(owned[1 - i],),
            ),
        )
        case.numerics(plan, block=blocks[i])
    program = pops.Program("steps")
    temporal = {i: program.state(blocks[i][owned[i]]) for i in order}
    bindings = {owned[i]: temporal[i].n for i in order}
    for i in order:
        rhs = rates[i](temporal[i].n, bindings=bindings)
        program.commit(
            temporal[i].next,
            program.value("next", temporal[i].n + program.dt * rhs, at=temporal[i].next.point),
        )
    program.step_strategy(FixedDt(1.0e-3))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 8), periodic=PeriodicAxes(frame.axes)))
    return pops.resolve(pops.validate(case), layout=layout)


@pytest.mark.parametrize("reverse,mixed", [(False, False), (True, False), (False, True)])
def test_joint_group_lowering_uses_exact_packing_and_row_capture_context(reverse, mixed):
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_emit_principal_numerics import emit_principal_reconstruction
    from pops.codegen.program_codegen import emit_cpp_program

    resolved = joint_case(reverse=reverse, mixed=mixed)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    entry = graph.model_for_block("block0")._m._principal_groups[0]
    source, _ = emit_principal_reconstruction(entry)
    assert "n_components=5" in source
    assert "{{0,1,2,3,4}}" in source
    assert ("{{2,3,4}}" if mixed else "{{2,3,4,0,1}}") in source
    assert "n_ghost=%d" % (2 if mixed else 3) in source
    assert "parameter_sets[0]" in source
    for target in ("system", "amr_system"):
        program = emit_cpp_program(resolved.time, model=graph, target=target)
        assert "stencil_face_state" in program
        assert "numerical_face_courant()" in program


def test_joint_foreign_model_capture_is_refused():
    from pops.codegen.program_models import ProgramModelGraph

    with pytest.raises(ValueError, match="another model|owner|belongs"):
        ProgramModelGraph.from_resolved_blocks(joint_case(foreign=True).blocks)


def test_joint_undeclared_state_and_bad_shape_are_refused():
    _, a, b = states()
    with pytest.raises(ValueError, match="undeclared"):
        reconstruction.User(lambda s: (s(0, b)[0], s(0)[1]), state=a, formal_order=1)
    with pytest.raises(TypeError, match="per output"):
        reconstruction.User(lambda s: (s(0)[0],), state=a, formal_order=1)


def test_compiled_joint_row_mapping_capture_rebind_and_orientation(tmp_path):
    """Executes emitted C++ policies; this is not a native mesh/MPI qualification."""
    import ctypes
    import shutil
    import subprocess
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_emit_principal_numerics import emit_principal_reconstruction

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    graph = ProgramModelGraph.from_resolved_blocks(joint_case(reverse=True).blocks)
    entry = graph.model_for_block("block0")._m._principal_groups[0]
    definition, _ = emit_principal_reconstruction(entry)
    source = (
        """#include <array>
#include <cmath>
#include <limits>
#define POPS_HD
namespace pops { using Real=double;
struct RuntimeParams { double v; double get(int) const { return v; } };
template<class T> concept ReconstructionPolicy=true;
template<class T> constexpr bool stencil_envelope_fits_storage=true;
}
struct Sample { int orientation; double operator()(int offset,int c) const {
 return 10.*(c+1)+orientation*offset*(c+2); } };
"""
        + definition
        + """
extern "C" void evaluate(int orientation,double alpha,double beta,double* output) {
 %s policy{}; policy.parameter_sets[0].v=alpha; policy.parameter_sets[1].v=beta;
 auto result=policy.stencil_face_state(Sample{orientation});
 for(int i=0;i<5;++i) output[i]=result[i];
}
"""
        % (entry["cpp_name"] + "Reconstruction")
    )
    cpp, library = tmp_path / "joint.cpp", tmp_path / "joint.so"
    cpp.write_text(source)
    result = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-shared",
            "-fPIC",
            "-O2",
            "-fno-fast-math",
            str(cpp),
            "-o",
            str(library),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    evaluate = ctypes.CDLL(str(library)).evaluate
    evaluate.argtypes = [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    for orientation in (-1, 1):
        for alpha, beta in ((0.125, 0.25), (0.5, -0.125)):

            def sample(o, c, orientation=orientation):
                return 10.0 * (c + 1) + orientation * o * (c + 2)

            expected = [
                sample(0, 0) + alpha * (sample(1, 4) - sample(-1, 2)),
                sample(0, 1) + alpha * (sample(1, 3) - sample(-1, 0)),
            ]
            expected += [
                sample(0, j + 2) + beta * (sample(2, j % 2) - sample(-1, (j + 1) % 2))
                for j in range(3)
            ]
            output = (ctypes.c_double * 5)()
            evaluate(orientation, alpha, beta, output)
            assert list(output) == expected


def test_joint_active_nonfinite_and_lazy_branch_are_observed(tmp_path):
    import ctypes
    import shutil
    import subprocess
    from types import SimpleNamespace
    from pops.math import where, minimum
    from pops.codegen.user_reconstruction_lowering import emit_user_reconstruction_policy

    _, a, b = states()
    descriptor = reconstruction.User(
        lambda s: (
            where(s(0)[0] > 0, lambda: s(1, b)[2], lambda: 1 / (s(0)[0] - s(0)[0])),
            minimum(s(0)[1], s(1, b)[1]),
        ),
        state=a,
        sampling=(b,),
        formal_order=1,
    )
    source = (
        """#include <array>
#include <cmath>
#include <limits>
#define POPS_HD
namespace Kokkos { using std::fmin; }
namespace pops { using Real=double; template<class T> concept ReconstructionPolicy=true;
template<class T> constexpr bool stencil_envelope_fits_storage=true; }
struct Sample { const double* values; int* calls;
 double operator()(int offset,int component) const { ++*calls; return values[component]; } };
"""
        + emit_user_reconstruction_policy(SimpleNamespace(_user_reconstruction=descriptor))
        + """
extern "C" void evaluate(const double* values,int* calls,double* output) {
 auto result=pops_generated::UserReconstructionPolicy{}.stencil_face_state(Sample{values,calls});
 output[0]=result[0]; output[1]=result[1]; }
"""
    )
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    cpp, library = tmp_path / "lazy.cpp", tmp_path / "lazy.so"
    cpp.write_text(source)
    result = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-shared",
            "-fPIC",
            "-O2",
            "-fno-fast-math",
            str(cpp),
            "-o",
            str(library),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    evaluate = ctypes.CDLL(str(library)).evaluate
    evaluate.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
    ]
    import math

    for values, accepted in (
        ((1, 2, 3, 4, 5), True),
        ((-1, 2, 3, 4, 5), False),
        ((1, float("nan"), 3, 4, 5), False),
    ):
        output = (ctypes.c_double * 2)()
        calls = ctypes.c_int()
        evaluate((ctypes.c_double * 5)(*values), ctypes.byref(calls), output)
        if accepted:
            assert list(output) == [5.0, 2.0]
        else:
            assert all(math.isnan(value) for value in output)


def test_joint_source_handle_substitution_is_not_authenticated_by_local_name():
    from pops.numerics.reconstruction.user import authenticated_user_reconstruction

    model, a, b = states()
    other = pops.Model("another_model", frame=model.frame)
    foreign = other.species("b", state=("b0", "b1", "b2"))
    policy = reconstruction.User(
        lambda s: (s(0, b)[0], s(0)[1]), state=a, sampling=(b,), formal_order=1
    )
    policy.options["sampling"] = (foreign,)
    with pytest.raises(ValueError, match="authority|owner|resolve"):
        authenticated_user_reconstruction(policy)


@pytest.mark.parametrize("foreign_instance", (False, True))
def test_joint_single_state_two_blocks_keep_capture_owner(foreign_instance):
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.user_reconstruction_lowering import emit_user_reconstruction_policy
    from pops.domain import Rectangle
    from pops.layouts import Uniform
    from pops.math import ddt, div
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan, FiniteVolume, riemann, variables
    from pops.params import RuntimeParam
    from pops.time import FixedDt

    frame = Rectangle("domain", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
    model = pops.Model("single", frame=frame)
    state = model.species("u", state=("a", "b"))
    parameter = model.param(RuntimeParam("alpha", default=0.25))
    flux = model.flux(
        "flux",
        state=state,
        frame=frame,
        components={axis: tuple(state) for axis in frame.axes},
        waves={axis: (1.0, 1.0) for axis in frame.axes},
    )
    rate = model.rate("rate", equation=ddt(state) == -div(flux))
    case = pops.Case("instances")
    blocks = [case.block(name, model) for name in ("first", "second")]
    program = pops.Program("step")
    for block in blocks:
        capture = model.value(parameter)
        if foreign_instance:
            # Adversarial already-qualified common-IR read; model.value itself
            # rejects an instance passed to its declaration-only registry.
            from pops._ir.values import RuntimeParamRef
            capture = RuntimeParamRef("alpha", .25, handle=blocks[0][parameter])
        descriptor = reconstruction.User(
            lambda s, capture=capture: (s(0)[0] + capture * (s(1)[1] - s(-1)[1]), s(0)[1]),
            state=state,
            formal_order=1,
        )
        plan = DiscretizationPlan()
        plan.rates.add(
            rate,
            FiniteVolume(
                flux=flux,
                variables=variables.Conservative(state),
                reconstruction=descriptor,
                riemann=riemann.Rusanov(),
            ),
        )
        case.numerics(plan, block=block)
        temporal = program.state(block[state])
        rhs = rate(temporal.n)
        program.commit(
            temporal.next,
            program.value("next", temporal.n + program.dt * rhs, at=temporal.next.point),
        )
    program.step_strategy(FixedDt(0.001))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 8), periodic=PeriodicAxes(frame.axes)))

    def lower():
        return ProgramModelGraph.from_resolved_blocks(
            pops.resolve(pops.validate(case), layout=layout).blocks
        )

    if foreign_instance:
        with pytest.raises(ValueError, match="another block|owner|belongs"):
            lower()
    else:
        graph = lower()
        for name in ("first", "second"):
            source = emit_user_reconstruction_policy(graph.model_for_block(name))
            assert "params.get(0)" in source and "stencil_face_state" in source
