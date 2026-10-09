"""C04 classifications come from real authoring/capability gates, not message matching."""
import pytest
import pops

from pops.blockage import (Blockage, BlockageClass, BlockageNotImplementedError,
                           BlockageValueError)
from pops.codegen.solvers.solver_cpp import _SolverCppLowering
from pops.codegen.loader import CompiledModel
from pops.descriptors import Availability
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.math import CoeffGradient
from pops.numerics import Diffusion
from pops.numerics.riemann import HLL, Rusanov
from pops.numerics.riemann._contract import riemann_capability_contract
from pops.numerics.riemann.availability import flux_available
from pops.runtime.routes import check_riemann_requirement_contract
from pops.time import Program, SolveRequestError, SolveUnknown


def test_missing_provider_does_not_prove_a_mathematical_contradiction():
    physical = dict(
        so_path="/no/such/c04-model.so", backend="production",
        cons_names=["q"], cons_roles=["other"], prim_names=[], n_vars=1,
        gamma=None, n_aux=0, params={}, caps={"cpu": True},
        abi_key="SIG|c++|c++23", model_hash="c04", cxx="c++", std="c++23",
        native_dimension=2, target="system",
    )
    absent = CompiledModel(**physical, wave_speeds=False)
    present = CompiledModel(**physical, wave_speeds=True,
                            wave_speed_provider="explicit_pair")
    assert absent.model_hash == present.model_hash
    hll = HLL()
    with pytest.raises(ValueError) as caught:
        check_riemann_requirement_contract(
            riemann_capability_contract(hll), absent, "test.source", flux=hll)
    assert not isinstance(caught.value, BlockageValueError)
    assert check_riemann_requirement_contract(
        riemann_capability_contract(hll), present, "test.source", flux=hll) is None
    available = flux_available(hll, {"model": absent})
    assert not available and available.missing == ["wave_speeds"]
    assert available.blockage is None
    assert flux_available(hll, {"model": present}).ok
    rusanov = Rusanov()
    assert check_riemann_requirement_contract(
        riemann_capability_contract(rusanov), absent, "test.source", flux=rusanov) is None


def _diffusion(diagonal):
    frame = Rectangle("c04-diffusion", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("c04-diffusion", frame=frame)
    state = model.state("U", components=("u",))
    flux = model.diffusive_flux(
        "D", state=state,
        value=CoeffGradient(state[0], ((diagonal, 0.), (0., 0.))),
    )
    return flux


def test_negative_diagonal_contradicts_the_selected_monotone_method():
    flux = _diffusion(-0.01)
    with pytest.raises(ValueError) as caught:
        Diffusion(flux=flux)
    assert isinstance(caught.value, BlockageValueError)
    assert caught.value.blockage.to_data() == {
        "schema_version": 1,
        "classification": "MATH",
        "phase": "validate",
        "source": flux.qualified_id,
        "cause": "incompatible_method_hypothesis",
        "capability": "nonnegative_diagonal_diffusion",
    }
    assert Diffusion(flux=_diffusion(0.01)).validate() is True
    with pytest.raises(ValueError) as invalid_source:
        _diffusion(float("nan"))
    assert not isinstance(invalid_source.value, BlockageValueError)


def test_unrepresentable_typed_unknown_is_expr_but_invalid_source_is_only_validation():
    program = Program("c04-unknown")
    field = program.scalar_field("rhs")
    assert SolveUnknown("field", field).template is field
    scalar = program.norm2(field)
    with pytest.raises(SolveRequestError) as caught:
        SolveUnknown("scalar", scalar)
    assert caught.value.blockage.to_data() == {
        "schema_version": 1,
        "classification": "EXPR",
        "phase": "author",
        "source": "SolveUnknown:scalar",
        "cause": "unknown_value_space_unrepresentable",
        "capability": "scalar",
    }
    with pytest.raises(SolveRequestError) as invalid:
        SolveUnknown("bad", object())
    assert invalid.value.blockage is None


def test_blockage_schema_is_exact_and_scope_is_not_inferred():
    assert Blockage(BlockageClass.SCOPE, "author", "request:hereditary",
                    "domain_extension_required").to_data()["classification"] == "SCOPE"
    with pytest.raises(TypeError):
        Blockage("MATH", "validate", "source", "missing_model_capability")
    with pytest.raises(ValueError):
        Blockage(BlockageClass.IMPL, "", "source", "lowering_unavailable")
    with pytest.raises(TypeError):
        Availability.no("not typed", blockage={"classification": "MATH"})


def test_authored_reduction_is_represented_but_custom_solver_lowering_is_absent():
    program = Program("c04-reduction")
    field = program.scalar_field("rhs")
    reduction = program.sum(field)
    assert reduction.op == "reduce" and reduction.attrs["kind"] == "sum"
    lowerer = _SolverCppLowering.__new__(_SolverCppLowering)
    lowerer._var = {field.id: "rhs"}
    with pytest.raises(NotImplementedError) as caught:
        lowerer._emit_reduce(reduction, [])
    assert isinstance(caught.value, BlockageNotImplementedError)
    assert caught.value.blockage.to_data() == {
        "schema_version": 1,
        "classification": "IMPL",
        "phase": "lower",
        "source": "solver_ir:%s" % reduction.name,
        "cause": "lowering_unavailable",
        "capability": "reduction:sum",
    }
