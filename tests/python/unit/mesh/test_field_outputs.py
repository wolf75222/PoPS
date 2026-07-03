#!/usr/bin/env python3
"""ADC-534: typed field outputs, composed multi-block RHS, and the field-solve guards.

Spec 5 sec.5.5 / sec.9 makes the outputs of an elliptic field solve typed descriptors
(:mod:`pops.fields.outputs`), lets the right-hand side compose several typed sources
(``ChargeDensity(...) + FixedSource(...)`` -> :class:`~pops.fields.rhs.SumRHS`), and rejects a
bare-string ``solver=`` on a :class:`~pops.fields.FieldProblem` (pointing at the typed
``pops.solvers.GeometricMG``). The pre-runtime refusals -- FFT on an AMR layout, FFT with a
non-periodic boundary, a required-but-missing output, a required-but-missing nullspace -- surface
through ``available`` / ``validate`` before any compile.

Pure Python: it imports the inert authoring packages only. Skips when ``pops`` cannot be imported.
"""

import pytest

pops = pytest.importorskip("pops")

from pops.fields import (  # noqa: E402
    ConstantNullspace, DerivedField, FieldOutput, FieldProblem, GradientOutput, PoissonProblem)
from pops.fields.bcs import Dirichlet, Periodic  # noqa: E402
from pops.fields.rhs import ChargeDensity, FixedSource, SumRHS  # noqa: E402
from pops.ir.expr import Var  # noqa: E402
from pops.math import laplacian, unknown  # noqa: E402
from pops.solvers.elliptic import FFT, GeometricMG  # noqa: E402


def _problem(solver, *bcs, **kw):
    phi = unknown("phi")
    rho = Var("rho", "cons")
    return PoissonProblem(unknown=phi, equation=(-laplacian(phi) == rho),
                          bcs=bcs, solver=solver, **kw)


# --- pops.fields.outputs: the typed field-output descriptors ----------------------------------
def test_field_outputs_construct_and_inspect():
    phi = FieldOutput("phi")
    E = GradientOutput("E", "phi")
    J = DerivedField("J", "ohm", source="phi")
    assert phi.options() == {"name": "phi", "recipe": "field", "source": "phi"}
    assert E.options()["recipe"] == "grad_phi" and E.options()["source"] == "phi"
    assert E.capabilities().to_dict()["vector"] is True
    assert J.options()["recipe"] == "ohm"
    for out in (phi, E, J):
        assert out.category == "field_output"
        assert isinstance(out.inspect(), dict)
        assert out.requirements().to_dict()["field"] == "phi"


def test_fields_package_exports_outputs():
    assert "outputs" in pops.fields.__all__
    assert pops.fields.outputs.GradientOutput is GradientOutput


# --- composed / multi-block RHS ---------------------------------------------------------------
def test_composed_multiblock_rhs():
    cd = ChargeDensity.from_blocks("ions", "electrons")
    composed = cd + FixedSource("rho_background")
    assert isinstance(composed, SumRHS)
    assert composed.options()["n_terms"] == 2
    assert composed.options()["terms"] == ["charge_density", "fixed_source"]
    req = composed.requirements().to_dict()
    assert req["blocks"] == ["ions", "electrons"]
    assert req["aux_fields"] == ["rho_background"]


def test_sumrhs_flattens_nested_sums():
    total = (ChargeDensity.from_blocks("ions") + FixedSource("bg")
             + ChargeDensity.from_blocks("beam"))
    assert isinstance(total, SumRHS)
    assert len(total.terms) == 3  # flattened, not nested


def test_sumrhs_rejects_non_rhs_term():
    with pytest.raises(TypeError):
        SumRHS(ChargeDensity.from_blocks("ions"), object())
    with pytest.raises(ValueError):
        SumRHS()


# --- NEGATIVE: a bare-string solver is rejected pointing at the typed descriptor ---------------
def test_string_solver_rejected_points_at_geometric_mg():
    with pytest.raises(TypeError) as exc:
        _problem("geometric_mg")
    msg = str(exc.value)
    assert "solver='geometric_mg'" in msg
    assert "GeometricMG" in msg
    # A typed solver is accepted and validates.
    assert _problem(GeometricMG()).validate() is True


# --- NEGATIVE: FFT refuses an AMR layout (via context) ----------------------------------------
def test_fft_refuses_amr_layout():
    class _AMRLayout:
        def capabilities(self):
            return {"layout": "amr"}

    status = FFT().available({"layout": _AMRLayout()})
    assert status.status == "no"
    assert "GeometricMG" in " ".join(status.alternatives)
    # The FieldProblem layout guard surfaces the same refusal at validate.
    with pytest.raises(ValueError):
        _problem(FFT(), Periodic()).validate(context={"layout": _AMRLayout()})


# --- NEGATIVE: FFT refuses a non-periodic boundary --------------------------------------------
def test_fft_refuses_non_periodic_bc():
    with pytest.raises(ValueError) as exc:
        _problem(FFT(), Dirichlet()).validate()
    assert "periodic" in str(exc.value).lower()
    # FFT with a periodic boundary validates (no false positive).
    assert _problem(FFT(), Periodic()).validate() is True


# --- NEGATIVE: a required output that the problem does not declare -----------------------------
def test_missing_required_output_refused():
    ctx = {"required_outputs": ["E"]}
    with pytest.raises(ValueError) as exc:
        _problem(GeometricMG(), outputs=[FieldOutput("phi")]).validate(context=ctx)
    assert "E" in str(exc.value)
    # Declaring the output satisfies the requirement.
    ok = _problem(GeometricMG(), outputs=[FieldOutput("phi"), GradientOutput("E", "phi")])
    assert ok.validate(context=ctx) is True


# --- NEGATIVE: a singular operator that declares no nullspace ----------------------------------
def test_required_nullspace_refused():
    ctx = {"requires_nullspace": True}
    with pytest.raises(ValueError) as exc:
        _problem(GeometricMG()).validate(context=ctx)
    assert "nullspace" in str(exc.value).lower()
    # Declaring a ConstantNullspace satisfies the singular operator.
    ok = _problem(GeometricMG(), nullspace=ConstantNullspace())
    assert ok.validate(context=ctx) is True
    assert ok.requirements().to_dict()["nullspace"] == "ConstantNullspace"


# --- NO FALSE POSITIVE: an unspecified context never triggers the opt-in guards ----------------
def test_unspecified_context_does_not_refuse():
    prob = _problem(GeometricMG())
    assert prob.validate() is True
    assert prob.validate(context={}) is True


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
