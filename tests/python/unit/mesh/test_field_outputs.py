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
    ConstantNullspace, DerivedField, FieldOutput, GradientOutput, PoissonProblem)
from pops.fields.bcs import Dirichlet, Periodic  # noqa: E402
from pops.fields.rhs import ChargeDensity, FixedSource, SumRHS  # noqa: E402
from pops.ir.expr import Var  # noqa: E402
from pops.math import laplacian, unknown  # noqa: E402
from pops.model import (  # noqa: E402
    AmbiguousReferenceError,
    Handle,
    MissingOwnershipError,
    Module,
    OwnerPath,
)
from pops.output import OutputPolicy  # noqa: E402
from pops.problem import Problem  # noqa: E402
from pops.solvers.elliptic import FFT, GeometricMG  # noqa: E402


def _problem(solver, *bcs, **kw):
    phi = unknown("phi")
    rho = Var("rho", "cons")
    return PoissonProblem(unknown=phi, equation=(-laplacian(phi) == rho),
                          bcs=bcs, solver=solver, **kw)


def _shared_field(name):
    return Handle(name, kind="field", owner=OwnerPath.shared("mesh.field_outputs"))


class _DeclaredModel:
    def __init__(self, name="field-model"):
        self.name = name
        self.module = Module(name)
        field = self.module.field_space("phi", ("phi",))
        aux = self.module.aux_field("rho_background")
        self.owner_path = self.module.owner_path
        self.field = self.module.field_handle(field)
        self.aux = self.module.aux_handle(aux)

    def declaration_index(self):
        return self.module.declaration_index()


def _registered_blocks(*names):
    problem = Problem(name="field-output-case")
    module = _DeclaredModel("charge-model")
    return problem, module, tuple(problem.add_block(name, module) for name in names)


# --- pops.fields.outputs: the typed field-output descriptors ----------------------------------
def test_field_outputs_construct_and_inspect():
    source = _shared_field("phi")
    phi = FieldOutput("phi")
    E = GradientOutput("E", source)
    J = DerivedField("J", "ohm", source=source)
    assert phi.options() == {"name": "phi", "recipe": "field", "source": None}
    assert E.options()["recipe"] == "grad_phi"
    assert E.options()["source"] == source.qualified_id
    assert E.capabilities().to_dict()["vector"] is True
    assert J.options()["recipe"] == "ohm"
    for out in (phi, E, J):
        assert out.category == "field_output"
        assert isinstance(out.inspect(), dict)
    assert phi.requirements().to_dict() == {}
    assert E.requirements().to_dict()["field"] == source.qualified_id
    assert J.requirements().to_dict()["field"] == source.qualified_id


def test_field_output_sources_reject_strings():
    with pytest.raises(TypeError, match="declaration Handle"):
        FieldOutput("phi", source="phi")
    with pytest.raises(TypeError, match="declaration Handle"):
        GradientOutput("E", "phi")
    with pytest.raises(TypeError, match="declaration Handle"):
        DerivedField("J", "ohm", source="phi")
    with pytest.raises(TypeError, match="declaration Handle"):
        OutputPolicy(fields=["phi"])


def test_fields_package_exports_outputs():
    assert "outputs" in pops.fields.__all__
    assert pops.fields.outputs.GradientOutput is GradientOutput


# --- composed / multi-block RHS ---------------------------------------------------------------
def test_composed_multiblock_rhs():
    problem, _, (ions, electrons) = _registered_blocks("ions", "electrons")
    background = _shared_field("rho_background")
    block_qids = [problem.resolve(block).qualified_id for block in (ions, electrons)]
    composed = ChargeDensity.from_blocks(ions, electrons) + FixedSource(background)
    resolved = composed.resolve_references(problem.resolve)
    assert isinstance(composed, SumRHS)
    assert resolved.options()["n_terms"] == 2
    assert resolved.options()["terms"] == ["charge_density", "fixed_source"]
    req = resolved.requirements().to_dict()
    assert req["blocks"] == block_qids
    assert req["aux_fields"] == [background.qualified_id]


def test_sumrhs_flattens_nested_sums():
    _, _, (ions, beam) = _registered_blocks("ions", "beam")
    total = (ChargeDensity.from_blocks(ions) + FixedSource(_shared_field("bg"))
             + ChargeDensity.from_blocks(beam))
    assert isinstance(total, SumRHS)
    assert len(total.terms) == 3  # flattened, not nested


def test_sumrhs_rejects_non_rhs_term():
    _, _, (ions,) = _registered_blocks("ions")
    with pytest.raises(TypeError):
        SumRHS(ChargeDensity.from_blocks(ions), object())
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
    ok = _problem(
        GeometricMG(),
        outputs=[FieldOutput("phi"), GradientOutput("E", _shared_field("phi"))],
    )
    assert ok.validate(context=ctx) is True


def test_duplicate_field_and_runtime_outputs_are_rejected():
    duplicate_fields = _problem(
        GeometricMG(), outputs=[FieldOutput("phi"), FieldOutput("phi")])
    with pytest.raises(ValueError, match="duplicate field output"):
        duplicate_fields.validate()

    policy = OutputPolicy(fields=[_shared_field("phi")])
    problem = Problem(name="duplicate-output-case")
    problem.output(policy)
    with pytest.raises(ValueError, match="already registered"):
        problem.output(policy)


def test_problem_rejects_foreign_and_ambiguous_field_output_sources():
    model = _DeclaredModel("transport")
    ambiguous = Problem(name="ambiguous-field-source")
    block_a = ambiguous.add_block("a", model)
    block_b = ambiguous.add_block("b", model)
    ambiguous.add_field(_problem(
        GeometricMG(), outputs=[GradientOutput("E", model.field)]))

    report = ambiguous.validate_report()
    issue = next(item for item in report.issues if item.code == "field.field_invalid")
    assert "matches 2 block instances" in issue.message
    assert str(block_a.instance_owner_path) in issue.message
    assert str(block_b.instance_owner_path) in issue.message

    foreign_model = _DeclaredModel("foreign")
    foreign = Problem(name="foreign-field-source")
    foreign.add_block("local", model)
    foreign.add_field(_problem(
        GeometricMG(), outputs=[GradientOutput("E", foreign_model.field)]))
    foreign_issue = next(
        item for item in foreign.validate_report().issues if item.code == "field.field_invalid")
    assert "no block in this case instantiates" in foreign_issue.message

    resolved = Problem(name="resolved-field-source")
    block = resolved.add_block("a", model)
    resolved.add_field(_problem(
        GeometricMG(), outputs=[GradientOutput("E", block[model.field])]))
    assert resolved.validate_report().ok


def test_field_problem_resolution_is_detached_qualified_and_visible_in_reports():
    import json

    from pops.fields.coefficients import ScalarCoefficient

    problem, model, (block,) = _registered_blocks("fluid")
    phi = unknown("phi")
    descriptor = PoissonProblem(
        name="poisson",
        unknown=phi,
        equation=(-laplacian(phi) == FixedSource(model.aux)),
        inputs=[model.field],
        coefficients=[ScalarCoefficient(model.aux)],
        outputs=[GradientOutput("E", model.field)],
        solver=GeometricMG(),
    )
    problem.add_field(descriptor)

    resolved = descriptor.resolve_references(problem.resolve)
    references = resolved.declaration_references()
    assert resolved is not descriptor
    assert resolved.equation is not descriptor.equation
    assert resolved.equation.rhs is not descriptor.equation.rhs
    assert resolved.coefficients[0] is not descriptor.coefficients[0]
    assert resolved.outputs[0] is not descriptor.outputs[0]
    assert references
    assert all(reference.is_resolved for reference in references)
    assert all(reference.block_ref.local_id == block.local_id for reference in references)
    assert descriptor.inputs[0] is model.field
    assert descriptor.inputs[0].owner_path.is_authoring
    assert descriptor.equation.rhs.aux_field is model.aux

    report = problem.inspect().to_dict()
    reported = report["fields"]["poisson"]
    qids = reported["options"]["references"]
    assert qids == reported["requirements"]["declaration_references"]
    assert qids
    assert all("block:fluid" in qid for qid in qids)
    assert all("#authoring=" not in qid for qid in qids)

    snapshot_json = json.dumps(problem.freeze().to_dict(), sort_keys=True)
    assert all(qid in snapshot_json for qid in qids)
    assert "#authoring=" not in snapshot_json


def test_field_problem_resolution_refuses_ambiguous_and_canonical_foreign_references():
    model = _DeclaredModel("transport")
    ambiguous = Problem(name="ambiguous-field-problem")
    ambiguous.add_block("a", model)
    ambiguous.add_block("b", model)
    descriptor = _problem(
        GeometricMG(), outputs=[GradientOutput("E", model.field)])
    with pytest.raises(AmbiguousReferenceError, match="matches 2 block instances"):
        descriptor.resolve_references(ambiguous.resolve)

    local = Problem(name="foreign-field-problem")
    local.add_block("local", model)
    foreign = Handle("phi", kind="field", owner=OwnerPath.model("foreign"))
    forged = _problem(
        GeometricMG(), outputs=[GradientOutput("E", foreign)])
    with pytest.raises(MissingOwnershipError, match="no block in this case instantiates"):
        forged.resolve_references(local.resolve)


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
