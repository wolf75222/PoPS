"""Actual authoring/source guards on frozen @3; no JIT or native qualification."""

from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def candidate_case(monkeypatch, *, width=3, order=(2, 0, 1), uniform=True, selected=True):
    # Reuse only the established public Case/layout setup, not its math oracles.
    import test_sol61_amr_public_original as api
    from pops.math import ValueExpr

    states = []
    old_state = api.pops.Model.state
    old_diffusion = api.DivCoeffGrad
    old_method = api.CellCenteredNonlinearCoupled

    def state(model, *args, **kwargs):
        result = old_state(model, *args, **kwargs)
        states.append(result)
        return result

    def matrix(m):
        result = [
            [Fraction((-1) ** (i + j) * (1 + 2 * i + j), 13) for j in range(m)] for i in range(m)
        ]
        if m > 1:
            result[-1] = [2 * x for x in result[0]]  # Signed, nonsymmetric and singular.
        return tuple(tuple(row) for row in result)

    def diffusion(field, coefficient):
        return old_diffusion(field, coefficient * (1 + states[-1][0] + ValueExpr(field) ** 2))

    def method(**kwargs):
        return old_method(
            **kwargs,
            face_policy="Arithmetic@1",
            coefficient_evaluation="PerCandidate@1" if selected else None,
        )

    monkeypatch.setattr(api.pops.Model, "state", state)
    monkeypatch.setattr(api, "matrix", matrix)
    monkeypatch.setattr(api, "DivCoeffGrad", diffusion)
    monkeypatch.setattr(api, "CellCenteredNonlinearCoupled", method)
    return api.authored(width, order, uniform=uniform)


@pytest.mark.parametrize(
    "uniform,width,order",
    [(True, 1, (0,)), (True, 3, (2, 0, 1)), (False, 3, (1, 2, 0)), (False, 5, (4, 2, 0, 3, 1))],
)
def test_actual_candidate_descriptor_and_conditional_contract(monkeypatch, uniform, width, order):
    from pops.fields._program_nonlinear_problem import validate_nonlinear_field_request

    case, layout, program, token, _ = candidate_case(
        monkeypatch, width=width, order=order, uniform=uniform
    )
    validate_nonlinear_field_request(program, token)
    program.validate()
    assert token.attrs["contract"] == "pops.spatial-field-residual@3"
    assert token.attrs["coefficient_evaluation"] == "pops.field.coefficients.per-candidate@1"
    criterion = "pops.field.linear.true-correction-residual@1"
    assert token.attrs["linear_residual_verification"] == criterion
    assert token.attrs["source_contract"]["linear_residual_verification"] == criterion
    assert token.attrs["solve_request"]["realization"]["linear_residual_verification"] == criterion
    assert token.attrs["solve_request"]["schema_version"] == 2
    assert program._serialize()["version"] == 11
    descriptor = token.inputs[1]
    assert descriptor.attrs["storage_role"] == "deferred_candidate_coefficient"
    captures = token.inputs[2 : 2 + token.attrs["capture_count"]]
    assert len(descriptor.inputs) == len(captures)
    assert all(a is b for a, b in zip(descriptor.inputs, captures, strict=True))
    assert any("unknown" in repr(expression) for expression in descriptor.attrs["expressions"])


def test_unknown_dependent_D_still_refuses_without_explicit_realization(monkeypatch):
    from pops.fields import FieldProblemError

    with pytest.raises(FieldProblemError, match="unknown-dependent D"):
        candidate_case(monkeypatch, selected=False)


@pytest.mark.parametrize(
    "attack",
    [
        "point",
        "captures",
        "body",
        "face",
        "policy",
        "contract",
        "preconditioner",
        "descriptor_role",
        "criterion_missing",
        "criterion_source",
    ],
)
def test_candidate_mutations_refuse_and_original_authoring_remains_intact(monkeypatch, attack):
    from pops.fields._program_nonlinear_problem import validate_nonlinear_field_request
    from pops.time import SolveRequestError
    from pops.time._program.serialization import _json_ready

    _, _, program, token, _ = candidate_case(monkeypatch)
    before = program._ir_hash()
    old_attrs, old_inputs = token.attrs, token.inputs
    descriptor = token.inputs[1]
    old_descriptor_attrs = descriptor.attrs
    attrs = deepcopy(_json_ready(token.attrs))
    descriptor_attrs = deepcopy(_json_ready(descriptor.attrs))
    if attack == "point":
        old_point = descriptor.point
        object.__setattr__(descriptor, "point", program.stage("foreign-candidate-D", c=1))
    elif attack == "captures":
        inputs = list(old_inputs)
        inputs[2 : 2 + attrs["capture_count"]] = reversed(inputs[2 : 2 + attrs["capture_count"]])
        object.__setattr__(token, "inputs", tuple(inputs))
    elif attack == "body":
        descriptor_attrs["expressions"] = [["literal", "0x0.0p+0"]] * 9
    elif attack == "face":
        descriptor_attrs["coefficient_face_policy"] = "pops.field.face-mean.harmonic@1"
    elif attack == "policy":
        attrs["coefficient_evaluation"] = "pops.field.coefficients.seed-frozen@1"
    elif attack == "contract":
        attrs["contract"] = "pops.spatial-field-residual@2"
    elif attack == "preconditioner":
        attrs["right_preconditioner"] = "pops.amr.original-spatial-jacobi.basis-response@1"
    elif attack == "criterion_missing":
        attrs.pop("linear_residual_verification")
    elif attack == "criterion_source":
        attrs["source_contract"]["linear_residual_verification"] = "projected-only@1"
    else:
        descriptor_attrs["storage_role"] = "material_field"
    object.__setattr__(token, "attrs", attrs)
    object.__setattr__(descriptor, "attrs", descriptor_attrs)
    try:
        with pytest.raises((ValueError, SolveRequestError)):
            validate_nonlinear_field_request(program, token)
    finally:
        object.__setattr__(token, "attrs", old_attrs)
        object.__setattr__(token, "inputs", old_inputs)
        object.__setattr__(descriptor, "attrs", old_descriptor_attrs)
        if attack == "point":
            object.__setattr__(descriptor, "point", old_point)
    assert program._ir_hash() == before
    program.validate()
    validate_nonlinear_field_request(program, token)


def test_actual_amr_order_and_two_full_F_derivative_samples():
    text = (ROOT / "include/pops/runtime/program/prepared_amr_field_residual.hpp").read_text()
    start = text.index("auto evaluate =")
    finish = text.index("auto residual =", start)
    body = text[start:finish]
    steps = [
        "copy_(q, evaluation_q_)",
        "synchronize_original_field_candidate(evaluation_q_)",
        "coefficient_body(",
        "prepare_original_field_operator()",
        "apply_original_field_operator(evaluation_q_, result)",
        "add_local(*physical_q",
    ]
    positions = [body.index(step) for step in steps]
    assert positions == sorted(positions)
    derivative = text[
        text.index("auto derivative =") : text.index("std::vector<field_type*> destinations")
    ]
    assert (
        "evaluate(perturbed_, plus_" in derivative and "evaluate(perturbed_, minus_" in derivative
    )
    assert "candidate_evaluation_->apply_original_field_operator" not in derivative
    provider = (
        ROOT / "include/pops/numerics/elliptic/nd/prepared_composite_general_field.hpp"
    ).read_text()
    assert "prepare_linear_coefficients(lane_, true)" in provider
    fac = (ROOT / "include/pops/numerics/elliptic/amr/composite_fac_poisson.hpp").read_text()
    preparation = fac[
        fac.index("void prepare_linear_coefficients") : fac.index(
            "void synchronize_linear_solution"
        )
    ]
    assert preparation.index("restrict_into(") < preparation.index("same_level_fill_")
    # The unchanged prepared owner lane remains authoritative for refusal votes.
    authority = text[text.index("void require_authority") : text.index("/// add_local")]
    assert (
        "local_phase_(authority_lane" in authority
        and "&lane != &op_->original_field_execution_lane()" in authority
    )


@pytest.mark.parametrize("uniform", [True, False])
def test_actual_resolved_lowering_evaluates_D_in_full_residual(monkeypatch, uniform):
    import test_sol61_amr_public_original as api

    case, layout, _, token, _ = candidate_case(monkeypatch, uniform=uniform)
    code, resolved = api.emit(case, layout)
    assert resolved.time._serialize()["version"] == 11
    assert "nonfinite_candidate_diffusion" in code
    assert "candidate(index," in code
    if uniform:
        assert "candidate diffusion local evaluation" in code
        assert (
            "prepare_general_field_coefficients<pops::kNativeDimension, 3, 9, false, true>" in code
        )
        assert "apply_general_field<pops::kNativeDimension, 3, 9, true, true>" in code
        assert "candidate original residual norm" in code
        assert ", true, &ctx.prepared_execution_lane(), true);" in code
    else:
        assert "AmrFieldCoefficientEvaluation::kPerCandidate" in code
        assert "_core->solve_candidate(" in code
        # The descriptor may allocate inert assembly storage on the retained provider;
        # actual coefficients are written by the candidate body into its owned evaluation.
        assert "const auto& captured, auto& coefficients, int evaluation)" in code
    assert token.attrs["contract"] == "pops.spatial-field-residual@3"


def test_resealed_literal_descriptor_cannot_change_registered_original_D(monkeypatch):
    import test_sol61_amr_public_original as api
    from pops._frozen_data import freeze_containers
    from pops.fields._program_nonlinear_problem import (
        _request_data,
        validate_nonlinear_field_request,
    )
    from pops.time.solve_request import SolveUnknown
    from pops.time._program.serialization import _json_ready

    case, layout, program, token, _ = candidate_case(monkeypatch)
    literal = tuple(("literal", {"kind": "integer", "value": "0"}) for _ in range(9))
    descriptor = program._replace_value(
        token.inputs[1],
        attrs={**token.inputs[1].attrs, "expressions": literal, "field_dependencies": ()},
    )
    object.__setattr__(descriptor, "inputs", ())
    source = {**token.attrs["source_contract"], "diffusion": literal}
    token = program._replace_value(
        token, attrs={**token.attrs, "source_contract": freeze_containers(source)}
    )
    object.__setattr__(token, "inputs", (token.inputs[0], descriptor, *token.inputs[2:]))
    previous = token.attrs["solve_request"]
    unknown = SolveUnknown(previous["unknowns"][0]["name"], token.inputs[0])
    request = _request_data(program, token, unknown, _json_ready(previous["physical_problem"]))
    token = program._replace_value(token, attrs={**token.attrs, "solve_request": request})
    validate_nonlinear_field_request(program, token)
    with pytest.raises(ValueError, match="equation_identity_drift"):
        api.emit(case, layout)


def test_opt_in_allocation_failures_drain_before_collective_votes():
    sources = (
        (
            "include/pops/numerics/elliptic/nd/prepared_composite_general_field.hpp",
            "if (apply_only_)",
            "all_reduce_max(failure, lane)",
        ),
        (
            "include/pops/numerics/elliptic/amr/composite_fac_poisson.hpp",
            "if (guard_local_setup)",
            "all_reduce_max(local_error ? 1L : 0L, *lane_)",
        ),
    )
    for relative, guard, vote in sources:
        text = (ROOT / relative).read_text()
        end = text.index(vote)
        begin = text.rfind("catch (...) {", 0, text.rfind(guard, 0, end))
        drain = text[begin:end]
        assert guard in drain
        assert "Kokkos::fence()" in drain
        assert drain.index(guard) < drain.index("Kokkos::fence()")


def legacy_images(source):
    """Explicit external review at identical fixture path/callsite; never fetches."""
    from contextlib import ExitStack
    from functools import partial
    import hashlib
    import json
    import runpy
    import sys
    from unittest.mock import patch

    sys.path.insert(0, str(Path(source).resolve() / "python"))
    from pops.solvers import Newton
    from pops.time._program.serialization import _json_ready

    fixture = runpy.run_path(str(Path(__file__).with_name("test_sol61_amr_public_original.py")))

    def digest(data):
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    profiles = []
    for width, order, uniform, policy in (
        (2, (1, 0), False, "original"),
        (3, (2, 0, 1), True, "original"),
        (5, (4, 2, 0, 3, 1), False, "jacobi"),
        (3, (2, 0, 1), True, "captured"),
    ):
        globals_ = fixture["authored"].__globals__
        with ExitStack() as context:
            if policy == "jacobi":
                context.enter_context(
                    patch.dict(
                        globals_,
                        Newton=partial(Newton, right_preconditioner="SpatialBasisJacobi@1"),
                    )
                )
            elif policy == "captured":
                states = []
                old_state, old_diffusion, old_method = (
                    globals_["pops"].Model.state,
                    globals_["DivCoeffGrad"],
                    globals_["CellCenteredNonlinearCoupled"],
                )

                def state(model, *args, _state=old_state, _states=states, **kwargs):
                    result = _state(model, *args, **kwargs)
                    _states.append(result)
                    return result

                def diffusion(field, coefficient, _diffusion=old_diffusion, _states=states):
                    return _diffusion(field, coefficient * _states[-1][0])

                context.enter_context(patch.object(globals_["pops"].Model, "state", state))
                context.enter_context(
                    patch.dict(
                        globals_,
                        DivCoeffGrad=diffusion,
                        CellCenteredNonlinearCoupled=partial(
                            old_method, face_policy="Arithmetic@1"
                        ),
                    )
                )
            case, layout, program, token, _ = fixture["authored"](
                width, order, seed=True, uniform=uniform
            )
        cpp, resolved = fixture["emit"](case, layout)
        modules = [block.model.module for block in resolved.blocks]
        profiles.append(
            {
                "policy": policy,
                "width": width,
                "uniform": uniform,
                "authored_ir": program._ir_hash(),
                "resolved_ir": resolved.time._ir_hash(),
                "ir_version": resolved.time._serialize()["version"],
                "cpp": hashlib.sha256(cpp.encode()).hexdigest(),
                "module_hashes": [module.module_hash() for module in modules],
                "module_full_manifest_sha256": [
                    digest(module.manifest().to_dict()) for module in modules
                ],
                "full_request_sha256": digest(_json_ready(token.attrs["solve_request"])),
                "physical_equation": token.attrs["solve_request"]["equation_identity"],
                "solver_identity": token.attrs["solver_identity"],
                "request_version": token.attrs["solve_request"]["schema_version"],
            }
        )
    print(json.dumps(profiles, sort_keys=True))


if __name__ == "__main__":
    import sys

    legacy_images(sys.argv[1])
