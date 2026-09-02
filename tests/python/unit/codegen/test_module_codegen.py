"""v5 candidate-table module records emitted into the single Program artifact."""
import json

from tests.python.support.requirements import require_native_or_skip
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_metadata import program_module_records
from types import SimpleNamespace

try:
    from pops._ir.expr import Const
    from pops.numerics.terms import DefaultSource, Flux
    from pops.physics._facade import Model
    from pops import time as adctime
    import pops.lib.time as libtime  # ready schemes live in pops.lib.time (Spec 4)
    from typed_program_support import fresh_field_refs, typed_state
except Exception as exc:  # pops not importable here -> skip, never fake
    require_native_or_skip('test_module_codegen (pops unavailable: %s)' % exc)


def _op(m, name):
    """A typed OperatorHandle for a registered operator (the de-stringed macro selector, ADC-532)."""
    return m.module.operator_handle(name)


def _lowered_emit_model(model):
    """Use the compiler-authenticated facade, never an unbound authoring model."""
    emit_model, source_module = lower_and_validate(model, facade=model)
    assert emit_model is model
    assert source_module is model.module
    assert type(emit_model._auxiliary_provider_pack).__name__ == "ProviderPack"
    assert type(emit_model._component_flux_provider_pack).__name__ == "ProviderPack"
    return emit_model


def _model():
    m = Model("ep")
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    bz = m.aux("B_z")
    m.flux(x=[mx, mx * mx / rho, mx * my / rho], y=[my, mx * my / rho, my * my / rho])
    m.source_term("electric", [Const(0.0), -rho * gx, -rho * gy])
    m.linear_source("lorentz", [[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]])
    m.elliptic_field("fields", rho - 1.0, aux=["grad_x", "grad_y", "B_z"])
    m.rate_operator("explicit_rhs", flux=True, sources=["electric"])
    return m


def _program_inputs(model):
    return fresh_field_refs(
        model,
        block_name="plasma",
        field_name="fields",
        provider=_op(model, "fields"),
    )


def _emit(program, model):
    solve = next(value for value in program._values if value.op == "solve_fields")
    field = solve.attrs["field"]
    plan = SimpleNamespace(
        name=field.local_id,
        native_options={
            "provider_slot": field.local_id,
            "output_route": {"components": list(solve.field_context.outputs)},
            "boundary_kernel_required": False,
        },
    )
    return emit_cpp_program(
        program, model=_lowered_emit_model(model), field_plans={field.local_id: plan})


def _state_step_lambda_body(source):
    """Extract the sole candidate ``state->step`` lambda, excluding descriptor setup."""

    marker = "state->step = [ctx_owner = state->ctx_owner](double dt) {"
    assert source.count(marker) == 1
    start = source.index(marker) + len(marker) - 1
    depth = 0
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start + 1:index]
    raise AssertionError("unterminated candidate state->step lambda")


def test_module_records_are_emitted_in_the_candidate_table():
    m = _model()
    state, fields = _program_inputs(m)
    P = libtime.PredictorCorrector(
        state, fields=fields,
        explicit=_op(m, "explicit_rhs"), implicit=_op(m, "lorentz"))
    src = _emit(P, m)
    # Module records and the install entry coexist in the one artifact.
    assert "pops_install_program" in src
    assert "kProgramCandidateModuleOperators" in src
    assert "kProgramCandidateModuleStateSpaces" in src
    assert "kProgramCandidateModuleFieldSpaces" in src
    # The candidate records include the registry names and kinds.
    reg = m.operator_registry()
    assert len(reg) > 0
    for op in ("flux_default", "electric", "lorentz", "fields", "explicit_rhs"):
        assert '"%s"' % op in src, "operator %r missing from the module metadata" % op
    for kind in ("grid_operator", "local_source", "local_linear_operator", "field_operator",
                 "local_rate"):
        assert '"%s"' % kind in src, "operator kind %r missing from the module metadata" % kind
    assert '"U"' in src and '"fields"' in src
    print("OK  v5 candidate module records emitted alongside the program")


def test_module_records_are_not_referenced_by_the_step_body():
    m = _model()
    state, fields = _program_inputs(m)
    P = libtime.PredictorCorrector(
        state, fields=fields,
        explicit=_op(m, "explicit_rhs"), implicit=_op(m, "lorentz"))
    src = _emit(P, m)
    body = _state_step_lambda_body(src)
    assert "kProgramCandidateModule" not in body, \
        "candidate module records must not be referenced in the hot step body"
    print("OK  candidate module records are install-time only")


def test_no_model_emits_empty_candidate_module_tables():
    P = adctime.Program("fe")
    u = typed_state(P, "plasma")
    target = typed_state(P, "plasma", state_name="U")
    P.commit(
        target.next,
        P.value(
            "u1", u + P.dt * P.rhs(
                state=u, terms=[Flux(), DefaultSource()]),
            at=target.next.point,
        ),
    )
    src = emit_cpp_program(P, model=None)
    for table in (
        "kProgramCandidateModuleOperators",
        "kProgramCandidateModuleStateSpaces",
        "kProgramCandidateModuleFieldSpaces",
    ):
        assert "ProgramAbiTable %s{};" % table in src
    print("OK  model=None emits empty v5 candidate module tables")


def test_module_records_include_every_declared_space_for_one_owner():
    from pops.codegen.program_models import ProgramModelGraph
    from pops.model import Module

    module = Module("multi_space_metadata")
    module.state_space("fluid", ("rho", "momentum"))
    module.state_space("tracer", ("c",))
    module.field_space("electrostatic", ("phi",))
    module.field_space("magnetic", ("bx", "by"))

    class RepresentativeEmitModel:
        """Kernel model exposes one representative space; source Module owns the full inventory."""

        def state_space(self):
            return module.state_spaces()["fluid"]

        def field_space(self):
            return module.field_spaces()["electrostatic"]

    owner = module.owner_path.canonical()
    graph = ProgramModelGraph(
        models_by_owner={owner: RepresentativeEmitModel()},
        source_modules_by_owner={owner: module},
        owners_by_block={"fluid": owner},
        authorities_by_owner={owner: module.owner_path},
    )
    operators, states, fields = program_module_records(adctime.Program("metadata_only"), graph)

    assert len(operators) == 0
    assert {row[0] for row in states} == {"fluid", "tracer"}
    assert {row[0] for row in fields} == {"electrostatic", "magnetic"}
    for _, _, signature, _, _ in (*states, *fields):
        assert signature
        assert json.dumps(json.loads(signature), sort_keys=True, separators=(",", ":")) == signature


def main():
    test_module_records_are_emitted_in_the_candidate_table()
    test_module_records_are_not_referenced_by_the_step_body()
    test_no_model_emits_empty_candidate_module_tables()
    test_module_records_include_every_declared_space_for_one_owner()
    print("OK  test_module_codegen")


if __name__ == "__main__":
    main()
