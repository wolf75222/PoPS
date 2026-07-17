"""Final operator-first time factories compose exact typed handles into ordinary Programs."""
from tests.python.support.requirements import require_native_or_skip
from pops.codegen.program_codegen import emit_cpp_program
import inspect

import pytest

try:
    from pops.physics._facade import Model
    from pops import time as adctime
    import pops.lib.time as libtime
    from typed_program_support import fresh_field_refs, state_refs
except Exception as exc:  # pops not importable here -> skip, never fake
    require_native_or_skip('test_operator_macros (pops unavailable: %s)' % exc)

_PHYSICS_TOKENS = ("electric", "lorentz", "poisson", "rho", "grad_x", "grad_y", "B_z")


def _model(name, gain=1.0):
    m = Model(name)
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    m.aux("phi")
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    m.flux(x=[mx, mx * mx / rho, mx * my / rho],
           y=[my, mx * my / rho, my * my / rho])
    m.source_term("electric", [0.0 * rho, rho * (-gx) * gain, rho * (-gy) * gain])
    m.linear_source("implicit", [[-1.0, 0.0, 0.0],
                                 [0.0, -1.0, 0.0],
                                 [0.0, 0.0, -1.0]])
    m.elliptic_field("fields", rho - 1.0, aux=["phi", "grad_x", "grad_y"])
    m.rate_operator("explicit_rhs", flux=True, sources=["electric"])
    return m


def _handle(m, name):
    return m.module.operator_handle(name)


def _references(m, name="plasma"):
    return fresh_field_refs(
        m,
        block_name=name,
        field_name="fields",
        provider=_handle(m, "fields"),
    )


def test_factories_are_model_free():
    for factory in (libtime.PredictorCorrector, libtime.RungeKutta, libtime.IMEX):
        source = inspect.getsource(factory)
        for token in _PHYSICS_TOKENS:
            assert token not in source, "%s must not mention %r" % (factory.__name__, token)


def test_predictor_corrector_factory():
    model = _model("ep")
    state, fields = _references(model)
    program = libtime.PredictorCorrector(
        state,
        fields=fields,
        explicit=_handle(model, "explicit_rhs"),
        implicit=_handle(model, "implicit"),
    )
    assert program.validate() is True


def test_explicit_runge_kutta_factory():
    model = _model("rk")
    state, fields = _references(model)
    program = libtime.RungeKutta(
        state,
        rate=_handle(model, "explicit_rhs"),
        fields=fields,
        tableau=libtime.SSPRK2_TABLEAU,
    )
    assert program.validate() is True


def test_imex_factory():
    model = _model("imex")
    state, fields = _references(model)
    program = libtime.IMEX(
        state,
        explicit_operator=_handle(model, "explicit_rhs"),
        implicit_operator=_handle(model, "implicit"),
        fields_operator=fields,
    )
    assert program.validate() is True


def test_factory_rejects_string_operator():
    model = _model("reject")
    state, fields = _references(model)
    with pytest.raises(TypeError, match="OperatorHandle"):
        libtime.IMEX(
            state,
            explicit_operator="explicit_rhs",
            implicit_operator=_handle(model, "implicit"),
            fields_operator=fields,
        )


def test_factory_ir_retains_each_exact_handle():
    model = _model("identity")
    state, fields = _references(model)
    expected = {
        _handle(model, "explicit_rhs"),
        _handle(model, "implicit"),
    }
    program = libtime.IMEX(
        state,
        explicit_operator=_handle(model, "explicit_rhs"),
        implicit_operator=_handle(model, "implicit"),
        fields_operator=fields,
    )
    retained = {
        value.attrs["operator_handle"]
        for value in program._values
        if "operator_handle" in value.attrs
    }
    assert expected <= retained
    solve = next(value for value in program._values if value.op == "solve_fields")
    assert solve.attrs["field"] is fields


def test_factory_reused_across_modules():
    def build(model):
        block, state = state_refs(adctime.Program("refs"), "plasma", model=model)
        program = libtime.IMEX(
            block[state],
            explicit_operator=_handle(model, "explicit_rhs"),
            implicit_operator=_handle(model, "implicit"),
        )
        return emit_cpp_program(program, model=model)

    def no_field_model(name, gain):
        model = Model(name)
        (u,) = model.conservative_vars("u")
        model.source_term("explicit_source", [gain * u])
        model.linear_source("implicit", [[-1.0]])
        model.rate_operator("explicit_rhs", flux=False, sources=["explicit_source"])
        return model

    source_a = build(no_field_model("A", 1.0))
    source_b = build(no_field_model("B", 2.0))
    assert "pops_install_program" in source_a and source_a != source_b


def _field_factory_builders(model):
    state, fields = _references(model)
    explicit = _handle(model, "explicit_rhs")
    implicit = _handle(model, "implicit")
    return {
        "RungeKutta": lambda action: libtime.RungeKutta(
            state, rate=explicit, fields=fields, tableau=libtime.SSPRK2_TABLEAU,
            solve_action=action),
        "AdamsBashforth": lambda action: libtime.AdamsBashforth(
            state, rate=explicit, fields=fields, order=2, solve_action=action),
        "BDF": lambda action: libtime.BDF(
            state, implicit=implicit, explicit=explicit, fields=fields, order=1,
            solve_action=action),
        "PredictorCorrector": lambda action: libtime.PredictorCorrector(
            state, fields=fields, explicit=explicit, implicit=implicit,
            solve_action=action),
        "IMEX": lambda action: libtime.IMEX(
            state, explicit_operator=explicit, implicit_operator=implicit,
            fields_operator=fields, solve_action=action),
    }


def _recorded_solve_actions(program):
    return tuple(
        value.attrs["action"]
        for value in program._values
        if value.op == "solve_outcome"
    )


def test_field_factories_consume_outcomes_with_default_and_custom_actions():
    from pops.time import FailRun, RejectAttempt

    for name, build in _field_factory_builders(_model("actions-default")).items():
        actions = _recorded_solve_actions(build(None))
        assert actions and all(isinstance(action, FailRun) for action in actions), name

    custom = RejectAttempt(statuses=("iteration_limit", "breakdown"))
    for name, build in _field_factory_builders(_model("actions-custom")).items():
        actions = _recorded_solve_actions(build(custom))
        assert actions and all(action == custom for action in actions), name


def test_field_factories_reject_invalid_solve_actions():
    for _, build in _field_factory_builders(_model("actions-invalid")).items():
        with pytest.raises(TypeError, match="solve_action"):
            build("reject")


def main():
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
    print("OK  test_operator_macros")


if __name__ == "__main__":
    main()
