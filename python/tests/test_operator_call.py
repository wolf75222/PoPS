"""Spec 2 (S2-2): typed P.call and m.rate_operator.

P.call resolves an operator handle against the model's typed registry, type-checks the
arguments against its Signature, records a first-class ``call`` IR node, and
codegen lowers that node through GeneratedModule::Operators. m.rate_operator names
a composite -div F + sources rate as a module operator. Pure Python
(emit_cpp_program returns the .so source text without compiling); skips cleanly if pops
is not importable.
"""
import sys

try:
    from pops.ir.expr import Const, Var
    from pops import physics
    from pops.math import laplacian
    from pops.model import OperatorHandle
    from pops.codegen.program_emit_module_ops import operator_function_name
    from pops import time as adctime
except Exception as exc:  # pops not importable here -> skip, never fake
    print("skip test_call (pops unavailable: %s)" % exc)
    sys.exit(0)


def build_model():
    m = physics.Model("euler_poisson_lorentz")
    U = m.state("U", components=["rho", "mx", "my"])
    rho, mx, my = U
    phi = m.field("phi")
    m.aux("phi")
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    bz = m.aux("B_z")
    m.flux("flux", on=U,
           x=[mx, mx * mx / rho, mx * my / rho],
           y=[my, mx * my / rho, my * my / rho],
           waves={"x": [1.0, 1.0, 1.0], "y": [1.0, 1.0, 1.0]})
    electric = m.source("electric", on=U, value=[Const(0.0), rho * (-gx), rho * (-gy)])
    lorentz = m.local_linear_operator("lorentz", on=U,
                                      matrix=[[0.0, 0.0, 0.0],
                                              [0.0, 0.0, bz],
                                              [0.0, -bz, 0.0]])
    m.operator("lorentz", inputs=["fields"], returns=lorentz)
    m.solve_field("fields_from_state", equation=(-laplacian(phi) == rho))
    module = m.lower()
    module.rate_operator("explicit_rhs", state_space="U", flux=True, sources=[electric.operator])
    return module


def _program(build, m, name="prog"):
    P = adctime.Program(name)
    build(P, m)
    return P


def _emit(build, m, name="prog"):
    return _program(build, m, name).emit_cpp_program(model=m)


def _state(P, m, block="plasma"):
    return P.state("U", block=block, space=m.state_spaces()["U"]).n


def _handle(m, name):
    op = m.operator_registry().get(name)
    return OperatorHandle(op.name, kind=op.kind)


def _generated_call(m, name):
    reg = m.operator_registry()
    return "GeneratedModule::Operators::%s" % operator_function_name(reg.id_of(name), name)


def test_call_predictor_records_operator_nodes_and_lowers():
    """A predictor step written with P.call keeps typed call nodes in IR and lowers them."""
    m = build_model()

    def opfirst(P, _m):
        P.bind_operators(_m)
        U = _state(P, _m)
        f = P.call(_handle(_m, "fields_from_state"), U)
        R = P.call(_handle(_m, "explicit_rhs"), U, f)
        P.commit("plasma", P.linear_combine("u1", U + P.dt * R))

    P = _program(opfirst, m)
    assert [v.op for v in P._values].count("call") == 2
    for v in P._values:
        if v.op == "call":
            assert set(v.attrs) == {"operator", "operator_id", "output_type"}
    src = P.emit_cpp_program(model=m)
    assert "namespace GeneratedModule" in src
    assert "GeneratedModule::Operators::op_" in src
    assert _generated_call(m, "fields_from_state") + "(ctx, 0," in src
    assert _generated_call(m, "explicit_rhs") + "(ctx, 0," in src
    assert "electric" in src
    print("OK  P.call(fields_from_state)+P.call(explicit_rhs) records call and lowers")


def test_call_lowers_source_and_flux():
    m = build_model()

    def opfirst(P, _m):
        P.bind_operators(_m)
        U = _state(P, _m)
        f = P.call(_handle(_m, "fields_from_state"), U)
        s = P.call(_handle(_m, "electric"), U, f)
        flux = P.call(_handle(_m, "flux"), U)
        P.commit("plasma", P.linear_combine("u1", U + P.dt * s + P.dt * flux))

    P = _program(opfirst, m)
    assert [v.op for v in P._values].count("call") == 3
    src = P.emit_cpp_program(model=m)
    assert _generated_call(m, "flux") + "(ctx, 0," in src
    assert _generated_call(m, "electric") + "(ctx, 0," in src
    assert "electric" in src
    print("OK  P.call(electric)/P.call(flux_default) lowers to source / flux-only C++")


def test_call_default_source():
    """P.call(source_default, ...) reaches the default source and lowers through a generated
    module operator, not a Program-side RHS selector."""
    m = physics.Model("ds")
    U = m.state("U", components=["rho", "mx", "my"])
    rho, mx, my = U
    phi = m.field("phi")
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    m.flux("flux", on=U, x=[mx, mx * mx / rho, mx * my / rho],
           y=[my, mx * my / rho, my * my / rho],
           waves={"x": [1.0, 1.0, 1.0], "y": [1.0, 1.0, 1.0]})
    m.source("source_default", on=U, value=[Const(0.0), -rho * gx, -rho * gy])
    m.solve_field("fields_from_state", equation=(-laplacian(phi) == rho - 1.0))
    m = m.lower()

    def opfirst(P, _m):
        P.bind_operators(_m)
        U = _state(P, _m)
        f = P.call(_handle(_m, "fields_from_state"), U)
        s = P.call(_handle(_m, "source_default"), U, f)
        P.commit("plasma", P.linear_combine("u1", U + P.dt * s))

    P = _program(opfirst, m)
    assert [v.op for v in P._values].count("call") == 2
    src = P.emit_cpp_program(model=m)
    assert _generated_call(m, "source_default") + "(ctx, 0," in src
    assert "ctx.source_default_into(b, state, out);" in src
    print("OK  P.call(source_default) lowers to default-source-only C++")


def test_call_linear_operator_matches_solve_local_linear():
    m = build_model()

    def opfirst(P, _m):
        P.bind_operators(_m)
        U = _state(P, _m)
        f = P.call(_handle(_m, "fields_from_state"), U)
        L = P.call(_handle(_m, "lorentz"), f)
        U1 = P.solve_local_linear("u1", operator=P.I - P.dt * L, rhs=U, fields=f)
        P.commit("plasma", U1)

    P = _program(opfirst, m)
    assert any(v.op == "call" and v.vtype == "operator"
               and "kind" not in v.attrs for v in P._values)
    src = P.emit_cpp_program(model=m)
    assert "/* local_linear_operator" not in src
    assert _generated_call(m, "lorentz") + "(ctx, 0," in src
    print("OK  P.call(lorentz) operator drives solve_local_linear")


def test_call_projection_records_call_node_and_lowers_through_module_operator():
    m = build_model()
    state = next(iter(m.state_spaces().values()))
    m.operator(
        "projection", signature=(state,) >> state, kind="projection",
        expr=[Var("rho", "cons"), Var("mx", "cons"), Var("my", "cons")])

    def opfirst(P, _m):
        P.bind_operators(_m)
        U = _state(P, _m)
        projected = P.call(_handle(_m, "projection"), U)
        P.commit("plasma", projected)

    P = _program(opfirst, m)
    call_nodes = [v for v in P._values if v.op == "call"]
    assert len(call_nodes) == 1
    assert call_nodes[0].vtype == "state"
    assert call_nodes[0].attrs["operator"] == "projection"
    assert "kind" not in call_nodes[0].attrs
    src = P.emit_cpp_program(model=m)
    assert _generated_call(m, "projection") + "(ctx, 0," in src
    assert "v.op == \"call\"" not in src
    assert "/* local_linear_operator" not in src
    print("OK  P.call(projection) records one state call and lowers through GeneratedModule")


def test_call_typing_errors():
    m = build_model()
    P = adctime.Program("p").bind_operators(m)
    U = _state(P, m)
    f = P.call(_handle(m, "fields_from_state"), U)

    # No bind -> clear error.
    P2 = adctime.Program("p2")
    try:
        P2.call(_handle(m, "electric"), P2.state("U", block="plasma").n)
        raise AssertionError("expected an error calling without bound operators")
    except ValueError as exc:
        assert "no operators bound" in str(exc)

    # Unknown operator -> clear KeyError.
    try:
        P.call(OperatorHandle("does_not_exist", kind="local_source"), U)
        raise AssertionError("expected KeyError for an unknown operator")
    except KeyError as exc:
        assert "unknown operator" in str(exc)

    # Arity mismatch -> electric needs (state, fields).
    try:
        P._call("electric", U)
        raise AssertionError("expected arity error")
    except ValueError as exc:
        assert "expects 2 argument" in str(exc)

    # vtype mismatch -> a fields value where a state is expected.
    try:
        P._call("electric", f, f)
        raise AssertionError("expected a vtype error")
    except ValueError as exc:
        assert "expects a state value" in str(exc)
    print("OK  P.call typing: no-bind / unknown / arity / vtype errors are clear")


def test_default_resolution_and_ambiguity():
    m = build_model()
    reg = m.operator_registry()
    # The privileged defaults resolve uniquely.
    assert reg.default_of_kind("field_operator").name == "fields_from_state"
    assert reg.default_of_kind("grid_operator").name == "flux"
    # Add a SECOND, non-privileged field operator; the privileged default still wins.
    fields = next(iter(m.field_spaces().values()))
    state = next(iter(m.state_spaces().values()))
    m.operator("psi", signature=(state,) >> fields, kind="field_operator", expr=Var("rho", "cons"))
    reg2 = m.operator_registry()
    assert len(reg2.operators_of_kind("field_operator")) == 2
    assert reg2.default_of_kind("field_operator").name == "fields_from_state"
    print("OK  default_of_kind resolves privileged defaults; second field op is explicit")


def test_rate_operator_alias_not_in_hash():
    m = build_model()
    electric = m.operator_registry().get("electric")
    before = set(m.list_operators())
    m.rate_operator("explicit_rhs_2", state_space="U", flux=True, sources=[electric])
    assert set(m.list_operators()) == before | {"explicit_rhs_2"}
    assert "explicit_rhs" in m.operator_registry()
    print("OK  Module.rate_operator registers an explicit operator-first alias")


def main():
    test_call_predictor_records_operator_nodes_and_lowers()
    test_call_lowers_source_and_flux()
    test_call_default_source()
    test_call_linear_operator_matches_solve_local_linear()
    test_call_projection_records_call_node_and_lowers_through_module_operator()
    test_call_typing_errors()
    test_default_resolution_and_ambiguity()
    test_rate_operator_alias_not_in_hash()
    print("OK  test_call")


if __name__ == "__main__":
    main()
