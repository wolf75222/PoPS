"""Spec 3 inspection / debug API (section 33).

Even when a model or program is written in the board style, the system must be able
to show the lowering: the operator-first Program IR and the C++ plan, and the typed
Module the board model lowers to. These dumps are pure introspection over the
existing IR -- they prove the board and operator-first writings share one kernel.
"""
from pops.params import ConstParam
import pytest

physics = pytest.importorskip("pops.physics")
amath = pytest.importorskip("pops.math")
from tests.python.unit.runtime._typed_program import typed_program_state  # noqa: E402


def _board_model():
    from pops.math import sqrt, grad, div, laplacian, ddt
    m = physics.Model("euler_poisson_lorentz")
    U = m.state("U", components=["rho", "mx", "my"],
                roles={"rho": "density", "mx": "momentum_x", "my": "momentum_y"})
    rho, mx, my = U
    u, v = m.primitive("u", mx / rho), m.primitive("v", my / rho)
    cs2 = m.value(m.param(ConstParam("cs2", 1.0)))
    p, c = m.scalar("p", cs2 * rho), m.scalar("c", sqrt(cs2))
    flux = m.flux("F", on=U, x=[mx, mx * u + p, mx * v], y=[my, my * u, my * v + p],
                  waves={"x": [u - c, u, u + c], "y": [v - c, v, v + c]})
    phi = m.field("phi")
    m.solve_field("fields_from_state", equation=(-laplacian(phi) == rho),
                  outputs={"phi": phi, "grad_x": grad(phi).x, "grad_y": grad(phi).y},
                  solver="geometric_mg")
    e_field = m.vector_field("E", x=-grad(phi).x, y=-grad(phi).y)
    a_src = m.source("electric", on=U, value=[0.0 * rho, rho * e_field.x, rho * e_field.y])
    bz = m.aux("B_z")
    c_b = m.local_linear_operator("C(B)", on=U,
                                  matrix=[[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]])
    m.rate("explicit_rate", ddt(U) == -div(flux) + a_src)
    m.operator("implicit_operator", returns=c_b, inputs=["fields"])
    return m


def test_program_dump_operator_ir_shows_the_lowering():
    P, _, _, _, _, temporal = typed_program_state("fe", block_name="plasma")
    dt = P.dt
    u = temporal.n
    f = P.solve_fields("f", u)
    r = P._rhs_legacy(name="R", state=u, fields=f, flux=True, sources=["electric"])
    u1 = P.linear_combine("U1", u + dt * r)
    P.commit(temporal.next, u1)
    txt = P.dump_operator_ir()
    assert "operator-first Program IR" in txt
    assert "solve_fields" in txt
    assert "linear_combine" in txt
    assert "T.commit(T.state(plasma, U).next" in txt


def test_program_dump_board_and_cpp_plan():
    P, _, _, _, _, temporal = typed_program_state("fe", block_name="plasma")
    u = temporal.n
    P.solve_fields("f", u)
    board = P.dump_board()
    plan = P.dump_cpp_plan()
    assert "board == operator-first" in board
    assert "operator-first Program IR" in board       # board view embeds the IR
    assert "ctx." in plan and "GeneratedProgram" in plan


def test_model_dump_module_ir_lists_spaces_and_operators():
    m = _board_model()
    txt = m.dump_module_ir()
    assert "StateSpace U" in txt and "rho" in txt
    assert "FieldSpace fields" in txt
    assert "explicit_rate" in txt and "local_rate" in txt
    assert "implicit_operator" in txt and "local_linear_operator" in txt


def test_explicit_program_binding_rejects_out_of_order_registry_replacement():
    # A Program is bound to one exact registry identity. Extending the facade may publish a new
    # registry for the same owner, but an existing Program must not silently swap authorities.
    from pops.math import sqrt, grad, div, laplacian, ddt
    m = physics.Model("ep")
    U = m.state("U", components=["rho", "mx", "my"])
    rho, mx, my = U
    u, v = m.primitive("u", mx / rho), m.primitive("v", my / rho)
    cs2 = m.value(m.param(ConstParam("cs2", 1.0)))
    p, c = m.scalar("p", cs2 * rho), m.scalar("c", sqrt(cs2))
    flux = m.flux("F", on=U, x=[mx, mx * u + p, mx * v], y=[my, my * u, my * v + p],
                  waves={"x": [u - c, u, u + c], "y": [v - c, v, v + c]})
    phi = m.field("phi")
    m.solve_field("fields_from_state", equation=(-laplacian(phi) == rho),
                  outputs={"phi": phi, "grad_x": grad(phi).x, "grad_y": grad(phi).y},
                  solver="geometric_mg")
    e_field = m.vector_field("E", x=-grad(phi).x, y=-grad(phi).y)
    a_src = m.source("electric", on=U, value=[0.0 * rho, rho * e_field.x, rho * e_field.y])
    explicit_rate = m.rate("explicit_rate", ddt(U) == -div(flux) + a_src)

    P, _, _, _, _, temporal = typed_program_state(
        "late", block_name="plasma", model=m, state="U")
    u_n = temporal.n
    f_n = P.solve_fields("f", u_n)
    explicit_rate(u_n, f_n)
    bz = m.aux("B_z")
    c_b = m.local_linear_operator("C(B)", on=U,
                                  matrix=[[0.0, 0.0, 0.0], [0.0, 0.0, bz], [0.0, -bz, 0.0]])
    implicit_operator = m.operator("implicit_operator", returns=c_b, inputs=["fields"])
    with pytest.raises(ValueError, match="already bound to a different registry"):
        P.bind_operators(m.module)

    fresh, _, _, _, _, fresh_temporal = typed_program_state(
        "late-fresh", block_name="plasma", model=m, state="U")
    fresh_fields = fresh.solve_fields("f", fresh_temporal.n)
    L = implicit_operator(fresh_fields)
    assert L.vtype == "operator"


def test_model_dump_physics_and_capabilities():
    m = _board_model()
    phys = m.dump_physics()
    caps = m.dump_capabilities()
    assert "physics.Model euler_poisson_lorentz" in phys
    assert "states:" in phys and "'U'" in phys
    assert "operators:" in phys and "explicit_rate" in phys
    assert "capabilities / requirements" in caps
    assert "explicit_rate" in caps
