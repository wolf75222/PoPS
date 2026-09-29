"""Independent public authoring witness for a non-hydrostatic three-state face.

The separate NumPy oracle lives in api040_coordinated_face_oracle_independent.
This source test checks the genuine public declaration, owner resolution and
generated C++ route; it does not claim native execution.
"""

import pops
import pytest
from pops._ir.expr import Var
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D, Cartesian2D
from pops.lib.time import ForwardEuler
from pops.math import ddt, div, maximum
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.layouts import Uniform
from pops.numerics import (CoordinatedFace, CoordinatedFiniteVolume,
                           DiscretizationPlan, FaceBalance)
from pops.time import FixedDt
from tests.python.support.api040_coordinated_face_oracle_independent import face as oracle_face


def _case(order=("c", "p", "r"), *, beta=.7, gamma=-.4, dimension=1):
    assert set(order) == {"c", "p", "r"} and len(order) == 3
    assert dimension in (1, 2)
    frame = CartesianDomain("three_state", (0.,)*dimension,
                            (1.,)*dimension).frame(
                                Cartesian1D() if dimension == 1 else Cartesian2D())
    axis = frame.axes[0]
    model = pops.Model("asymmetric_product", frame=frame)
    state = model.state("U", components=order)
    c, p, r = (state[name] for name in ("c", "p", "r"))
    physical_flux = tuple({"c": c, "p": p/2, "r": -r}[name] for name in order)
    flux = model.flux("transport", state=state, frame=frame,
                      components={entry: (physical_flux if entry == axis else (0.,)*3)
                                  for entry in frame.axes})
    matrix = tuple(tuple((beta*r if row == "p" and column == "c" else
                          gamma*c if row == "r" and column == "p" else 0.)
                         for column in order) for row in order)
    product = model.nonconservative_product("coupling", state=state,
        matrices={entry: (matrix if entry == axis else ((0.,)*3,)*3)
                  for entry in frame.axes}, conservative_components=("c",))
    rate = model.rate("complete", equation=ddt(state) == -div(flux) - product)
    model.primitive_state(*state, conservative=tuple(state))
    index = {name: order.index(name) for name in order}

    def body(left, right, _axis):
        if _axis != 0:
            return FaceBalance((0.,)*3, (0.,)*3, (0.,)*3, 0.)
        lc, lp, lr = (left[index[name]] for name in ("c", "p", "r"))
        rc, rp, rr = (right[index[name]] for name in ("c", "p", "r"))
        integral = {"c": 0., "p": beta*(lr+rr)*(rc-lc)/2,
                    "r": gamma*(lc+rc)*(rp-lp)/2}
        speed = 1 + maximum(abs(beta)*maximum(abs(lr), abs(rr)),
                            abs(gamma)*maximum(abs(lc), abs(rc)))
        fl = {"c": lc, "p": lp/2, "r": -lr}
        fr = {"c": rc, "p": rp/2, "r": -rr}
        shared = tuple((fl[name]+fr[name])/2 - speed*(right[index[name]]-left[index[name]])/2
                       for name in order)
        return FaceBalance(shared,
            tuple(0. if name == "c" else -.3*integral[name] for name in order),
            tuple(0. if name == "c" else -.7*integral[name] for name in order), speed)

    face = CoordinatedFace(flux=flux, product=product, frame=frame, body=body)
    plan = DiscretizationPlan()
    plan.rates.add(rate, CoordinatedFiniteVolume(face=face))
    case = pops.Case("asymmetric_product_case")
    block = case.block("transport", model)
    case.numerics(plan, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(.05/8))
    case.program(program)
    return case, Uniform(CartesianGrid(frame=frame, cells=(8,)*dimension,
                                      periodic=PeriodicAxes(frame.axes))), face


def test_public_three_state_permutation_resolves_and_emits_distinct_face_policy():
    emitted = []
    identities = []
    for order in (("c", "p", "r"), ("r", "c", "p")):
        case, layout, face = _case(order)
        resolved = pops.resolve(pops.validate(case), layout=layout)
        graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
        brick = graph.model_for_block("transport")._m.emit_cpp_brick()
        assert "coordinated_face_contract_version = 1" in brick
        assert "coordinated_face(const State& left" not in brick  # signature spans lines
        assert "coordinated_face(" in brick
        assert "left_ncp.values[" in brick and "right_ncp.values[" in brick
        assert "result.speed_bound" in brick
        assert "PathRusanovFlux" not in brick
        emitted.append(brick)
        identities.append(face.to_data())
    assert emitted[0] != emitted[1]
    assert identities[0] != identities[1]


def test_public_face_refuses_nonzero_conservative_side_and_free_same_name_capture():
    _, _, original = _case()
    left = original.left_symbols
    with pytest.raises(ValueError, match="conservative component c"):
        CoordinatedFace(flux=original.flux, product=original.product,
            frame=original.frame,
            body=lambda a, b, axis: FaceBalance(a, (1., 0., 0.), (0., 0., 0.), 1.))
    same_name_other_owner = Var(left[0].name, left[0].kind)
    with pytest.raises(ValueError, match="free variable"):
        CoordinatedFace(flux=original.flux, product=original.product,
            frame=original.frame,
            body=lambda a, b, axis: FaceBalance(
                (same_name_other_owner, a[1], a[2]), (0., 0., 0.), (0., 0., 0.), 1.))


def test_public_two_dimensional_extrusion_emits_zero_transverse_face():
    case, layout, _ = _case(dimension=2)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    brick = graph.model_for_block("transport")._m.emit_cpp_brick()
    assert "coordinated_face_contract_version = 1" in brick
    assert "else if constexpr (Axis == 1)" in brick
    assert "result.right_ncp.values[2]" in brick


@pytest.mark.parametrize("order", (("c", "p", "r"), ("r", "c", "p")))
def test_authored_face_expr_matches_separate_three_state_oracle(order):
    import numpy as np
    _, _, authored = _case(order)
    left = np.array((.23, -.11, .32))[[ ("c", "p", "r").index(name) for name in order]]
    right = np.array((.19, -.06, .27))[[ ("c", "p", "r").index(name) for name in order]]
    values = {symbol.name: float(value)
              for symbols, data in ((authored.left_symbols, left),
                                    (authored.right_symbols, right))
              for symbol, value in zip(symbols, data, strict=True)}
    row = authored.balances[0]
    actual = (np.array([entry.eval(values) for entry in row.flux]),
              np.array([entry.eval(values) for entry in row.left]),
              np.array([entry.eval(values) for entry in row.right]),
              row.stability.eval(values))
    expected = oracle_face(left, right, beta=.7, gamma=-.4, order=order)
    for observed, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed, reference, rtol=0., atol=2.e-15)
