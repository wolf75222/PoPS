"""Spec 3 multi-species: coupled_rate operator kind + multi-output P.call (ADC-457).

A coupled operator (collisions, ionization, radiation) takes an arbitrary arity of states and
returns a typed RateBundle -- one Rate per participating block. P.call lowers it to a bundle of
per-block rate values usable in affine combinations. This is the IR/authoring slice (pure
Python, locally testable); the C++ coupled-rate kernel codegen is the deferred runtime part.
"""
from pops.codegen.program_codegen import _check_lowerable
from pops.codegen.program_codegen import emit_cpp_program
import pytest

from pops import model
from pops._ir.expr import Var
from pops.problem import Case
from tests.python.unit.runtime._typed_program import add_typed_block

adctime = pytest.importorskip("pops.time")


def _two_fluid_module():
    """(e, i) -> RateBundle{electrons: Rate(e), ions: Rate(i)} collision operator."""
    mod = model.Module("two_fluid")
    e = mod.state_space("electron_state", ("ne", "mex", "mey"))
    i = mod.state_space("ion_state", ("ni", "mix", "miy"))
    bundle = model.RateBundle({"electrons": model.Rate(e), "ions": model.Rate(i)})
    ne, ni = Var("ne", "cons"), Var("ni", "cons")
    mod.operator(name="collision", signature=model.Signature((e, i), bundle),
                 kind="coupled_rate",
                 expr={"electrons": [ni - ne, ne, ne], "ions": [ne - ni, ni, ni]})
    return mod, e, i, bundle


def _program_states(mod, name, declarations):
    """Build real case-qualified temporal states for a coupled Program."""
    case = Case(name="%s_case" % name)
    program = adctime.Program(name)
    result = {}
    for block_name, space in declarations:
        block, state = add_typed_block(case, mod, block_name, space)
        result[block_name] = program.state(block[state])
    return program, result


def test_coupled_rate_is_a_valid_kind():
    assert "coupled_rate" in model.OPERATOR_KINDS


def test_rate_bundle_equality_and_hash():
    e = model.StateSpace("e", ("a",))
    i = model.StateSpace("i", ("b",))
    b1 = model.RateBundle({"electrons": model.Rate(e), "ions": model.Rate(i)})
    b2 = model.RateBundle({"electrons": model.Rate(e), "ions": model.Rate(i)})
    assert b1 == b2 and hash(b1) == hash(b2)
    assert b1 != model.RateBundle({"electrons": model.Rate(e)})


def test_rate_bundle_is_an_immutable_hash_stable_signature_value():
    e = model.StateSpace("e", ("a",))
    bundle = model.RateBundle({"electrons": model.Rate(e)})
    signature = model.Signature((e,), bundle)
    lookup = {bundle: "rate", signature: "signature"}

    assert lookup[bundle] == "rate" and lookup[signature] == "signature"
    assert not hasattr(bundle, "add")
    with pytest.raises(TypeError):
        bundle._rates["ions"] = model.Rate(model.StateSpace("i", ("b",)))
    with pytest.raises(AttributeError, match="immutable"):
        bundle._rates = {}
    assert lookup[bundle] == "rate" and lookup[signature] == "signature"


def test_coupled_rate_operator_registers_with_bundle_output():
    mod, _, _, bundle = _two_fluid_module()
    op = mod.operator_registry().get("collision")
    assert op.kind == "coupled_rate"
    assert op.signature.output == bundle


def test_p_call_coupled_rate_returns_indexable_bundle():
    mod, e, i, _ = _two_fluid_module()
    P, states = _program_states(mod, "step", (("electrons", e), ("ions", i)))
    e_n, i_n = states["electrons"].n, states["ions"].n
    C = mod.operator_handle("collision")(e_n, i_n)
    re_, ri_ = C[e_n.block], C[i_n.block]
    assert re_.vtype == "rhs" and ri_.vtype == "rhs"
    # each per-block rate is usable in an affine combination of its block's state
    e1 = P.value("e1", e_n + P.dt * re_, at=states["electrons"].next.point)
    i1 = P.value("i1", i_n + P.dt * ri_, at=states["ions"].next.point)
    assert e1.vtype == "state" and i1.vtype == "state"


def test_coupled_rate_arbitrary_arity_three_blocks():
    mod = model.Module("three_fluid")
    e = mod.state_space("e", ("ne",))
    i = mod.state_space("i", ("ni",))
    n = mod.state_space("n", ("nn",))
    bundle = model.RateBundle({"e": model.Rate(e), "i": model.Rate(i), "n": model.Rate(n)})
    z = Var("ne", "cons")
    mod.operator(name="coll3", signature=model.Signature((e, i, n), bundle),
                 kind="coupled_rate", expr={"e": [z], "i": [z], "n": [z]})
    P, states = _program_states(mod, "s", (("e", e), ("i", i), ("n", n)))
    en, inn, nn = states["e"].n, states["i"].n, states["n"].n
    C = mod.operator_handle("coll3")(en, inn, nn)
    assert set(C.keys()) == {en.block, inn.block, nn.block}


def test_coupled_rate_bundle_unknown_block_errors():
    mod, e, i, _ = _two_fluid_module()
    P, states = _program_states(mod, "step", (("electrons", e), ("ions", i)))
    C = mod.operator_handle("collision")(
        states["electrons"].n, states["ions"].n)
    with pytest.raises(KeyError):
        _ = C["neutrals"]


def test_coupled_rate_rejects_schedule_clearly():
    # schedule= on a coupled_rate has no single output to schedule yet -> clear error, not a raw
    # AttributeError from the _CoupledResult having no .attrs.
    mod, e, i, _ = _two_fluid_module()
    P, states = _program_states(mod, "step", (("electrons", e), ("ions", i)))
    with pytest.raises(ValueError, match="coupled_rate"):
        mod.operator_handle("collision")(
            states["electrons"].n, states["ions"].n,
            schedule=adctime.every(2, clock=P.clock))


def test_dump_cpp_plan_shows_coupled_rate_kernel():
    # the C++ plan shows the coupled_rate as ONE multi-state kernel (ADC-457), never a
    # ctx.coupled_rate(...) call that does not exist.
    mod, e, i, _ = _two_fluid_module()
    P, states = _program_states(mod, "step", (("electrons", e), ("ions", i)))
    e_n, i_n = states["electrons"].n, states["ions"].n
    C = mod.operator_handle("collision")(e_n, i_n)
    P.value(
        "e1", e_n + P.dt * C[e_n.block], at=states["electrons"].next.point)
    plan = P.dump_cpp_plan()
    assert "ADC-457" in plan and "ctx.coupled_rate(" not in plan
    assert "multi-state for_each_cell rate kernel" in plan
    assert "electrons" in plan and "ions" in plan


def test_coupled_rate_now_lowers_to_cpp():
    # ADC-457 part B: a coupled_rate lowers to a multi-state kernel rather than refusing. The honest
    # deferral is now scoped to prim/aux formulas (the cons-only MVP) -- see
    # test_coupled_rate_codegen.py for the emitted kernel shape and the prim-var raise.
    mod, e, i, _ = _two_fluid_module()
    P, states = _program_states(mod, "step", (("electrons", e), ("ions", i)))
    e_n, i_n = states["electrons"].n, states["ions"].n
    C = mod.operator_handle("collision")(e_n, i_n)
    P.commit_many({
        states["electrons"].next:
            P.value(
                "e1", e_n + P.dt * C[e_n.block],
                at=states["electrons"].next.point),
        states["ions"].next:
            P.value(
                "i1", i_n + P.dt * C[i_n.block],
                at=states["ions"].next.point),
    })
    _check_lowerable(P, None)  # no longer raises for a cons-only coupled_rate
    src = emit_cpp_program(P, model=None)
    assert src.count("pops::for_each_cell") == 1
    assert "const pops::Real ne =" in src and "const pops::Real ni =" in src
