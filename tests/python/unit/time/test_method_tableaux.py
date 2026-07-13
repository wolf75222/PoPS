"""ADC-663: exact RK/ARK authorities and proof-carrying method properties."""
from fractions import Fraction

from pops.lib.time.rk import RK4_TABLEAU, SSPRK2_TABLEAU, rk, rk4
from pops.lib.time.ssprk import SSPRK2, SSPRK3_TABLEAU, ssprk3
from pops.time import Program
from pops.time.method_properties import UnknownOrder
from pops.time.method_properties import certify_program_graph
from pops.time.method_tableau import AdditiveRungeKuttaTableau, RungeKuttaTableau
from pops.time import StagePoint, TimePoint
from pops.numerics.terms import DefaultSource, Flux

from typed_program_support import state_refs


def test_classical_tableaux_have_exact_certified_properties():
    assert RK4_TABLEAU.properties.order == 4
    assert RK4_TABLEAU.properties.stability_polynomial == (
        1, 1, Fraction(1, 2), Fraction(1, 6), Fraction(1, 24))
    assert RK4_TABLEAU.properties.flux_weights == (
        Fraction(1, 6), Fraction(1, 3), Fraction(1, 3), Fraction(1, 6))
    assert RK4_TABLEAU.properties.ssp is None

    assert SSPRK2_TABLEAU.properties.order == 2
    assert SSPRK2_TABLEAU.properties.ssp.coefficient == 1
    assert SSPRK3_TABLEAU.properties.order == 3
    assert SSPRK3_TABLEAU.properties.abscissae == (0, 1, Fraction(1, 2))
    assert SSPRK3_TABLEAU.properties.ssp.coefficient == 1

    renamed = RungeKuttaTableau(
        SSPRK2_TABLEAU.A, SSPRK2_TABLEAU.b, SSPRK2_TABLEAU.c, name="manual-heun")
    impostor = RungeKuttaTableau(A=[[]], b=[1], c=[0], name="ssprk2")
    assert renamed.properties == SSPRK2_TABLEAU.properties
    assert renamed.certificate == SSPRK2_TABLEAU.certificate
    assert impostor.properties.ssp is None


def test_lower_order_is_proved_from_exact_conditions():
    method = RungeKuttaTableau(A=[[], [Fraction(1, 3)]], b=[0, 1], c=[0, Fraction(1, 3)])
    assert method.properties.order == 1


def test_ark_partition_abscissae_are_exact_and_immutable():
    explicit = RungeKuttaTableau(A=[[], [1]], b=[Fraction(1, 2)] * 2, name="heun")
    ark = AdditiveRungeKuttaTableau(
        explicit,
        implicit_A=[[Fraction(1, 2)], [0, Fraction(1, 2)]],
        implicit_b=[Fraction(1, 2)] * 2,
        name="partitioned",
    )
    assert ark.abscissae == ((0, Fraction(1, 2)), (1, Fraction(1, 2)))
    assert ark.properties.order == 2
    assert dict(ark.properties.flux_weights) == {
        "explicit": (Fraction(1, 2), Fraction(1, 2)),
        "implicit": (Fraction(1, 2), Fraction(1, 2)),
    }
    try:
        ark.name = "changed"
    except (AttributeError, TypeError):
        pass
    else:
        raise AssertionError("ARK tableau must be immutable")


def _graph(builder, name):
    program = Program(name)
    builder(program, *state_refs(program, "plasma"))
    return program.to_graph()


def test_presets_and_tableau_authority_have_identical_program_graphs():
    preset = _graph(rk4, "rk4")
    manual = Program("rk4")
    rk(manual, *state_refs(manual, "plasma"), tableau=RK4_TABLEAU)
    assert preset.to_data() == manual.to_graph().to_data()
    assert preset.graph_hash == manual.to_graph().graph_hash

    for preset_builder, tableau, name in (
        (SSPRK2, SSPRK2_TABLEAU, "ssprk2"),
        (ssprk3, SSPRK3_TABLEAU, "ssprk3"),
    ):
        preset = _graph(preset_builder, name)
        manual = Program(name)
        rk(manual, *state_refs(manual, "plasma"), tableau=tableau)
        assert preset.graph_hash == manual.to_graph().graph_hash


def test_method_certificate_is_reconstructed_from_program_graph_not_preset_identity():
    preset = _graph(ssprk3, "ssprk3")
    manual = Program("manual-three-stage")
    rk(manual, *state_refs(manual, "plasma"), tableau=SSPRK3_TABLEAU)
    manual_graph = manual.to_graph()
    preset_certificate = certify_program_graph(preset)
    manual_certificate = certify_program_graph(manual_graph)
    assert preset_certificate.properties == SSPRK3_TABLEAU.properties
    assert manual_certificate.properties == preset_certificate.properties
    assert manual_certificate.tableau == preset_certificate.tableau
    assert manual_certificate.tableau.A == SSPRK3_TABLEAU.certificate.A
    assert manual_certificate.tableau.b == SSPRK3_TABLEAU.certificate.b
    assert manual_certificate.tableau.c == SSPRK3_TABLEAU.certificate.c
    assert manual_certificate.properties.stability_polynomial == (
        1, 1, Fraction(1, 2), Fraction(1, 6))
    assert manual_certificate.properties.flux_weights == SSPRK3_TABLEAU.properties.flux_weights
    assert manual_certificate.properties.ssp == SSPRK3_TABLEAU.properties.ssp
    assert manual_certificate.graph_hash != preset_certificate.graph_hash  # Program names differ.


def _rhs_at(program, state, name, offset):
    point = StagePoint(name, {"main": TimePoint(program.clock, offset)})
    fields = program._replace_value(program.solve_fields(state), point=point)
    return program._replace_value(program.rhs(
        state=state, fields=fields, terms=[Flux(), DefaultSource()]), point=point)


def test_arbitrary_valid_three_stage_program_compiles_with_unknown_order():
    """A valid graph outside consistent RK is executable; lack of proof is metadata, not rejection."""
    program = Program("arbitrary-three-stage")
    block, state_handle = state_refs(program, "plasma")
    temporal = program.state(block, state_handle)
    u0 = temporal.n
    k0 = _rhs_at(program, u0, "arbitrary-0", 0)
    p1 = StagePoint("arbitrary-1", {"main": TimePoint(program.clock, Fraction(1, 3))})
    u1 = program.value("arbitrary-u1", u0 + Fraction(1, 3) * program.dt * k0, at=p1)
    k1 = _rhs_at(program, u1, "arbitrary-1", Fraction(1, 3))
    p2 = StagePoint("arbitrary-2", {"main": TimePoint(program.clock, Fraction(2, 3))})
    u2 = program.value(
        "arbitrary-u2", u0 + Fraction(2, 3) * program.dt * k1, at=p2)
    k2 = _rhs_at(program, u2, "arbitrary-2", Fraction(2, 3))
    out = program.value(
        "arbitrary-step",
        u0 + Fraction(1, 4) * program.dt * k0
        + Fraction(1, 4) * program.dt * k1 + Fraction(1, 4) * program.dt * k2,
        at=temporal.next.point,
    )
    program.commit(temporal.next, out)
    assert program.validate() is True
    assert "ctx.rhs_into(" in program.emit_cpp_program()
    graph = program.to_graph()
    certificate = certify_program_graph(graph)
    assert certificate.graph_hash == graph.graph_hash
    assert certificate.tableau is None
    assert isinstance(certificate.properties.order, UnknownOrder)
    assert certificate.properties.abscissae == (0, Fraction(1, 3), Fraction(2, 3))
