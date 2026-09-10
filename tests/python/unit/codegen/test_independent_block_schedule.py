"""Independent states require no cross-block residual evaluation barrier."""
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.time import FixedDt
from pops.codegen._orchestration_compile import build_program_model_graph
from pops.codegen.program_codegen import emit_cpp_program


def _independent_periodic_program():
    frame = Rectangle("domain", lower=(0, 0), upper=(1, 1)).frame(Cartesian2D())
    model = pops.Model("transport", frame=frame)
    state = model.state("U", components=("u",))
    flux = model.flux("flux", frame=frame, state=state,
                      components={axis: (state[0],) for axis in frame.axes},
                      waves={axis: (1,) for axis in frame.axes})
    rate = model.rate("transport", equation=ddt(state) == -div(flux))
    case = pops.Case("independent_blocks")
    blocks = (case.block("left", model), case.block("right", model))
    for block in blocks:
        plan = DiscretizationPlan()
        plan.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
                                         reconstruction=reconstruction.FirstOrder(),
                                         riemann=riemann.Rusanov()))
        case.numerics(plan, block=block)
    program = pops.Program("ordinary_sequential_authoring")
    point = program.stage("evaluate", c=0)
    for block in blocks:
        q = program.state(block[state])
        rhs = program.value(block.local_id + "_rhs", rate(q.n), at=point)
        advanced = program.value(block.local_id + "_next", q.n + program.dt * rhs,
                                  at=q.next.point)
        program.commit(q.next, advanced)
    program.step_strategy(FixedDt(.001))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 8), periodic=PeriodicAxes(frame.axes)))
    resolved = pops.resolve(pops.validate(case), layout=layout)
    graph = build_program_model_graph(resolved)
    return resolved, graph


def test_independent_periodic_blocks_allow_each_complete_update_before_the_next():
    from pops.codegen.program_emit_solve import _rhs_evaluation_identity
    from pops.time._program.detach import detach_compiled_program
    resolved, graph = _independent_periodic_program()
    assert dict(graph.rhs_coherence_neighbours) == {"left": frozenset(), "right": frozenset()}
    for program in (resolved.time, detach_compiled_program(resolved.time)):
        for target in ("system", "amr_system"):
            source = emit_cpp_program(program, model_graph=graph, target=target)
            assert "ctx.rhs_group(" not in source
        for value in program._values:
            if value.op == "rhs":
                assert _rhs_evaluation_identity(program, value, model=graph) == value.id


def _sequential_program(names):
    from pops.numerics.terms import Flux
    from typed_program_support import typed_state
    program = pops.Program("sequential_rhs")
    for name in names:
        state = typed_state(program, name, state_name="U")
        residual = program.rhs(name + "_rhs", state=state.n, terms=[Flux()])
        value = program.value(name + "_next", state.n + program.dt * residual, at=state.next.point)
        program.commit(state.next, value)
    return program


def test_connected_blocks_still_refuse_early_consumer_and_unknown_graph_is_conservative():
    import pytest
    from pops.codegen._rhs_coherence import plan_rhs_coherence
    program = _sequential_program(("left", "right"))
    connected = {"left": frozenset(("right",)), "right": frozenset(("left",))}
    for neighbours in (None, connected):
        with pytest.raises(ValueError, match="cannot delay.*left_next"):
            plan_rhs_coherence(program, program._values, neighbours=neighbours)


def test_transitive_interface_cohort_does_not_absorb_an_independent_block():
    from pops.codegen._rhs_coherence import plan_rhs_coherence
    from pops.numerics.terms import Flux
    from typed_program_support import typed_state
    program = _sequential_program(("independent",))
    states = [typed_state(program, name, state_name="U") for name in ("left", "middle", "right")]
    rates = [program.rhs(name + "_rhs", state=state.n, terms=[Flux()])
             for name, state in zip(("left", "middle", "right"), states, strict=True)]
    neighbours = {"independent": frozenset(), "left": frozenset(("middle",)),
                  "middle": frozenset(("left", "right")), "right": frozenset(("middle",))}
    plan = plan_rhs_coherence(program, program._values, neighbours=neighbours)
    assert tuple(tuple(value.name for value in group) for group in plan.schedule.values()) == (
        ("left_rhs", "middle_rhs", "right_rhs"),)
    assert plan.grouped_ids == frozenset(value.id for value in rates)


def test_shared_field_state_dependencies_connect_otherwise_separate_blocks():
    from pops.codegen._rhs_coherence import plan_rhs_coherence
    from pops.numerics.terms import DefaultSource, Flux
    from typed_program_support import typed_state, solve_field_blocks
    program = pops.Program("shared_field_dependency")
    left = typed_state(program, "left", state_name="U")
    right = typed_state(program, "right", state_name="U")
    fields = solve_field_blocks(program, (left.n, right.n))
    left_rate = program.rhs("left_rhs", state=left.n, fields=fields, terms=[Flux(), DefaultSource()])
    right_rate = program.rhs("right_rhs", state=right.n, fields=fields, terms=[Flux(), DefaultSource()])
    plan = plan_rhs_coherence(program, program._values,
                             neighbours={"left": frozenset(), "right": frozenset()})
    assert plan.grouped_ids == frozenset((left_rate.id, right_rate.id))


def test_resolved_unknown_boundary_contract_cannot_prove_independence():
    from types import SimpleNamespace
    from pops.codegen._rhs_coherence import resolved_rhs_neighbours
    for numerics in (None, SimpleNamespace(boundaries=(object(),))):
        blocks = (SimpleNamespace(name="left", state_identities=("left-state",), numerics=numerics),)
        assert resolved_rhs_neighbours(blocks) is None


def test_compiled_boundary_interfaces_and_state_reads_preserve_transitive_connections():
    from types import SimpleNamespace
    from pops.codegen._rhs_coherence import resolved_rhs_neighbours
    from pops.mesh.boundaries.compiled_plan import CompiledBoundaryPlan
    interface = {"identity": "shared-left-right", "left": "left-boundary", "right": "right-boundary"}
    def block(name, interfaces=(), state_reads=()):
        boundary = CompiledBoundaryPlan({"schema_version": 1,
            "authority_type": "prepared_boundary_plan_compile", "ghost_plan_identity": name,
            "faces": [], "component_region_templates": [{"states": list(state_reads), "fields": []}],
            "interfaces": list(interfaces)})
        return SimpleNamespace(name=name, state_identities=(name + "-state",),
                               numerics=SimpleNamespace(boundaries=(boundary,)))
    blocks = (block("left", (interface,)), block("right", (interface,), ("third-state",)), block("third"))
    expected = {"left": frozenset(("right",)), "right": frozenset(("left", "third")),
                "third": frozenset(("right",))}
    assert resolved_rhs_neighbours(blocks) == expected
    assert resolved_rhs_neighbours(tuple(reversed(blocks))) == expected
    assert resolved_rhs_neighbours((block("left", state_reads=("missing-state",)),)) is None


def test_partial_connectivity_cannot_silently_declare_a_missing_block_independent():
    import pytest
    from pops.codegen._rhs_coherence import plan_rhs_coherence
    program = _sequential_program(("left", "right"))
    with pytest.raises(ValueError, match="omits participating"):
        plan_rhs_coherence(program, program._values, neighbours={"left": frozenset()})


def test_two_refreshes_of_one_shared_field_cannot_split_a_coherent_residual_round():
    import pytest
    from pops.codegen._rhs_coherence import plan_rhs_coherence
    from pops.numerics.terms import DefaultSource, Flux
    from typed_program_support import typed_state, solve_field
    program = pops.Program("same_field_two_refreshes")
    left = typed_state(program, "left", state_name="U")
    right = typed_state(program, "right", state_name="U")
    first = solve_field(program, left.n)
    program.rhs("left_rhs", state=left.n, fields=first, terms=[Flux(), DefaultSource()])
    second = solve_field(program, right.n)
    program.rhs("right_rhs", state=right.n, fields=second, terms=[Flux(), DefaultSource()])
    with pytest.raises(ValueError, match="ordering barrier"):
        plan_rhs_coherence(program, program._values,
                           neighbours={"left": frozenset(), "right": frozenset()})
