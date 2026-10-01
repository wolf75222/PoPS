"""Original history storage authority survives real public snapshot boundaries."""
import pytest

import pops
from pops._ir.elliptic import Reaction
from pops.domain import Rectangle
from pops.fields import FieldBoundary, FieldDiscretization, FieldProblem, bcs
from pops.fields.methods import CellCenteredSecondOrder
from pops.frames import Cartesian2D
from pops.math import ddt, div, laplacian
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.solvers import CG
from pops.time import FailRun
from pops.time._program.detach import detach_compiled_program
from pops.time._program.global_history_storage import storage_contract, validate_issuances


def authored(endpoint):
    frame = Rectangle("history-box", lower=(0, 0), upper=(1, 1)).frame(Cartesian2D())
    case, program = pops.Case("linear-history"), pops.Program("linear-history-program")
    blocks, states, times = [], [], []
    for index, width in enumerate((2, 5)):
        model = pops.Model("material-%d" % index, frame=frame)
        state = model.state("S", components=tuple("s%d" % i for i in range(width)))
        flux = model.flux("stationary", frame=frame, state=state,
            components={axis: tuple(0*u for u in state) for axis in frame.axes},
            waves={axis: tuple(0*u for u in state) for axis in frame.axes})
        rate = model.rate("balance", equation=ddt(state) == -div(flux))
        numerics = DiscretizationPlan()
        numerics.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
        block = case.block("block-%d" % index, model)
        case.numerics(numerics, block=block)
        blocks.append(block)
        states.append(state)
        times.append(program.state(block[state]))
    q = blocks[0]._instance_registry._blocks[blocks[0].local_id]["model"].field("T")
    problem = FieldProblem("original-linear", unknowns=(q,),
        equations=(-laplacian(q)+Reaction(q, 2) == states[0][0],),
        boundaries=(FieldBoundary(q, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic())),))
    field = case.field(problem, FieldDiscretization(method=CellCenteredSecondOrder(), boundaries=(), solver=CG(max_iter=200)))
    point = times[0].next.point if endpoint else program.stage("observation", c=0)
    outcome = program.solve(field, values={blocks[0][states[0]]: times[0].n}, at=point)
    observation = field.observe(outcome.consume(action=FailRun()))[field[q]]
    program.store_history("temperature", observation, depth=1, owner_block=blocks[1])
    return program, blocks, states


@pytest.mark.parametrize("endpoint", (False, True))
def test_public_linear_storage_detaches_original_proof_without_case_registry(endpoint):
    program, blocks, states = authored(endpoint)
    image = program._serialize()
    semantic_image = program._serialize(include_provenance=False)
    node = next(node for node in program._values if node.op == "store_history")
    assert node.inputs[0].block is None and program._histories_ncomp["temperature"] == 1
    assert node.block is blocks[1]
    assert len(tuple(states[1])) == 5
    graph = program.to_graph()
    assert graph.to_data()
    clone = detach_compiled_program(program)
    # Rebuild records its transformation provenance; the entire executable IR stays exact.
    assert clone._serialize(include_provenance=False) == semantic_image
    assert clone._ir_hash() == program._ir_hash() and program._serialize() == image
    assert clone._global_field_history_issuance is not program._global_field_history_issuance
    record = clone._global_field_history_issuance["temperature"]
    assert record.metadata["owner_block"]._instance_registry is None
    assert record.metadata["layout_witness"]._instance_registry is None
    assert record.metadata["storage_state_witness"].block_ref._instance_registry is None
    with pytest.raises(TypeError):
        record.metadata["ncomp"] = 5
    program.freeze()
    assert program._serialize() == image
    assert detach_compiled_program(program)._serialize(include_provenance=False) == semantic_image


@pytest.mark.parametrize("boundary", ("serialize", "freeze", "rebuild", "graph"))
@pytest.mark.parametrize("attack", ("owner-reseal", "qualifier-delete"))
def test_snapshot_boundaries_do_not_remint_or_downgrade_issued_storage(boundary, attack):
    program, blocks, _ = authored(False)
    node = next(node for node in program._values if node.op == "store_history")
    attrs = dict(node.attrs)
    if attack == "owner-reseal":
        attrs["global_field_storage"] = storage_contract(program, node.inputs[0], blocks[0])
        object.__setattr__(node, "block", blocks[0])
        program._history_blocks["temperature"] = blocks[0]
    else:
        del attrs["global_field_storage"]
    object.__setattr__(node, "attrs", attrs)
    action = {"serialize": program._serialize, "freeze": program.freeze,
              "rebuild": lambda: program._rebuild(lambda value: True), "graph": program.to_graph}[boundary]
    with pytest.raises(ValueError, match="originally issued|immutable authority"):
        action()
    assert not program._frozen


def test_original_issuance_cannot_be_reassigned_deleted_or_mutated():
    program, _, _ = authored(True)
    with pytest.raises(AttributeError, match="cannot be replaced"):
        program._global_field_history_issuance = {}
    with pytest.raises(AttributeError, match="cannot be deleted"):
        del program._global_field_history_issuance
    with pytest.raises(TypeError):
        program._global_field_history_issuance["temperature"] = None
    validate_issuances(program)


@pytest.mark.parametrize("key,replacement", (("ncomp", True), ("ncomp", 1.0), ("region", False)))
def test_issued_storage_metadata_preserves_exact_scalar_types(key, replacement):
    program, _, _ = authored(True)
    node = next(node for node in program._values if node.op == "store_history")
    attrs = dict(node.attrs)
    attrs["global_field_storage"] = dict(attrs["global_field_storage"], **{key: replacement})
    object.__setattr__(node, "attrs", attrs)
    with pytest.raises(ValueError, match="immutable authority"):
        program._serialize()


@pytest.mark.parametrize("width", (True, 1.0))
def test_issued_storage_registered_width_is_an_exact_integer(width):
    program, _, _ = authored(True)
    program._histories_ncomp["temperature"] = width
    with pytest.raises(ValueError, match="registration changed"):
        program._serialize()
