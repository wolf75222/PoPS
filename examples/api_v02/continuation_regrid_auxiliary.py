"""A two-component State and an independently shaped, consumed scalar history.

State storage has two ghost cells; the retained one-component Field has one.
Its new target allocation is explicit, and every value is transferred by the
same Python index algebra used in the State-history reception.
"""
from __future__ import annotations

import numpy as np
import pops
from pops import math as pm
from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
                      Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
from pops.continuations import ContinuationSchema
from pops.fields import FieldStorageBinding
from pops.layouts import AMR
from pops.linalg import FiniteSupport
from pops.lib.amr import StateTransfer
from pops.mesh import CartesianGrid, LayoutPlanBuilder, PeriodicAxes
from pops.model import PhysicalDimension, PhysicalSupport
from pops.numerics import DiscretizationPlan, StateStorage
from pops.params import RuntimeParam
from pops.time import FixedDt, every
from pops.time._program.distributed_arrays import Distributed
from pops.time.expressions import ProgramScalar

from .common import CaseSpec, non_owner_metrics
from .continuation_regrid_amr import _approve, _bundle_mass, _injection, ab2_factors
from .models_M01_M14 import _domain, _initial


def initial_auxiliary_values(spec, plan):
    """Initialize this declared input FieldSpace with the actual scalar q(0).

    FieldSample subsequently creates the independently owned AB2 history from
    State. The input is required by the FieldSpace declaration; it is not a
    substitute for the calculated history used by the method.
    """
    from pops.codegen.provider_instances import require_instance_contract, runtime_key
    from pops.model.provider_pack import ProviderPack

    initial = next(iter(spec.initial_values.values()))
    result = {}
    for block in plan.blocks:
        operations = block.resolved_operations
        pack = ProviderPack.from_data(operations.to_data()["provider_evidence"]["auxiliary"])
        instance = require_instance_contract(operations, owner_qid=block.instance_owner_qid)
        for key in pack:
            entry = pack.declared_entry(key)
            if entry.producer != "runtime_input" or not entry.availability or entry.slot is None:
                continue
            assert (key.space_kind, key.space_name, key.component) == (
                "field", "previous", "q_previous"), "an additional required input needs its own data"
            result[runtime_key(key, instance)] = np.ascontiguousarray(initial[0], dtype=np.float64)
    assert len(result) == 1, "the declared retained FieldSpace input must remain explicit"
    return result


def build_continuation_regrid_auxiliary(*, cells=(16, 8), dt=.1, steps=3):
    frame, _ = _domain("auxiliary_regrid", cells)
    model = pops.Model("independent_retained_field", frame=frame)
    unit = PhysicalDimension()
    support = PhysicalSupport((("x", "auxiliary_regrid"), ("y", "auxiliary_regrid")))
    state = model.state("q", components=("q", "twice_q"), units=(unit, unit),
                         support=support, sampling="cell")
    old_space = model.module.field_space("previous", ("q_previous",), units=(unit,),
        support=support, frame=frame.canonical_id, sampling="cell", representation="field")
    case = pops.Case("explicit_target_auxiliary_storage")
    block = case.block("method", model, states=(state,))
    numerics = DiscretizationPlan()
    numerics.states.add(state, StateStorage(ghost_depth=2))
    case.numerics(numerics, block=block)
    p = pops.Program("ab2_distinct_field_storage")
    q = p.state(block[state])
    transfer = AMRTransfer()
    transfer.state(block[state], StateTransfer())
    threshold = case.param(RuntimeParam("auxiliary_regrid_threshold", default=1.55))
    grid = AMR(grid=CartesianGrid(frame=frame, cells=cells, periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(pm.ValueExpr(block[state])["q"] > case.value(threshold)),
                                 Buffer(cells=1, selection="refine-requests")),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid(schedule=every(2, clock=p.clock)), transfer=transfer,
        execution=AMRExecution.synchronous())
    layouts = LayoutPlanBuilder(case.owner_path.canonical())
    layout = layouts.layout("amr", grid.resolve_for_case(case.resolve))
    layouts.assign_state(case.resolve(block[state]), layout)
    layouts.assign_block(case.resolve(block), layout)
    storage = FieldStorageBinding((model.module.field_handle(old_space),), layout, (0, 1))

    def sample(b, state_value):
        return b.field_sample(state_value, component="q", template=state_value,
                              storage=storage, space=old_space)

    current = q.n
    previous = sample(p, current)
    domain = p.range_set(3, capacity=3, ownership="replicated")
    controls = p.map(domain, lambda b, i: b.index_value(i))
    initial = p.product(state=current, previous=previous, time=p.float64(0), step=p.int64(0),
                        counter=p.int64(0), controls=controls)
    schema = ContinuationSchema(initial, fields=((q.next, ("state",)),),
        retained_fields=(("previous",),), time=("time",), step=("step",))
    base = p.continuation_open(initial, schema)
    entry = p.continuation_read(base)
    proposal = p.product(h=p.requested_dt())
    increment = ProgramScalar(proposal["h"])*(1.5*entry["state"][0]-.5*entry["previous"][0])
    advanced = FiniteSupport("ab2_distinct_width", ("q", "twice_q")).bind((
        entry["state"][0]+increment, entry["state"][1]+2.*increment)).materialize(
            p, "ab2_distinct_width", template=entry["state"], at=q.next.point)
    trial = p.product(state=advanced, previous=sample(p, entry["state"]),
        time=entry["time"]+proposal["h"], step=entry["step"]+p.int64(1),
        counter=entry["counter"]+p.int64(1), controls=entry["controls"])
    approved = _approve(p, base, proposal, trial, p.product(), advancing=True)

    def synchronize(b, base, proposal):
        entry, candidate = b.continuation_read(base), proposal["candidate"]
        result = b.product(state=proposal["synchronized"]["field_0"], previous=candidate["previous"],
            time=proposal["interval"]["end"], step=entry["step"]+b.int64(1),
            counter=candidate["counter"], controls=candidate["controls"])
        return _approve(b, base, proposal, result, b.product(counter=candidate["counter"]), advancing=True)

    def initialize(b):
        mass = b.composite_integral("auxiliary_initial_mass", b.attempt_start(q.n)[0], over=q.n)
        value = b.product(time=b.float64(0), step=b.int64(0), counter=b.int64(0), mass=mass, regrids=b.int64(0))
        return value, ContinuationSchema(value, time=("time",), step=("step",))

    def observe(b):
        return b.product(mass=b.composite_integral("auxiliary_mass", q.n[0], over=q.n))

    def parent(b, base, proposal):
        entry = b.continuation_read(base)
        value = b.product(time=proposal["interval"]["end"], step=entry["step"]+b.int64(1),
            counter=entry["counter"]+b.int64(1), mass=proposal["observations"]["mass"], regrids=entry["regrids"])
        return _approve(b, base, proposal, value, b.product(), advancing=True)

    def remap(b, proposal):
        algebra = Distributed(b, storage_bytes=64*1024*1024, storage_versions=4096)
        source_level = b.int64(0)
        base = b.regrid_base(source_level)
        entry = b.continuation_read(base)
        source = b.regrid_source(source_level)
        template = proposal["templates"]["field_0"]

        def move(field, target_template, family=None):
            source_view = b.regrid_topology(field, level=source_level, side="source", dimension=2)
            exported = b.field_bundle(field, topology=source_view, cell_capacity=32768,
                                       patch_capacity=256, family=family)
            target_view = b.regrid_topology(target_template, level=proposal["level"], side="target", dimension=2)
            if family is None:
                storage = target_template
            else:
                storage = b.scalar_field(like=target_template, topology=target_view, initial=b.float64(0))
            target = b.field_bundle(storage, topology=target_view, cell_capacity=32768,
                                     patch_capacity=256, family=exported)
            transferred = _injection(b, algebra, exported, target,
                target_level=proposal["level"], cells=cells)
            return b.bundle_field(transferred, template=target_template, topology=target_view), transferred

        current, bundle = move(source["state"], template)
        old, _ = move(source["previous"], source["previous"], family=bundle)
        value = b.product(state=current, previous=old, time=entry["time"], step=entry["step"],
                           counter=entry["counter"], controls=entry["controls"])
        return _approve(b, base, proposal, value, b.product(mass=_bundle_mass(b, algebra, bundle)), advancing=False)

    def remap_parent(b, base, proposal):
        entry = b.continuation_read(base)
        mass = b.reduce(proposal["children"], 0.*entry["mass"],
                        lambda z, total, child: total+child["mass"], order="ascending")
        value = b.product(time=entry["time"], step=entry["step"], counter=entry["counter"],
                            mass=mass, regrids=entry["regrids"]+b.int64(1))
        return _approve(b, base, proposal, value, b.product(), advancing=False, mass=True)

    p.continuation_forest(approved, synchronize=synchronize, initialize=initialize, parent=parent,
        observe=observe, level_capacity=8, regrid=remap, regrid_parent=remap_parent, dimension=2)
    p.step_strategy(FixedDt(dt))
    case.program(p)
    layout_plan = layouts.resolve(states=(case.resolve(block[state]),), blocks=(case.resolve(block),))
    data = np.ones((2, cells[1], cells[0]))
    data[0] *= (1.+.5*np.sin(2*np.pi*(np.arange(cells[0])+.5)/cells[0]))[None, :]
    data[1] = 2.*data[0]

    def check(runtime):
        arrays = [runtime.block_level_state_global("method", i) for i in range(runtime.n_levels())]
        if arrays[0] is None:
            return non_owner_metrics()
        expected = ab2_factors(dt, runtime.macro_step())[-1]*data
        # Native receiver checks every valid coarse and fine cell, including retained history.
        error = float(np.max(abs(np.asarray(arrays[0]).reshape(expected.shape)-expected)))
        return dict(passed=error < 5.e-13, metrics_owner=True, reference_error=error)

    spec = CaseSpec(case, layout_plan, _initial(case, block[state], data), {"method": block[state]},
        steps*dt, dt, check, "AB2 retained Field1/ghost1 remapped alongside State2/ghost2")
    return spec, {layout: grid.resolve_for_case(case.resolve)}
