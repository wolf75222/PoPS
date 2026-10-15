"""AB2 with a retained Field, and a complete Python-authored AMR remap.

The physical equation is q'=q.  Its deliberately simple spatial behavior makes
the independent oracle exact while still requiring the old Field to move to a
new fine layout.  The regrid law is periodic piecewise-constant injection from
level zero; neither that choice nor AB2 is implemented by the native port.
"""
from __future__ import annotations

import numpy as np
from pops import math as pm
from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging,
                      AMRTransfer, Buffer, ConflictPolicy, EqualityPolicy,
                      Hysteresis, Tag)
from pops.bundles import BundleSchema
from pops.continuations import Assess, Candidate, ContinuationSchema, accept, assess
from pops.graph_functions import graph
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.params import RuntimeParam
from pops.time import FixedDt, every
from pops.time._program.distributed_arrays import Distributed
from pops.time.variant_types import Bool, Float64, Int64, Product, Result, type_of

from .common import CaseSpec, non_owner_metrics
from .methods_T01_T12 import _storage
from .models_M01_M14 import _initial


def ab2_factors(dt, steps):
    """Independent scalar recurrence; no use of the Program equations."""
    factors = [1.0]
    previous = 1.0
    for _ in range(steps):
        current = factors[-1]
        factors.append((1.0 + 1.5*dt)*current - 0.5*dt*previous)
        previous = current
    return factors


def _indicator(p, condition):
    return p.branch(condition, lambda q: q.int64(1), lambda q: q.int64(0))


def _approve(p, base, proposal, after, report, *, advancing, mass=False, reject=False):
    before = p.continuation_read(base)
    candidate = Candidate(p, base=base, proposal=proposal, continuation=after,
                          increment=p.product(), observations=report, checks=p.product())
    envelope = p.continuation_candidate_value(candidate)

    @graph(inputs=Product(base=type_of(before), proposal=type_of(proposal), candidate=type_of(envelope)),
           output=Product(accepted=Bool, checks=Product(stage=Int64)))
    def assessment(q, args):
        entry, result = args["base"], args["candidate"]["continuation"]
        decision = _indicator(q, result["step"] == entry["step"]+q.int64(int(advancing)))
        if not advancing:
            decision = decision*_indicator(q, result["time"] == entry["time"])
        if mass:
            decision = decision*_indicator(q, result["mass"] >= entry["mass"]*(1.-3.e-13))
            decision = decision*_indicator(q, result["mass"] <= entry["mass"]*(1.+3.e-13))
        if reject:
            # This is assessed only after every remapped child was prepared.
            decision = decision*q.int64(0)
        return q.product(accepted=decision == q.int64(1),
                         checks=q.product(stage=q.int64(71 if reject else 0)))

    return p.require_success(accept(p, candidate, assess(p, candidate, Assess(assessment))))


def _with_values(p, target, values):
    raw = p.product(instance=target["instance"], fields=p.product(values=values),
                    geometry=target["geometry"], maps=target["maps"], histories=target["histories"])
    bindings = {("fields", "values"): "cells", ("maps", "patches"): "patches"}
    for name in ("coordinates", "indices", "is_active", "is_covered", "is_valid",
                 "measure", "volume_fraction"):
        bindings[("geometry", name)] = "cells"
    return BundleSchema(raw, domains={"cells": ("instance",),
        "patches": ("maps", "patch_instance")}, bindings=bindings).bind(p, raw)


def _injection(p, algebra, source, target, *, target_level, cells):
    """Global target-to-source index map, written entirely in finite Python algebra."""
    scale = p.branch(target_level == p.int64(0), lambda q: q.int64(1), lambda q: q.int64(2))

    def source_index(q, index):
        position = algebra.at_local(target["geometry"]["indices"], index)
        x = (position["i0"]//scale) % q.int64(cells[0])
        y = (position["i1"]//scale) % q.int64(cells[1])

        def locate(b, found, patch):
            inside = (_indicator(b, x >= patch["lo0"])*_indicator(b, x <= patch["hi0"])
                      *_indicator(b, y >= patch["lo1"])*_indicator(b, y <= patch["hi1"])
                      *b.branch(patch["is_replica"], lambda z: z.int64(0), lambda z: z.int64(1)))
            width = patch["hi0"]-patch["lo0"]+1+2*patch["ghost0"]
            offset = ((y-patch["lo1"]+patch["ghost1"])*width
                      +x-patch["lo0"]+patch["ghost0"])
            return b.branch(inside == b.int64(1), lambda z: patch["global_offset"]+offset,
                            lambda z: found)

        gid = q.reduce(source["maps"]["patches"], q.int64(-1), locate, order="ascending")
        return algebra.index(source["instance"], gid)

    requests = algebra.requests(target["instance"], source["instance"], source_index)
    values = algebra.import_values(source["fields"]["values"], requests, byte_capacity=8*1024*1024)
    return _with_values(p, target, values)


def _bundle_mass(p, algebra, bundle):
    def weighted(q, i):
        value = algebra.at_local(bundle["fields"]["values"], i)["c0"]
        measure = algebra.at_local(bundle["geometry"]["measure"], i)
        return q.variant(Result(Float64, Int64), "ok", value*measure)
    values = p.require_success(algebra.local_map(bundle["instance"], weighted,
                                                byte_capacity=8*1024*1024))
    return algebra.reduce(values, p.float64(0), lambda q, total, value: total+value,
                          order="global_gid_ascending", byte_capacity=8*1024*1024)


def build_continuation_regrid_amr(*, cells=(16, 8), dt=.1, steps=3, reject_regrid_parent=False):
    parts = _storage("ab2_complete_regrid", ("q",), (1.,), cells=cells)
    case, _, model, block, state, p, q, data = parts
    data[0] = 1.+.5*np.sin(2*np.pi*(np.arange(cells[0])+.5)/cells[0])[None, :]
    current_field = q.n
    domain = p.range_set(3, capacity=3, ownership="replicated")
    controls = p.map(domain, lambda b, i: b.index_value(i))
    initial = p.product(state=current_field, previous=current_field, time=p.float64(0),
                        step=p.int64(0), counter=p.int64(0), controls=controls)
    schema = ContinuationSchema(initial, fields=((q.next, ("state",)),),
        retained_fields=(("previous",),), time=("time",), step=("step",))
    base = p.continuation_open(initial, schema)
    entry = p.continuation_read(base)
    value = p.value("ab2_growth", entry["state"]+p.dt*(1.5*entry["state"]-.5*entry["previous"]),
                    at=q.next.point)
    proposal = p.product(h=p.requested_dt())
    trial = p.product(state=value, previous=entry["state"], time=entry["time"]+proposal["h"],
                      step=entry["step"]+p.int64(1), counter=entry["counter"]+p.int64(1),
                      controls=entry["controls"])
    approved = _approve(p, base, proposal, trial, p.product(), advancing=True)

    def synchronize(b, base, proposal):
        entry, candidate = b.continuation_read(base), proposal["candidate"]
        result = b.product(state=proposal["synchronized"]["field_0"], previous=candidate["previous"],
            time=proposal["interval"]["end"], step=entry["step"]+b.int64(1),
            counter=candidate["counter"], controls=candidate["controls"])
        return _approve(b, base, proposal, result, b.product(counter=candidate["counter"]), advancing=True)

    def initialize(b):
        mass = b.composite_integral("ab2_initial_mass", b.attempt_start(q.n)[0], over=q.n)
        initial = b.product(time=b.float64(0), step=b.int64(0), counter=b.int64(0),
                            mass=mass, regrids=b.int64(0))
        return initial, ContinuationSchema(initial, time=("time",), step=("step",))

    def observe(b):
        return b.product(mass=b.composite_integral("ab2_composite_mass", q.n[0], over=q.n))

    def parent(b, base, proposal):
        entry = b.continuation_read(base)
        result = b.product(time=proposal["interval"]["end"], step=entry["step"]+b.int64(1),
            counter=entry["counter"]+b.int64(1), mass=proposal["observations"]["mass"],
            regrids=entry["regrids"])
        return _approve(b, base, proposal, result, b.product(), advancing=True)

    def remap(b, proposal):
        algebra = Distributed(b, storage_bytes=64*1024*1024, storage_versions=4096)
        source_level = b.int64(0)  # Explicit equation for this profile, not an engine default.
        base = b.regrid_base(source_level)
        old = b.continuation_read(base)
        source = b.regrid_source(source_level)
        template = proposal["templates"]["field_0"]

        def move(field, output_template, family=None):
            old_view = b.regrid_topology(field, level=source_level, side="source", dimension=2)
            exported = b.field_bundle(field, topology=old_view, cell_capacity=16384,
                                       patch_capacity=256, family=family)
            new_view = b.regrid_topology(output_template, level=proposal["level"],
                                        side="target", dimension=2)
            # Geometry comes from target storage.  An old retained Field is only
            # an import type/provenance template; it is never read as new storage.
            target = (b.field_bundle(output_template, topology=new_view, cell_capacity=16384,
                                     patch_capacity=256, family=exported)
                      if family is None else family)
            transferred = _injection(b, algebra, exported, target,
                                     target_level=proposal["level"], cells=cells)
            return b.bundle_field(transferred, template=output_template, topology=new_view), transferred

        current, bundle = move(source["state"], template)
        previous, _ = move(source["previous"], source["previous"], family=bundle)
        result = b.product(state=current, previous=previous, time=old["time"], step=old["step"],
                           counter=old["counter"], controls=old["controls"])
        return _approve(b, base, proposal, result,
            b.product(mass=_bundle_mass(b, algebra, bundle)), advancing=False)

    def remap_parent(b, base, proposal):
        entry = b.continuation_read(base)
        mass = b.reduce(proposal["children"], 0.*entry["mass"],
                        lambda z, total, child: total+child["mass"], order="ascending")
        result = b.product(time=entry["time"], step=entry["step"], counter=entry["counter"],
                            mass=mass, regrids=entry["regrids"]+b.int64(1))
        return _approve(b, base, proposal, result, b.product(), advancing=False,
                        mass=True, reject=reject_regrid_parent)

    p.continuation_forest(approved, synchronize=synchronize, initialize=initialize,
        parent=parent, observe=observe, level_capacity=8, regrid=remap,
        regrid_parent=remap_parent, dimension=2)
    p.step_strategy(FixedDt(dt))
    case.program(p)
    transfer = AMRTransfer()
    transfer.state(block[state], StateTransfer())
    threshold = case.param(RuntimeParam("ab2_regrid_threshold", default=1.55))
    frame = model.frame
    layout = AMR(grid=CartesianGrid(frame=frame, cells=cells, periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(pm.ValueExpr(block[state]) > case.value(threshold)), Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid(schedule=every(2, clock=p.clock)), transfer=transfer,
        execution=AMRExecution.synchronous())

    def check(runtime):
        from .method_amr_reception import composite_base_projection
        arrays = [runtime.block_level_state_global("method", i) for i in range(runtime.n_levels())]
        projection = composite_base_projection(arrays, runtime.patch_boxes(), cells=cells)
        if projection is None:
            return non_owner_metrics()
        step = runtime.macro_step()
        expected = ab2_factors(dt, step)[-1]*data[0]
        error = float(np.max(abs(projection["means"]-expected)))
        return dict(passed=error < 5.e-13, metrics_owner=True, reference_error=error,
                    levels=runtime.n_levels(), step=step)

    return CaseSpec(case, layout, _initial(case, block[state], data), {"method": block[state]},
        steps*dt, dt, check, "AB2 consumes a retained Field after a fully assessed Python AMR remap")
