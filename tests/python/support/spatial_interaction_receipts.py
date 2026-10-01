"""Public spatial-map fixture and capture of actual native owner/EB pieces.

This helper creates no ROOT approvals and does not replace the independent reader.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np
import pops

from tests.python.support.collective_checks import collective_call, collective_check

CELLS = (8, 6)
LOWER, UPPER = (.3, -.4), (2.3, 2.6)
DT, GAMMA, WORKSPACE = .01, .2, 8 << 20


def build(width, *, adaptive=False, failure=None):
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian2D
    from pops.mesh import CartesianGrid
    from pops.mesh.geometry import Disc, EmbeddedBoundary
    from pops.mesh.masks import CutCell
    from pops.boundary import ZeroFlux
    from pops.layouts import Uniform
    from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
    from pops.math import ddt, div
    from pops._ir.expr import Const
    from pops.numerics.terms import Flux, DefaultSource
    from pops.time import Program, FixedDt
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.projection import ConservativeCellAverage
    from pops.fields import SpatialInteractionKernel, CellVolumeMeasure, CellMidpoint, DirectSpatialInteraction
    from pops.model.spaces import FieldSpace

    if width not in (1, 3):
        raise ValueError("this native witness explicitly declares width one or three")
    frame = CartesianDomain("physical", lower=LOWER, upper=UPPER).frame(Cartesian2D())
    model = pops.Model("nonlocal_density", frame=frame)
    labels = ("north",) if width == 1 else ("north", "east", "third")
    rho = model.state("rho", components=labels, sampling="cell_average")
    flux = model.flux("zero_flux", frame=frame, state=rho,
        components={axis: tuple(0*rho[c] for c in range(width)) for axis in frame.axes},
        waves={axis: (Const(0),)*width for axis in frame.axes})
    reaction = model.source("invalid_input" if failure == "nonfinite" else "decay", on=rho,
        value=tuple(rho[c]/(3-rho[c]) if failure == "nonfinite" else -GAMMA*rho[c] for c in range(width)))
    rate = model.rate("evolve", equation=ddt(rho) == -div(flux)+reaction)
    case = pops.Case("actual_spatial_interaction")
    block = case.block("density", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(rho),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case.numerics(numerics, block=block)
    program = Program("actual_direct_interaction")
    u = program.state(block[rho])
    program.keep_history(u, depth=1)
    selected = (0,) if width == 1 else (2, 0)
    output = FieldSpace("interaction", components=tuple(labels[c] for c in selected),
        frame=u.n.space.frame, support=u.n.space.support, sampling="cell_center")
    kernel = SpatialInteractionKernel(2, (lambda x, y: 1/(x[0]-y[0])) if failure == "pole"
        else lambda x, y: 1+x[0]*y[1]-2*y[0])
    realization = DirectSpatialInteraction(1 if failure == "budget" else WORKSPACE)
    accepted_source = u.n
    accepted_scope = "accepted"
    if failure == "nonfinite":
        # The IC stays finite and identical on every rank. A single actual cell
        # with rho=3 produces nonfinite issued data on its real resident ranks.
        accepted_source = program.value("invalid_issued",
            u.n+program.dt*program.rhs(state=u.n, terms=[DefaultSource()]), at=u.next.point)
        accepted_scope = "issued"
    for name, source, scope in (("I_accepted", accepted_source, accepted_scope), ("I_history", u.prev, "issued")):
        field = program.spatial_interaction(source, kernel, output_space=output, components=selected,
            measure=CellVolumeMeasure(), quadrature=CellMidpoint(), realization=realization, source_scope=scope)
        program.store_history(name, field, depth=1)
    candidate = program.value("decayed", u.n+program.dt*program.rhs(state=u.n, terms=[Flux(), DefaultSource()]),
                              at=u.next.point)
    program.commit(u.next, candidate)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    case.initials.add(InitialCondition(state=block[rho], value=BindArray(), projection=ConservativeCellAverage()))
    grid = CartesianGrid(frame=frame, cells=CELLS)
    embedded = EmbeddedBoundary(Disc(center=(1.3, 1.1), radius=1.1), CutCell(), ZeroFlux())
    layout = Uniform(grid, embedded_boundary=embedded)
    if adaptive:
        from pops.layouts import AMR
        from pops.amr import AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer, Tag, Hysteresis, EqualityPolicy, ConflictPolicy, Buffer
        from pops.lib.amr import BergerRigoutsos, StateTransfer
        from pops.math import ValueExpr
        from pops.params import RuntimeParam
        from pops.time import every
        threshold = case.param(RuntimeParam("refine", default=1.08))
        transfer = AMRTransfer()
        transfer.state(block[rho], StateTransfer())
        layout = AMR(grid=grid, embedded_boundary=embedded,
            hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
            tagging=AMRTagging(rules=(Tag(ValueExpr(block[rho])[labels[0]] > case.value(threshold)), Buffer(cells=0)),
                hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
            regrid=AMRRegrid(schedule=every(1000, clock=program.clock)), transfer=transfer,
            execution=AMRExecution.synchronous(), clustering=BergerRigoutsos(maximum_box_size=8))
    return case, layout, selected, program


def initial_values(width, *, failure=None):
    x, y = np.meshgrid((np.arange(CELLS[0])+.5)/CELLS[0], (np.arange(CELLS[1])+.5)/CELLS[1])
    data = np.stack((1-.15*np.cos(2*np.pi*y)+.025*np.sin(2*np.pi*x), -.4+.2*x+.1*np.sin(2*np.pi*y),
                     .2-.3*np.sin(2*np.pi*x)-y))
    result = data[:width].copy()
    if failure == "nonfinite":
        # This finite value is bound to the declared BindArray initial plan;
        # its declared source alone creates nonfinite issued candidate data.
        result[0, CELLS[1]//2, CELLS[0]//2] = 3.
    return result


def bound_initial_values(initial_plan, width, *, failure=None):
    """The resolved InitialConditionPlan is the single initialization authority."""
    if initial_plan is None or len(initial_plan.bindings) != 1:
        raise ValueError("spatial witness requires exactly one declared initial subject")
    subject = initial_plan.bindings[0].subject
    if initial_plan.canonical_subject(subject) is not subject:
        raise ValueError("spatial initial subject is not the declared canonical Handle")
    return {subject: np.ascontiguousarray(initial_values(width, failure=failure))}


def digest(path):
    path = Path(path)
    with path.open("rb", buffering=0) as stream:
        before = os.fstat(stream.fileno())
        remaining = before.st_size
        checksum = hashlib.sha256()
        while remaining:
            chunk = stream.read(min(remaining, 1 << 20))
            if not chunk:
                raise ValueError("evidence file truncated during checksum")
            remaining -= len(chunk)
            checksum.update(chunk)
        after = os.fstat(stream.fileno())
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in fields):
            raise ValueError("evidence file changed during checksum")
        return checksum.hexdigest()



_WIRE_CONTRACT = "pops.spatial-interaction-fixture-array-wire@1"


def _array_dtype(value):
    if type(value) is not str or re.fullmatch(r"(?:[<>][iu][248]|[<>]f[48]|\|[biu]1)", value) is None:
        raise ValueError("array wire dtype must be an exact canonical numeric string")
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as error:
        raise ValueError("array wire dtype is invalid") from error
    if dtype.str != value or dtype.fields is not None or dtype.subdtype is not None \
            or dtype.kind not in "biuf" or dtype.itemsize not in (1, 2, 4, 8) \
            or (dtype.kind == "f" and dtype.itemsize not in (4, 8)):
        raise ValueError("array wire dtype must be a canonical plain numeric dtype")
    return dtype


def _hex_bytes(value, *, expected=None):
    if type(value) is not str or len(value) % 2 or re.fullmatch("[0-9a-f]*", value) is None:
        raise ValueError("array wire Cbytes must be exact lowercase hexadecimal")
    if len(value)//2 > sys.maxsize or (expected is not None and len(value)//2 != expected):
        raise ValueError("array wire byte length differs from its bounded shape")
    return bytes.fromhex(value)


def _wire_node(value, *, decode, depth=0):
    if depth > 64:
        raise ValueError("array wire nesting is too deep")
    def visit(item):
        return _wire_node(item, decode=decode, depth=depth+1)
    if not decode:
        if type(value) is np.ndarray:
            dtype = _array_dtype(value.dtype.str)
            return ["array", {"dtype": dtype.str, "shape": list(value.shape),
                              "Cbytes": value.tobytes(order="C").hex()}]
        if type(value) is bytes:
            return ["bytes", value.hex()]
        if type(value) is float:
            return ["float", struct.pack(">d", value).hex()]
        if value is None or type(value) in (str, int, bool):
            return ["scalar", value]
        if type(value) in (tuple, list):
            return ["tuple" if type(value) is tuple else "list", [visit(item) for item in value]]
        if type(value) is dict and all(type(key) is str and key for key in value):
            return ["map", [[key, visit(item)] for key, item in value.items()]]
        raise TypeError("array wire refuses unsupported values")
    if type(value) is not list or len(value) != 2 or type(value[0]) is not str:
        raise ValueError("array wire node is invalid")
    tag, payload = value
    if tag == "scalar" and (payload is None or type(payload) in (str, int, bool)):
        return payload
    if tag == "float":
        return struct.unpack(">d", _hex_bytes(payload, expected=8))[0]
    if tag == "bytes":
        return _hex_bytes(payload)
    if tag in ("tuple", "list") and type(payload) is list:
        result = [visit(item) for item in payload]
        return tuple(result) if tag == "tuple" else result
    if tag == "map" and type(payload) is list:
        result = {}
        for row in payload:
            if type(row) is not list or len(row) != 2 or type(row[0]) is not str \
                    or not row[0] or row[0] in result:
                raise ValueError("array wire mapping keys must be unique exact strings")
            result[row[0]] = visit(row[1])
        return result
    if tag == "array" and type(payload) is dict and set(payload) == {"dtype", "shape", "Cbytes"}:
        dtype = _array_dtype(payload["dtype"])
        shape = payload["shape"]
        if type(shape) is not list or len(shape) > 32 or any(
                type(size) is not int or size < 0 or size > sys.maxsize for size in shape):
            raise ValueError("array wire shape requires bounded nonnegative exact integers")
        count, strides = 1, 1
        for size in shape:
            # Even a zero-size axis cannot hide unrepresentable NumPy strides.
            if strides > sys.maxsize // dtype.itemsize // max(size, 1):
                raise ValueError("array wire shape overflows native byte arithmetic")
            strides *= max(size, 1)
            count *= size
        data = _hex_bytes(payload["Cbytes"], expected=count*dtype.itemsize)
        return np.frombuffer(data, dtype=dtype).reshape(tuple(shape)).copy(order="C")
    raise ValueError("array wire tag or payload is invalid")


def array_wire_encode(value):
    return {"contract": _WIRE_CONTRACT, "value": _wire_node(value, decode=False)}


def array_wire_decode(value):
    if type(value) is not dict or set(value) != {"contract", "value"} \
            or value["contract"] != _WIRE_CONTRACT:
        raise ValueError("array wire envelope differs from its versioned contract")
    return _wire_node(value["value"], decode=True)


def _allgather_array_values(world, value):
    from pops._native_collectives import allgather_value
    # A local encoding error must vote before peers enter the payload collective.
    encoded = collective_call(world, lambda: array_wire_encode(value))
    gathered = collective_call(world, lambda: tuple(allgather_value(world, encoded)))
    return collective_call(world, lambda: tuple(array_wire_decode(copy) for copy in gathered))

def capture(world, runtime, width, *, adaptive, step):
    """Preserve replicas as replicas; rank zero writing does not assign ownership."""
    executor = runtime._executor
    count = collective_call(world, lambda: runtime.n_levels()) if adaptive else 1
    epoch = collective_call(world, lambda: executor.checkpoint_topology_epoch()) if adaptive else 0
    levels = []
    for level in range(count):
        scale = 2**level
        shape = (CELLS[0]*scale, CELLS[1]*scale)
        spacing = tuple((UPPER[a]-LOWER[a])/shape[a] for a in range(2))
        geometry = collective_call(world, lambda level=level, spacing=spacing, shape=shape: executor._output_geometry_snapshot(
            level, LOWER, spacing, shape, (2, 2) if level+1 < count else (0, 0), "pops://cell-measures/cartesian-area@1")) if adaptive else collective_call(
                world, lambda spacing=spacing, shape=shape: executor._output_geometry_snapshot(LOWER, spacing, shape, "pops://cell-measures/cartesian-area@1"))
        pieces = {}
        pieces["rho"] = collective_call(world, lambda level=level: executor.output_state_local_pieces("density", level))
        for key, native_name in (("active", "pops_active"), ("kappa", "pops_kappa")):
            pieces[key] = collective_call(world, lambda native_name=native_name, level=level:
                executor.output_embedded_boundary_local_pieces(native_name, level))
        history_values, history_meta = {}, {}
        if step:
            # Recover the keeper from the actual Native inventory; no name inference.
            names = collective_call(world, lambda: tuple(executor.history_names()))
            with collective_check(world):
                rho_names = tuple(name for name in names if name not in ("I_accepted", "I_history"))
                assert len(rho_names) == 1, names
                rho_name = rho_names[0]
            history_meta["rho_name"] = rho_name
            for name in (rho_name, "I_accepted", "I_history"):
                depth = collective_call(world, lambda name=name: runtime.history_depth(name))
                sample = collective_call(world, lambda name=name, level=level: bytes(executor.history_sample_identity(name, level)
                    if adaptive else executor.history_sample_identity(name)))
                filled = collective_call(world, lambda name=name, level=level: executor.history_fill_count(name, level)
                    if adaptive else executor.history_fill_count(name))
                initialized = collective_call(world, lambda name=name, level=level: executor.history_initialized(name, level)
                    if adaptive else executor.history_initialized(name))
                durations = []
                canonical_name = "rho" if name == rho_name else name
                for slot in range(depth):
                    duration = collective_call(world, lambda name=name, slot=slot, level=level: runtime.history_slot_dt(name, level, slot)
                        if adaptive else runtime.history_slot_dt(name, slot))
                    durations.append(duration)
                    values = collective_call(world, lambda name=name, level=level, slot=slot: runtime.history_global(name, level, slot)
                        if adaptive else runtime.history_global(name, slot))
                    with collective_check(world):
                        value = np.asarray(values).reshape(-1, shape[1], shape[0]).copy()
                        history_values[canonical_name+"_slot_%d" % slot] = value
                        if slot == 1:
                            history_values["rho_retained" if name == rho_name else name] = value.copy()
                history_meta[canonical_name] = dict(native_name=name, sample_hex=sample.hex(),
                    depth=depth, fill_count=filled, initialized=initialized, slot_dt=durations)
        local = {"pieces": pieces, "history_values": history_values, "history_meta": history_meta}
        copies = _allgather_array_values(world, local)
        with collective_check(world):
            levels.append({"level": level, "shape": shape, "origin": LOWER, "spacing": spacing,
                           "geometry": geometry, "copies": copies})
    clock = collective_call(world, lambda: (runtime.time(), runtime.macro_step()))
    auxiliary = collective_call(world, lambda: tuple(bytes(part) for part in executor.capture_auxiliary_checkpoint_accepted_state())
        if adaptive else (bytes(executor.capture_auxiliary_checkpoint_accepted_state()),))
    auxiliary = _allgather_array_values(world, auxiliary)
    return {"levels": levels, "clock": clock, "auxiliary": auxiliary, "epoch": epoch}


def arrays(image):
    """Store dense Native buffers and each local piece, with its exact owner metadata."""
    result = {"time": np.asarray(image["clock"][0]), "step": np.asarray(image["clock"][1]),
              "topology_epoch": np.asarray(image["epoch"])}
    for rank, parts in enumerate(image["auxiliary"]):
        for ordinal, part in enumerate(parts):
            result["rank_%d_auxiliary_%d" % (rank, ordinal)] = np.frombuffer(part, dtype=np.uint8)
    manifest = []
    for row in image["levels"]:
        prefix = "level_%d" % row["level"]
        for key in ("coverage", "valid_cells", "cell_volumes"):
            result[prefix+"_"+key] = np.asarray(row["geometry"][key]).copy()
        result[prefix+"_boxes"] = np.asarray(row["geometry"]["boxes"], dtype=np.int64)
        result[prefix+"_native_cell_shape"] = np.asarray(row["geometry"]["cell_shape"], dtype=np.int64)
        result[prefix+"_origin"] = np.asarray(row["origin"])
        result[prefix+"_spacing"] = np.asarray(row["spacing"])
        for rank, copy in enumerate(row["copies"]):
            rank_prefix = prefix+"_rank_%d" % rank
            for field, pieces in copy["pieces"].items():
                for ordinal, piece in enumerate(pieces):
                    name = rank_prefix+"_"+field+"_piece_%d" % ordinal
                    result[name] = np.asarray(piece["values"]).copy()
                    manifest.append({"array": name, "level": row["level"], "resident_rank": rank,
                        "lower": piece["lower"], "upper": piece["upper"], "field": field,
                        "global_box_index": piece["global_box_index"], "reported_owner": piece["owner_rank"],
                        "replicated": piece["replicated"],
                        "physical_owner": 0 if piece["replicated"] else piece["owner_rank"]})
            for name, values in copy["history_values"].items():
                result[rank_prefix+"_"+name] = values
            result[rank_prefix+"_history_metadata_json"] = np.frombuffer(
                json.dumps(copy["history_meta"], sort_keys=True).encode(), dtype=np.uint8)
    result["piece_manifest_json"] = np.frombuffer(json.dumps(manifest, sort_keys=True).encode(), dtype=np.uint8)
    return result


def same(left, right):
    a, b = arrays(left), arrays(right)
    assert a.keys() == b.keys()
    for key in a:
        assert a[key].shape == b[key].shape and a[key].dtype == b[key].dtype and a[key].tobytes() == b[key].tobytes(), key
