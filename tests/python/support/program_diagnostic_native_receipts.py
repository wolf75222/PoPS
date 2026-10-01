"""Real-owner diagnostic checkpoint fixture; no native or MPI substitutes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pops

from pops.analytic import cos, x
from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
                      Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR, Uniform
from pops.lib.amr import BergerRigoutsos, StateTransfer
from pops.lib.initial import Analytic
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt, every
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.support.amr_snapshots import composite_active_mask
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.evidence_json import ENCODING, evidence_dumps
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context

DT, CELLS = .01, 16
STATE_KEY, OFFSET_KEY = "program_diagnostics_state", "program_diagnostics_offsets"
RAW_BITS = {"": 0x3FE0000000000000, "raw\0unicode-π": 0x0000000000000001,
            "raw-negative-zero": 0x8000000000000000, "raw-nan-payload": 0x7FF8000000001234,
            "raw-positive-infinity": 0x7FF0000000000000}


def rank(world):
    return 0 if world is None else int(world.rank)


def size(world):
    return 1 if world is None else int(world.size)


def build(amr):
    frame = Rectangle("diagnostic-archive-box", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("diagnostic-carrier", frame=frame)
    state = model.state("U", components=("first", "second"))
    source = model.source("decay", on=state, value=(-state[0], .5 * state[1]))
    flux = model.flux("stationary", frame=frame, state=state,
        components={axis: tuple(0 * q for q in state) for axis in frame.axes},
        waves={axis: tuple(0 * q for q in state) for axis in frame.axes})
    rate = model.rate("balance", equation=ddt(state) == -div(flux) + source)
    case = pops.Case("diagnostic-checkpoint-" + ("amr" if amr else "uniform"))
    block = case.block("fluid", model)
    numerical = DiscretizationPlan()
    numerical.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case.numerics(numerical, block=block)
    program = pops.Program("diagnostic-accepted-map")
    current = program.state(block[state])
    program.store_history("accepted-fluid", current.n, depth=2)
    rhs = program.source(model.module.operator_handle("decay"), current.n)
    candidate = program.value("updated", current.n + program.dt * rhs, at=current.next.point)
    program.record_scalar("global-energy", program.dot_all(current.n, current.n))
    program.commit(current.next, candidate)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    profile = 1 + .04 * cos(2 * np.pi * x(frame))
    case.initials.add(InitialCondition(state=block[state], value=Analytic(frame=frame,
        components=(profile, 0 * profile + .7)), projection=ConservativeCellAverage()))
    grid = CartesianGrid(frame=frame, cells=(CELLS, CELLS), periodic=PeriodicAxes(frame.axes))
    if not amr:
        return case, Uniform(grid)
    transfer = AMRTransfer()
    transfer.state(block[state], StateTransfer())
    threshold = case.param(RuntimeParam("diagnostic-refinement-threshold", default=1.035))
    layout = AMR(grid=grid, hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(ValueExpr(block[state])["first"] > case.value(threshold)), Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid(schedule=every(1000, clock=program.clock)), transfer=transfer,
        execution=AMRExecution.synchronous(), clustering=BergerRigoutsos(maximum_box_size=8))
    return case, layout


def prepare(world, tmp_path, amr):
    directory = collective_directory(world, tmp_path / ("diagnostics-amr" if amr else "diagnostics-uniform"))
    case, layout = collective_call(world, lambda: build(amr))
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = (compile_resolved_plan_once(world, resolved, route="accepted diagnostic map",
                 compile_artifact=pops.compile) if world is not None
                else collective_call(world, lambda: pops.compile(resolved)))
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    return artifact, context, directory


def bind(world, artifact, context):
    return collective_call(world, lambda: pops.bind(artifact, resources={"execution_context": context}))


def decoded(image):
    """Independent reader compares IEEE bits, never NaN float equality."""
    assert image[:8] == b"POPSDIA1" and len(image) >= 40
    width, owner, ranks, count = struct.unpack_from("<QQQQ", image, 8)
    assert width == 64, "this fixture requires the authenticated float64 native provider"
    position, result = 40, {}
    for _ in range(count):
        length, = struct.unpack_from("<Q", image, position)
        position += 8
        name = image[position:position + length].decode("utf-8")
        position += length
        bits, = struct.unpack_from("<Q", image, position)
        position += 8
        assert name not in result
        result[name] = bits
    assert position == len(image)
    return owner, ranks, result


def record_raw(world, runtime):
    native = runtime._executor._s
    for name, bits in RAW_BITS.items():
        value, = struct.unpack("<d", struct.pack("<Q", bits))
        collective_call(world, lambda name=name, value=value: native.record_program_diagnostic(name, value))
    collective_call(world, lambda: native.record_program_diagnostic("rank-local", rank(world) + .125))
    data = collective_call(world, native._checkpoint_program_diagnostics)
    with collective_check(world):
        owner, ranks, values = decoded(data)
        assert (owner, ranks) == (rank(world), size(world))
        assert all(values[name] == bits for name, bits in RAW_BITS.items())
        assert values["rank-local"] == struct.unpack("<Q", struct.pack("<d", rank(world) + .125))[0]


def snapshot_geometry(world, runtime, amr):
    if amr:
        return collective_call(world, lambda: tuple(runtime.patch_boxes()))
    shape = collective_call(world, runtime.spatial_shape)
    boxes = collective_call(world, lambda: runtime.local_boxes("fluid"))
    return ("uniform", tuple(shape), tuple(boxes))


def snapshot(world, runtime, amr):
    native = runtime._executor._s
    levels = collective_call(world, runtime.n_levels) if amr else 1
    arrays, history = {}, []
    for level in range(levels):
        value = collective_call(world, (lambda level=level: runtime.block_level_state_global("fluid", level))
                                if amr else lambda: runtime.state_global("fluid"))
        with collective_check(world):
            arrays["level%d_state" % level] = np.asarray(value).copy()
        if amr:
            mask = collective_call(world, lambda level=level: composite_active_mask(runtime, level, refinement_ratio=2))
            with collective_check(world):
                arrays["level%d_active" % level] = np.asarray(mask).copy()
    for name in collective_call(world, runtime.history_names):
        depth = collective_call(world, lambda name=name: runtime.history_depth(name))
        width = collective_call(world, lambda name=name: runtime.history_ncomp(name))
        for level in range(levels):
            args = (name, level) if amr else (name,)
            initialized = collective_call(world, lambda args=args: native.history_initialized(*args))
            fill = collective_call(world, lambda args=args: native.history_fill_count(*args))
            identity = collective_call(world, lambda args=args: bytes(native.history_sample_identity(*args)))
            history.append((name, level, width, depth, initialized, fill, identity.hex()))
            for slot in range(depth):
                args = (name, level, slot) if amr else (name, slot)
                duration = collective_call(world, lambda args=args: native.history_slot_dt(*args))
                value = collective_call(world, lambda args=args: native.history_global(*args))
                with collective_check(world):
                    arrays["history_%s_l%d_s%d" % (name, level, slot)] = np.asarray(value).copy()
                    history.append((name, level, slot, float(duration).hex()))
    diagnostics = collective_call(world, native._checkpoint_program_diagnostics)
    geometry = snapshot_geometry(world, runtime, amr)
    clock = collective_call(world, lambda: (float(runtime.time()).hex(), runtime.macro_step()))
    lifecycle = (*clock, geometry)
    return {"arrays": arrays, "history": tuple(history), "diagnostics": diagnostics, "lifecycle": lifecycle}


def same_images(world, left, right, *, diagnostics=True):
    with collective_check(world):
        assert left["lifecycle"] == right["lifecycle"]
        assert left["history"] == right["history"]
        assert set(left["arrays"]) == set(right["arrays"])
        for name in left["arrays"]:
            a, b = left["arrays"][name], right["arrays"][name]
            assert (a.shape, a.dtype, a.tobytes()) == (b.shape, b.dtype, b.tobytes()), name
        if diagnostics:
            assert left["diagnostics"] == right["diagnostics"]


def save(world, runtime, artifact, directory, phase, image, *, checkpoint=None):
    """Actual per-rank data/retained source; receipts are not an external owner seal."""
    from pops._native_selector import selected_native_module

    native = selected_native_module(required=True)
    with collective_check(world):
        target = directory / ("rank%d" % rank(world))
        target.mkdir(exist_ok=True)
        path = target / (phase + ".npz")
        np.savez_compressed(path, **image["arrays"], diagnostics=np.frombuffer(image["diagnostics"], dtype=np.uint8))
        cpp = artifact.program._generated_cpp
        retained = None
        if cpp is not None:
            source = target / "program-retained.cpp"
            source.write_text(cpp)
            retained = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        fixture = Path(__file__).resolve().parents[1] / "integration/runtime/test_program_diagnostic_checkpoint_runtime.py"
        receipt = {"contract": "sol61.program-diagnostics.native-fixture@1", "phase": phase,
            "metadata_encoding": ENCODING,
            "rank": rank(world), "ranks": size(world), "native_dimension": int(native.__native_dimension__),
            "native_path": str(Path(native.__file__).resolve()),
            "native_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
            "native_abi": str(native.abi_key()), "python_package": str(Path(pops.__file__).resolve()),
            "native_capabilities": dict(native.module_capabilities("module")),
            "platform": artifact.platform_manifest.to_data(),
            "compiled_components": artifact._current_component_evidence(),
            "compiled_plan": artifact.plan._payload(),
            "artifact": artifact.artifact_identity.token, "bind": runtime.bind_identity.token,
            "lifecycle": image["lifecycle"], "history": image["history"],
            "diagnostic_header_and_bits": decoded(image["diagnostics"]),
            "saved_path": str(path), "saved_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "retained_cpp": retained, "checkpoint": checkpoint,
            "fixture_sources": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (Path(__file__).resolve(), fixture)}}
        if checkpoint is not None:
            receipt["checkpoint_sha256"] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        (target / (phase + "-receipt.json")).write_text(evidence_dumps(receipt))


def resealed_fault(world, runtime, checkpoint, directory, attack):
    """Harness reseal only; preserves the real run/consumer identities and all other arrays."""
    from pops.runtime._checkpoint_manifest import IDENTITY_KEY, MANIFEST_KEY, seal_checkpoint_payload

    path = directory / (attack + ".npz")
    with collective_check(world):
        if rank(world) == 0:
            with np.load(checkpoint, allow_pickle=False) as archive:
                family = json.loads(str(archive[MANIFEST_KEY]))["runtime_kind"]
                payload = {name: archive[name].copy() for name in archive.files if name not in (IDENTITY_KEY, MANIFEST_KEY)}
            if attack == "legacy-absence":
                del payload[STATE_KEY], payload[OFFSET_KEY]
            elif attack == "offset":
                payload[OFFSET_KEY][1] += 1
            elif attack == "rank-authority":
                # In MPI this duplicates rank zero's authority in the rank-one header.
                start = int(payload[OFFSET_KEY][1]) if size(world) > 1 else 0
                owner = 0 if size(world) > 1 else 1
                payload[STATE_KEY][start + 16:start + 24] = np.frombuffer(owner.to_bytes(8, "little"), dtype=np.uint8)
            elif attack == "rank-order":
                assert size(world) > 1
                raw, offsets = payload[STATE_KEY], payload[OFFSET_KEY]
                chunks = [raw[int(a):int(b)].copy() for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
                chunks.reverse()
                payload[STATE_KEY] = np.concatenate(chunks)
                payload[OFFSET_KEY] = np.array([0, *np.cumsum([len(chunk) for chunk in chunks])], dtype=np.int64)
            elif attack == "native-body":
                # Fixed header remains valid; impossible record count reaches the real native reader.
                payload[STATE_KEY][32:40] = 255
            else:
                raise AssertionError(attack)
            seal_checkpoint_payload(runtime, payload, runtime_kind=family)
            with path.open("wb") as stream:
                np.savez_compressed(stream, **payload)
    return path
