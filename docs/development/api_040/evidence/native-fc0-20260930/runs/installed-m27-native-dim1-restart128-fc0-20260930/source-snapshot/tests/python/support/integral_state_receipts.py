"""Independent readers/oracle for the public IntegralState reception (no native hooks)."""
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from tests.python.support.collective_checks import collective_call, collective_check


def decode_exchange_image(image):
    data = memoryview(image)
    assert bytes(data[:8]) == b"POPSEX02", "integral receipt requires POPSEX02"
    position = 8

    def word():
        nonlocal position
        value, = struct.unpack_from("<Q", data, position)
        position += 8
        return value

    def text():
        nonlocal position
        length = word()
        assert position + length <= len(data)
        value = bytes(data[position:position + length]).decode("utf-8")
        position += length
        return value

    def number():
        return struct.unpack("<d", struct.pack("<Q", word()))[0]

    records = []
    for _ in range(word()):
        operation, occurrence, context, quadrature = (text() for _ in range(4))
        orientation = word()
        if orientation >= 1 << 63:
            orientation -= 1 << 64
        measure, flux, weight = number(), number(), number()
        multiplicity = word()
        axis, side, component = (word() - 1 for _ in range(3))
        exterior, evaluation = word(), text()
        records.append({"operation": operation, "occurrence": occurrence,
                        "context": context, "quadrature": quadrature,
                        "orientation": orientation, "measure": measure, "flux": flux,
                        "weight": weight, "multiplicity": multiplicity, "axis": axis,
                        "side": side, "component": component, "exterior": bool(exterior),
                        "evaluation": evaluation})
    quantities = {}
    for _ in range(word()):
        name, initial, value = text(), number(), number()
        assert name not in quantities
        quantities[name] = (initial, value)
    consumed = [tuple(text() for _ in range(4)) for _ in range(word())]
    assert position == len(data), "reader did not consume the exact exchange image"
    return {"records": records, "quantities": quantities, "consumed": consumed}


def checkpoint_exchange_images(path):
    with np.load(Path(path), allow_pickle=False) as archive:
        raw = archive["program_exchange_state"].tobytes()
        offsets = archive["program_exchange_offsets"]
        assert offsets[0] == 0 and offsets[-1] == len(raw)
        assert np.all(np.diff(offsets) > 0)
        images = tuple(raw[int(a):int(b)] for a, b in zip(offsets[:-1], offsets[1:], strict=True))
    return images, tuple(decode_exchange_image(image) for image in images)


def selected_external_records(ledger, *, axis=0, side=1, component=0):
    return [record for record in ledger["records"] if record["exterior"]
            and (record["axis"], record["side"], record["component"]) == (axis, side, component)]


def ssprk_upwind_step(density, dt):
    """One-dimensional upwind incidence with outflow ghost averages, exact in y."""
    density = np.asarray(density, dtype=np.float64)
    n = density.size

    def rhs(value):
        # a=1: the right-oriented numerical face flux is its left cell value.
        lower = np.concatenate((value[:1], value[:-1]))
        return -n * (value - lower)

    predictor = density + dt * rhs(density)
    accepted = .5 * density + .5 * predictor + .5 * dt * rhs(predictor)
    increments = (.5 * dt * density[-1], .5 * dt * predictor[-1])
    return accepted, increments


def _root(world):
    return world is None or int(world.rank) == 0


def collective_directory(world, tmp_path):
    if world is not None:
        from pops._native_collectives import broadcast_value
        tmp_path = Path(broadcast_value(world, str(tmp_path) if _root(world) else None, root=0))
    with collective_check(world):
        if _root(world):
            tmp_path.mkdir(parents=True, exist_ok=True)
    return tmp_path


def save_public_snapshot(world, runtime, artifact, quantity, directory, phase, *, amr=False):
    levels = collective_call(world, runtime.n_levels) if amr else 1
    if amr and world is not None:
        from pops._native_collectives import allgather_value
        counts = allgather_value(world, levels)
        with collective_check(world):
            assert len(set(counts)) == 1, counts
    gathered_levels = []
    for level in range(levels):
        gathered = collective_call(world, (lambda level=level: runtime.block_level_state_global("fluid", level))
                                   if amr else lambda: runtime.state_global("fluid"))
        with collective_check(world):
            gathered_levels.append(np.asarray(gathered).copy() if _root(world) else None)
    with collective_check(world):
        scalar = runtime.integral_state(quantity)
        time, step = runtime.time(), runtime.macro_step()
    checkpoint = collective_call(world, lambda: runtime.checkpoint(directory / phase))
    result = None
    with collective_check(world):
        if _root(world):
            state = gathered_levels[0]
            images, ledgers = checkpoint_exchange_images(checkpoint)
            receipt = {"phase": phase, "artifact": artifact.artifact_identity.token,
                "platform": artifact.platform_manifest.to_data(), "dimension": artifact.resolved_dimension,
                "quantity_identity": quantity.identity, "quantity": scalar,
                "time": time, "macro_step": step, "checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                "ledger_sha256": [hashlib.sha256(image).hexdigest() for image in images],
                "ledgers": ledgers}
            np.savez(directory / (phase + "-state.npz"), state=state, q=scalar, time=time, step=step,
                     **{"level_%d" % level: value for level, value in enumerate(gathered_levels)})
            (directory / (phase + "-receipt.json")).write_text(json.dumps(receipt, indent=2) + "\n")
            result = (state, images, ledgers)
    return checkpoint, result
