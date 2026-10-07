"""Persist the native AMR state storage image, including every patch's ghosts.

Python transports an opaque, versioned native image. It neither reconstructs
patch values nor fills ghosts. The native codec owns geometry, placement and
exact scalar bits, including redistribution onto a recorded patch topology.
"""
from __future__ import annotations

import numpy as np

STATE_CARRIERS_KEY = "state_carriers_checkpoint"
STATE_CARRIERS_CONTRACT = "pops.amr.state-carriers-checkpoint@1"


def _native_image(image):
    if type(image) is not bytes or len(image) <= 8 or image[:8] != b"POPSCAR1":
        raise ValueError("AMR state carrier checkpoint requires an exact native @1 image")
    return image


def _method(sim, name):
    method = getattr(sim, name, None)
    if not callable(method):
        raise TypeError("AMR engine lacks native full-state storage checkpoint: " + name)
    return method


def capture_checkpoint_state_carriers(owner, sim, payload, *, prepared_capture=None):
    from pops.output._checkpoint_collective import checkpoint_topology, consensus

    topology = checkpoint_topology(owner)
    error = None
    capture = None
    try:
        capture = _method(sim, "checkpoint_state_carriers" if prepared_capture is None
                          else "_prepared_checkpoint_state_carriers")
    except BaseException as exc:
        error = exc
    # A missing method on one rank must fail before peers enter native collectives.
    consensus(topology, "AMR state carrier capture preflight", error=error)
    error = None
    image = None
    try:
        image = _native_image(capture() if prepared_capture is None else capture(prepared_capture))
        payload[STATE_CARRIERS_KEY] = np.frombuffer(image, dtype=np.uint8).copy()
    except BaseException as exc:
        error = exc
    consensus(topology, "AMR state carrier serialization", error=error)


def prepare_checkpoint_state_carriers(owner, sim, payload):
    from pops.output._checkpoint_collective import checkpoint_topology, consensus

    topology = checkpoint_topology(owner)
    error = None
    image = None
    validate = None
    try:
        validate = _method(sim, "validate_checkpoint_state_carriers")
        _method(sim, "restore_checkpoint_state_carriers")
        if STATE_CARRIERS_KEY not in payload:
            raise ValueError("AMR checkpoint lacks its exact full-state storage image")
        raw = np.asarray(payload[STATE_CARRIERS_KEY])
        if raw.dtype != np.dtype(np.uint8) or raw.ndim != 1:
            raise ValueError("AMR state carrier checkpoint must be one-dimensional uint8")
        image = _native_image(raw.tobytes(order="C"))
    except BaseException as exc:
        error = exc
    consensus(topology, "AMR state carrier restore preflight", error=error)
    # The codec validates recorded source geometry; the rebuilt destination is
    # checked again before publication within the native restart transaction.
    validate(image)
    return image
