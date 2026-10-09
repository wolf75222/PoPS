"""Join an immediate cursor receipt to saved checkpoint bytes, without a runtime import."""
import json
import math

import numpy as np


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate checkpoint cursor key: " + key)
        result[key] = value
    return result


def _typed_json(value):
    """JSON equality must retain bool/int/float distinctions in authority metadata."""
    kind = type(value)
    if value is None or kind in (str, bool, int):
        return kind.__name__, value
    if kind is float and math.isfinite(value):
        return "float", value.hex()
    if kind is list:
        return "list", tuple(_typed_json(item) for item in value)
    if kind is dict and all(type(key) is str for key in value):
        return "dict", tuple((key, _typed_json(value[key])) for key in sorted(value))
    raise ValueError("checkpoint cursor receipt must contain finite, exact JSON data")


def join_checkpoint_cursors(checkpoint, expected):
    """Authenticate cursor identity in the common outer Uniform/AMR checkpoint."""
    with np.load(checkpoint, allow_pickle=False) as archive:
        if "runtime_consumer_cursors" not in archive:
            raise ValueError("checkpoint is missing runtime consumer cursors")
        encoded = archive["runtime_consumer_cursors"]
        if encoded.shape != () or encoded.dtype.kind != "U":
            raise ValueError("checkpoint consumer cursors must be a Unicode scalar")
        text = encoded.item()
    stored = json.loads(text, object_pairs_hook=_unique_object)
    if type(stored) is not dict or type(expected) is not dict:
        raise ValueError("checkpoint and immediate cursor receipt must be objects")
    if _typed_json(stored) != _typed_json(expected):
        raise ValueError("checkpoint consumer cursors differ from immediate receipt")
    return {"schema": "sol61.checkpoint-consumer-cursor-join@1",
            "consumer_cursors_CP_joined": True, "consumer_cursors": stored,
            "scope": "saved outer checkpoint and immediate receipt only; no readiness or Native seal"}
