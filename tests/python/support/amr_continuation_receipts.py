"""Independent bounded inspection of the native accepted Continuation forest."""
import struct

from tests.python.support.api_v02_receipts import _decode
from tests.python.support.relaxed_flux_receipts import fragments


def forest(spec, image, *, rank=None):
    assert image[:8] == b"POPSANDF"
    faces, interfaces, cursor = fragments(image, _with_cursor=True)

    def take(count):
        nonlocal cursor
        assert 0 <= count <= len(image)-cursor
        result = image[cursor:cursor+count]
        cursor += count
        return result

    def word():
        return int.from_bytes(take(8), "little")

    def signed():
        return int.from_bytes(take(8), "little", signed=True)

    def blob():
        return take(word())

    def count(minimum):
        value = word()
        assert value <= (len(image)-cursor)//minimum
        return value

    # Accepted synchronization events: three integer fields, a phase string,
    # then the exact level/tick/rational/Real clock descriptor.
    events = []
    for _ in range(count(72)):
        levels = signed(), signed()
        block, phase = signed(), blob().decode()
        clock = signed(), signed(), signed(), signed(), struct.unpack("<d", take(8))[0]
        assert clock[3] > 0
        events.append(dict(levels=levels, block=block, phase=phase, clock=clock))
    tag = word()
    assert tag in (0, 1)
    if tag:
        assert blob()
        word(), word(), word()
    node = next(v for v in spec.case._time._values
                if v.op == "post_synchronization" and "continuation_forest" in v.attrs)
    contract = node.attrs["continuation_forest"]
    records = []
    for ordinal in range(count(56)):
        level = signed()
        assert level == ordinal-1
        epoch, generation, high_water = word(), word(), signed()
        identity, collective, payload = blob().decode(), blob(), blob()
        assert identity and collective and payload and high_water >= 0
        schema = contract["root_schema" if level == -1 else "level_schema"]
        decoded = _decode(None, payload, rank=rank, schema=schema["read"])
        assert identity == decoded["identity"]
        assert high_water >= max((entry["version"] for entry in decoded["domains"]), default=0)
        records.append(dict(level=level, epoch=epoch, generation=generation,
                            high_water=high_water, collective=collective, image=payload, **decoded))
    assert len(records) >= 2 and cursor == len(image)
    assert len(records[0]["collective"]) == 32
    return dict(records=records, faces=faces, interfaces=interfaces, events=events)
