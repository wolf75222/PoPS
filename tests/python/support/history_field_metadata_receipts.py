"""Independent reader of the AMR History metadata envelope, without native decoders."""
from dataclasses import dataclass
import math
import struct


@dataclass
class _Reader:
    data: bytes
    at: int = 0

    def take(self, size):
        assert 0 <= size <= len(self.data)-self.at, "truncated History metadata"
        out = self.data[self.at:self.at+size]
        self.at += size
        return out

    def word(self):
        return int.from_bytes(self.take(8), "little")

    def blob(self):
        return self.take(self.word())

    def text(self):
        return self.blob().decode("utf-8")

    def count(self, minimum):
        count = self.word()
        assert count <= (len(self.data)-self.at)//minimum
        return count

    def end(self):
        assert self.at == len(self.data), "trailing History metadata"


def _signed(word):
    return word if word < (1 << 63) else word-(1 << 64)


def unwrap_history_field_metadata(payload, *, program=None):
    """Return verbatim FLX bytes and independent metadata/sample descriptions."""
    if not payload.startswith(b"POPSHFM1"):
        return payload, {}, {}
    envelope = _Reader(payload, 8)
    flux, encoded = envelope.blob(), envelope.blob()
    reader = _Reader(encoded)
    supports = []
    for _ in range(reader.count(56)):
        support = dict(program_identity=reader.text(), origin_point=reader.text(),
                       program_owner=_signed(reader.word()), producer_level=_signed(reader.word()),
                       topology_epoch=reader.word(), materialization_generation=reader.word(),
                       field_provider=reader.text())
        assert support["program_identity"] and support["origin_point"]
        assert program is None or support["program_identity"] == program
        assert 0 <= support["producer_level"] <= 2**31-1
        assert ((0 <= support["program_owner"] <= 2**31-1 and not support["field_provider"])
                or (support["program_owner"] == -1 and support["field_provider"]))
        supports.append(support)
    obligations = []
    for _ in range(reader.count(16)):
        obligation = dict(contract=reader.text(), origin=reader.text())
        assert obligation["contract"] and obligation["origin"]
        obligations.append(obligation)
    fields = {}
    seen_support = seen_obligation = 0
    previous = None
    for _ in range(reader.count(16)):
        name = reader.text()
        assert previous is None or previous < name
        previous = name
        rows = []
        for _ in range(reader.count(16)):
            support, obligation = reader.word(), reader.word()
            assert support <= len(supports) and obligation <= len(obligations)
            assert support <= seen_support+1 and obligation <= seen_obligation+1
            assert not obligation or support
            seen_support, seen_obligation = max(seen_support, support), max(seen_obligation, obligation)
            rows.append(dict(support=None if not support else supports[support-1],
                             representation=None if not obligation else obligations[obligation-1]))
        fields[name] = rows
    reader.end()
    assert seen_support == len(supports) and seen_obligation == len(obligations)
    assert envelope.word() == len(fields)
    samples = {}
    for name, rows in fields.items():
        assert envelope.text() == name and envelope.word() == len(rows)
        slots = []
        for row in rows:
            kind, start, interval, ordinal = (envelope.word() for _ in range(4))
            reconstruction = tuple(envelope.word() for _ in range(4))
            assert kind in (0, 1, 2, 3)
            assert (kind == 3) == any(reconstruction)
            if kind in (0, 1):
                assert (start, interval, ordinal) == (0, 0, 0)
            else:
                t, dt = (struct.unpack("<d", value.to_bytes(8, "little"))[0]
                         for value in (start, interval))
                assert math.isfinite(t) and math.isfinite(dt) and dt > 0 and ordinal > 0
            assert kind != 0 or (row["support"] is None and row["representation"] is None)
            slots.append(dict(kind=kind, start_bits=start, interval_bits=interval,
                              ordinal=ordinal, reconstruction=reconstruction))
        samples[name] = slots
    envelope.end()
    return flux, fields, samples
