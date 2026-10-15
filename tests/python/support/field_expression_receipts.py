"""Bounded independent reader of retained native face-expression DAGs."""
import math
import struct

from tests.python.support.integral_state_receipts import decode_exchange_image


class _Reader:
    def __init__(self, data):
        self.data = bytes(data)
        self.at = 0

    def word(self):
        assert self.at + 8 <= len(self.data), "truncated expression word"
        value, = struct.unpack_from("<Q", self.data, self.at)
        self.at += 8
        return value

    def blob(self):
        n = self.word()
        assert n <= len(self.data) - self.at, "truncated expression blob"
        value = self.data[self.at:self.at+n]
        self.at += n
        return value

    def count(self, width=8):
        n = self.word()
        assert n <= (len(self.data) - self.at) // width
        return n

    def done(self):
        assert self.at == len(self.data), "trailing expression bytes"


def decode_field_expressions(data):
    reader = _Reader(data)
    assert reader.word() == 0x3158504553504F50  # POPSEPX1
    bases, identities = [], set()
    for _ in range(reader.count()):
        row = _Reader(reader.blob())
        identity = row.blob().decode()
        owner = row.word()
        assert identity and identity not in identities and owner <= (1 << 31) - 1
        identities.add(identity)
        ledger = decode_exchange_image(row.blob())
        assert not ledger["quantities"] and not ledger["consumed"]
        cells = []
        for _ in range(row.count()):
            n = row.count()
            assert 1 <= n <= 3
            cells.append(tuple(struct.unpack("<q", struct.pack("<Q", row.word()))[0]
                               for _ in range(n)))
        row.done()
        assert len(cells) == len(ledger["records"])
        keys = set()
        prior = None
        for cell, record in zip(cells, ledger["records"], strict=True):
            assert 0 <= record["axis"] < len(cell) and record["side"] in (0, 1)
            incidence = "cell" + "".join(":" + str(x) for x in cell)
            incidence += "/axis:%d/side:%d" % (record["axis"], record["side"])
            assert record["quadrature"] in (incidence, incidence + "/component:%d" % record["component"])
            key = tuple(record[name] for name in ("operation", "occurrence", "context", "quadrature"))
            assert key not in keys
            assert prior is None or prior < key, "noncanonical source incidence order"
            keys.add(key)
            prior = key
        bases.append(dict(identity=identity, owner=owner, records=ledger["records"], cells=cells))
    expressions, used_bases = [], []
    for _ in range(reader.count(16)):
        exact = reader.word()
        assert exact in (0, 1)
        terms, term_bases = [], set()
        for _ in range(reader.count(16)):
            slot, bits = reader.word(), reader.word()
            coefficient, = struct.unpack("<d", struct.pack("<Q", bits))
            assert slot < len(bases) and slot not in term_bases
            assert coefficient != 0 and math.isfinite(coefficient)
            term_bases.add(slot)
            if slot not in used_bases:
                used_bases.append(slot)
            terms.append(dict(basis=bases[slot], coefficient=coefficient, coefficient_bits=bits))
        expressions.append(dict(exact=bool(exact), terms=terms))
    roots, used_expressions = [], []
    for _ in range(reader.count()):
        slot = reader.word()
        assert slot <= len(expressions)
        if slot and slot - 1 not in used_expressions:
            used_expressions.append(slot - 1)
        roots.append(expressions[slot - 1] if slot else None)
    reader.done()
    assert used_bases == list(range(len(bases))), "noncanonical or unreferenced basis"
    assert used_expressions == list(range(len(expressions))), "noncanonical or unreferenced expression"
    return roots
