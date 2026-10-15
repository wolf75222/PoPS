"""Closed Python algebra for temporal remapping of an AMR history ring."""
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any
import json


@dataclass(frozen=True, slots=True)
class HistorySample:
    time: object
    interval: object


@dataclass(frozen=True, slots=True, init=False)
class HistoryTransfer:
    """Author ``body(samples, accepted_time, ratio) -> (valid, rows)``.

    ``slots=N`` retains the square ring contract. For an explicitly bound source
    bundle, use ``source_slots=M, target_slots=N, parameters=(names, ...)``;
    the body receives those runtime scalar parameters after ``ratio``. Each
    consuming operation must bind that exact ordered source bundle and every
    named parameter; the graph never fetches implicit states or timestamps.

    Each row is ``(target_time, outgoing_interval, weights)``. ``samples``
    includes the recycled slot zero and every readable lag, in native ring
    order. The matrix acts on an immutable source image, including its flux
    ancestry. A new child ring is reconstructed entirely from that parent
    image. Existing fully covered child rings retain their exact samples.

    The body executes only during authoring. Its closed scalar DAG is emitted
    as C++; no Python callback executes at remap time. ``valid`` is evaluated
    before the row expressions, permitting explicit rejection of coincident
    times before any interpolation division.

    The validity predicate observes the native binary64 clock coordinates.
    Row arithmetic (+, -, *, / and integer powers) keeps exact rational values
    of those coordinates and literals until the one final Real conversion.
    The exact weights accompany the numerical matrix into flux ancestry and
    reconstruction identity. Transcendentals and explicit ``rounded`` nodes
    mark binary64 boundaries; no normalization changes the authored weights.
    """
    _json: str
    __pops_history_interpolation__ = True
    __pops_ir_immutable__ = True

    def __init__(self, *, body: Callable[..., tuple[Any, Any]], slots: int | None = None,
                 source_slots: int | None = None, target_slots: int | None = None,
                 parameters: tuple[str, ...] = ()) -> None:
        legacy = slots is not None and source_slots is None and target_slots is None and parameters == ()
        if slots is not None:
            if source_slots is not None or target_slots is not None:
                raise ValueError("HistoryTransfer slots cannot be combined with explicit source/target widths")
            if type(slots) is not int or slots < 2:
                raise ValueError("HistoryTransfer requires at least two exact ring slots")
            source_slots = target_slots = slots
        elif (type(source_slots) is not int or source_slots < 1
              or type(target_slots) is not int or target_slots < 1):
            raise ValueError("HistoryTransfer requires positive exact source and target widths")
        if type(parameters) is not tuple or any(
                type(name) is not str or not name.isascii() or not name.isidentifier()
                for name in parameters) or len(set(parameters)) != len(parameters):
            raise ValueError("HistoryTransfer parameters require unique ASCII identifiers")
        slots = target_slots
        if not callable(body):
            raise TypeError("HistoryTransfer body must author scalar equations")
        from pops._ir.expr import Var
        from pops.time.expressions import encode_expressions
        from pops.model.hash_data import canonical_hash_data
        scalars = tuple(Var("history_transfer_input_%d" % i, "history_transfer")
                        for i in range(2*source_slots+2+len(parameters)))
        samples = tuple(HistorySample(scalars[2*i], scalars[2*i+1]) for i in range(source_slots))
        tail = 2*source_slots
        valid, rows = body(samples, scalars[tail], scalars[tail+1], *scalars[tail+2:])
        if type(rows) is not tuple or len(rows) != slots:
            raise ValueError("HistoryTransfer must return every target ring slot")
        flattened = [valid]
        for row in rows:
            if type(row) is not tuple or len(row) != 3 or type(row[2]) is not tuple or len(row[2]) != source_slots:
                raise ValueError("HistoryTransfer rows require time, interval and one weight per source")
            flattened.extend((row[0], row[1], *row[2]))
        roots, nodes, inputs = encode_expressions(flattened, None, symbolic_inputs=scalars)
        encoded = []
        for node in nodes:
            if node[0] == "input":
                encoded.append(tuple(node))
            elif node[0] == "literal":
                encoded.append(("literal", node[1].to_data()))
            elif node[0] == "coefficient":
                raise TypeError("HistoryTransfer uses explicit intervals, not ambient method coefficients")
            else:
                encoded.append(node)
        data = dict(kind="history_transfer", schema_version=1 if legacy else 2, slots=slots,
                    input_count=len(inputs), roots=roots, nodes=tuple(encoded))
        if not legacy:
            data.update(source_slots=source_slots, parameter_names=parameters)
        # Validate through exactly the decoder used by code generation.
        validate_transfer_data(data)
        object.__setattr__(self, "_json", json.dumps(canonical_hash_data(data),
            sort_keys=True, separators=(",", ":"), allow_nan=False))

    def to_data(self):
        return json.loads(self._json)


def transfer_dimensions(data):
    """Return the validated ordered input/output widths without decoding equations."""
    from collections.abc import Mapping
    common = {"kind", "schema_version", "slots", "input_count", "roots", "nodes"}
    if not isinstance(data, Mapping) or type(data.get("schema_version")) is not int:
        raise ValueError("invalid HistoryTransfer closed graph contract")
    version = data["schema_version"]
    required = common if version == 1 else common | {"source_slots", "parameter_names"}
    if version not in (1, 2) or set(data) != required or data["kind"] != "history_transfer":
        raise ValueError("invalid HistoryTransfer closed graph contract")
    targets = data["slots"]
    sources = targets if version == 1 else data["source_slots"]
    parameters = () if version == 1 else data["parameter_names"]
    if (any(type(width) is not int or width < (2 if version == 1 else 1)
            for width in (sources, targets)) or type(parameters) not in (list, tuple)
            or any(type(name) is not str or not name.isascii() or not name.isidentifier()
                   for name in parameters) or len(set(parameters)) != len(parameters)
            or type(data["input_count"]) is not int
            or data["input_count"] != 2*sources+2+len(parameters)
            or type(data["roots"]) not in (list, tuple)
            or len(data["roots"]) != 1+targets*(sources+2)
            or type(data["nodes"]) not in (list, tuple) or not data["nodes"]
            or any(type(node) not in (list, tuple) or not node for node in data["nodes"])):
        raise ValueError("invalid HistoryTransfer closed graph contract")
    return sources, targets, tuple(parameters)


def validate_transfer_data(data):
    from pops.time.expression_dag import decode_expression_literal, restore_expression_dag
    transfer_dimensions(data)
    nodes = tuple(("literal", decode_expression_literal(node[1]))
                  if node[0] == "literal" else tuple(node) for node in data["nodes"])
    # This validation shares the existing scalar language and branch structure; a transfer cannot invent an opcode or capture another Program.
    restore_expression_dag(tuple(data["roots"]), nodes,
                           tuple(("input_%d" % i,) for i in range(data["input_count"])))
    return nodes


def require_ring_transfer(data, slots):
    """A topology remap binds one square parent ring and no extra runtime parameters."""
    validate_transfer_data(data)
    if transfer_dimensions(data) != (slots, slots, ()):
        raise ValueError("HistoryTransfer remap requires equal source/target widths and no unbound parameters")


__all__ = ["HistoryTransfer", "HistorySample"]
