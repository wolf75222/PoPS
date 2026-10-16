"""Transport checked scientific graphs across the registry-free compile boundary.

Only the live checker can issue a proof. The immutable proof retains canonical
bytes, never a Case, registry, GraphFunction or authoring Program. Its detached
binding belongs to one exact frozen clone; it is not a public authoring capability.
"""
from dataclasses import dataclass
import json


def _image(program):
    return json.dumps(program._serialize(include_provenance=False),
                      sort_keys=True, separators=(",", ":"))


def _nodes(program):
    from .temporal_manifest import _walk
    values = list(_walk(program._values))
    if program._dt_bound is not None:
        values.extend(_walk(program._dt_bound[0]))
    return {value.id: value for value in values}


@dataclass(frozen=True, slots=True, init=False)
class _PreparedScientificContracts:
    image: str
    contracts: tuple

    def __new__(cls):
        raise TypeError("scientific compile proof requires live graph validation")

    def bind(self, program):
        if not getattr(program, "_compiled_detached", False) or not program._frozen:
            raise TypeError("scientific compile binding requires a frozen detached Program")
        if _image(program) != self.image:
            raise ValueError("detached scientific Program differs from its validated graph")
        from pops.scientific_graph import CONTRACT, _key
        nodes = _nodes(program)
        bindings = []
        for identifier, contract in self.contracts:
            node = nodes.get(identifier)
            if node is None or _key(node.attrs.get(CONTRACT)) != contract:
                raise ValueError("detached scientific contract differs from its validated graph")
            bindings.append((node, contract))
        bound = object.__new__(_DetachedScientificContracts)
        object.__setattr__(bound, "proof", self)
        object.__setattr__(bound, "program", program)
        object.__setattr__(bound, "nodes", tuple(bindings))
        return bound


@dataclass(frozen=True, slots=True, init=False)
class _DetachedScientificContracts:
    proof: _PreparedScientificContracts
    program: object
    nodes: tuple

    def __new__(cls):
        raise TypeError("detached scientific binding requires live graph validation")

    def require_program(self, program):
        if (program is not self.program or not program._frozen
                or not getattr(program, "_compiled_detached", False)
                or _image(program) != self.proof.image):
            raise ValueError("detached scientific Program differs from its validated graph")
        current = _nodes(program)
        if any(current.get(node.id) is not node for node, _ in self.nodes):
            raise ValueError("scientific contract is not the bound detached node")

    def require_node(self, value, contract):
        from pops.scientific_graph import _key
        if value.prog is not self.program or not any(
                value is node and _key(contract) == data for node, data in self.nodes):
            raise ValueError("scientific contract has no exact detached validation proof")


def prepare_scientific_contracts(program):
    """Replay live contracts, or rebind an already received immutable detached proof."""
    from pops.scientific_graph import CONTRACT, validate_scientific_program
    expected = {value.id for value in _nodes(program).values()
                if value.attrs.get(CONTRACT, {}).get("mode") == "strict"}
    if not expected:
        return None
    if getattr(program, "_compiled_detached", False):
        bound = require_scientific_binding(program)
        return bound.proof
    checked = {}
    validate_scientific_program(program, _contract_receipts=checked)
    if set(checked) != expected:
        raise ValueError("scientific compile proof requires every strict contract to be revalidated")
    proof = object.__new__(_PreparedScientificContracts)
    object.__setattr__(proof, "image", _image(program))
    object.__setattr__(proof, "contracts", tuple(sorted(checked.items())))
    return proof


def require_scientific_binding(program):
    bound = getattr(program, "_scientific_compile_binding", None)
    if type(bound) is not _DetachedScientificContracts:
        raise ValueError("detached scientific Program has no prepared validation proof")
    bound.require_program(program)
    return bound
