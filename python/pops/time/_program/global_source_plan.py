"""Live-issued authority transported across the registry-free compiler boundary.

The resolved proof contains immutable data only. Its detached binding names exact
clone-owned nodes and ports, never a Case or an authoring registry.
"""
from dataclasses import dataclass
import json

from .temporal_manifest import _walk


def _signature(program, *, live):
    from pops.model.global_quantity import GlobalQuantityHandle
    from pops.time.references import canonical_handle
    from .integrals import integral_units_bytes

    rows = []
    for value in _walk(program._values):
        bindings = value.attrs.get("physical_global_inputs_v1", ())
        if not bindings:
            continue
        if value.op != "source":
            raise ValueError("physical global bindings require a named source evaluation")
        for row in bindings:
            if (set(row) != {"port", "units", "input", "version"}
                    or type(row["version"]) is not int or row["version"] != 1):
                raise ValueError("physical global source binding metadata changed")
            port = row["port"]
            if (not isinstance(port, GlobalQuantityHandle) or not port.is_instance
                    or port.block_ref != value.state_ref.block_ref):
                raise ValueError("physical global source binding block owner changed")
            if live and port.block_ref[port.declaration_ref] is not port:
                raise ValueError("physical global source input is not the registry-issued port")
            index = row["input"]
            if type(index) is not int or not 1 <= index < len(value.inputs):
                raise ValueError("physical global source binding input index changed")
            capture = program._canonical_value(value.inputs[index])
            units = integral_units_bytes(port.units)
            if (capture.op != "integral_candidate" or capture.vtype != "scalar"
                    or capture.point != value.point or row["units"] != units
                    or capture.attrs.get("units") != units
                    or capture.attrs.get("scope") != "candidate"):
                raise ValueError("physical global source binding capture point/units/scope changed")
            port_data = json.dumps(canonical_handle(port).canonical_identity(),
                                   sort_keys=True, separators=(",", ":"))
            rows.append((value.id, port_data, units, index, capture.id, row["version"]))
    return (program._ir_hash(), tuple(rows)) if rows else None


@dataclass(frozen=True, slots=True, init=False)
class _PreparedSourceGlobals:
    signature: tuple
    module_hashes: tuple

    def __new__(cls):
        raise TypeError("physical global authority requires live registry preparation")

    def require_live(self, program):
        if (_signature(program, live=True) != self.signature
                or _module_hashes(program) != self.module_hashes):
            raise ValueError("prepared physical global authority changed")

    def bind(self, program):
        if not getattr(program, "_compiled_detached", False):
            raise TypeError("physical global binding requires a detached Program")
        if _signature(program, live=False) != self.signature:
            raise ValueError("detached physical global authority changed")
        nodes = tuple((value, tuple(row["port"] for row in
                                   value.attrs["physical_global_inputs_v1"]))
                      for value in _walk(program._values)
                      if value.attrs.get("physical_global_inputs_v1"))
        bound = object.__new__(_DetachedSourceGlobals)
        object.__setattr__(bound, "proof", self)
        object.__setattr__(bound, "program", program)
        object.__setattr__(bound, "nodes", nodes)
        return bound


@dataclass(frozen=True, slots=True, init=False)
class _DetachedSourceGlobals:
    proof: _PreparedSourceGlobals
    program: object
    nodes: tuple

    def __new__(cls):
        raise TypeError("detached physical global authority requires prepared proof")

    def require(self, program, evaluation=None):
        if program is not self.program or _signature(program, live=False) != self.proof.signature:
            raise ValueError("detached physical global authority changed")
        current = tuple(value for value in _walk(program._values)
                        if value.attrs.get("physical_global_inputs_v1"))
        if len(current) != len(self.nodes):
            raise ValueError("detached physical global source nodes changed")
        for value, (issued, ports) in zip(current, self.nodes, strict=True):
            if value is not issued:
                raise ValueError("detached physical global source is not the bound node")
            rows = value.attrs["physical_global_inputs_v1"]
            if len(rows) != len(ports) or any(row["port"] is not port
                                            for row, port in zip(rows, ports, strict=True)):
                raise ValueError("physical global source input is not the bound detached port")
        if evaluation is not None and not any(evaluation is value for value, _ in self.nodes):
            raise ValueError("physical global source evaluation has no detached authority")

    def require_module(self, evaluation, module):
        expected = dict(self.proof.module_hashes).get(evaluation.id)
        if module is None or module.module_hash() != expected:
            raise ValueError("physical global source Module/body authority changed")


def _module_hashes(program):
    result = []
    for value in _walk(program._values):
        if value.attrs.get("physical_global_inputs_v1"):
            block = value.state_ref.block_ref
            model = block._instance_registry.spec(block.local_id)["model"]
            module = getattr(model, "module", model)
            result.append((value.id, module.module_hash()))
    return tuple(result)


def prepare_source_globals(program):
    if getattr(program, "_compiled_detached", False):
        bound = getattr(program, "_physical_global_source_bindings", None)
        if _signature(program, live=False) is None:
            return None
        if type(bound) is not _DetachedSourceGlobals:
            raise ValueError("detached physical global source has no prepared authority")
        bound.require(program)
        return bound.proof
    signature = _signature(program, live=True)
    if signature is None:
        return None
    proof = object.__new__(_PreparedSourceGlobals)
    object.__setattr__(proof, "signature", signature)
    object.__setattr__(proof, "module_hashes", _module_hashes(program))
    return proof


def require_source_global_plan(proof, program):
    if proof is None:
        if prepare_source_globals(program) is not None:
            raise ValueError("resolved physical global source lost its prepared authority")
    elif type(proof) is not _PreparedSourceGlobals:
        raise TypeError("physical global source requires exact prepared authority")
    else:
        proof.require_live(program)


def require_detached_source_global(evaluation, *, module=None):
    bound = getattr(evaluation.prog, "_physical_global_source_bindings", None)
    if type(bound) is not _DetachedSourceGlobals:
        raise ValueError("detached physical global source has no prepared authority")
    bound.require(evaluation.prog, evaluation)
    if module is not None:
        bound.require_module(evaluation, module)
