"""Explicit observation of authenticated runtime inputs at one exact state stage."""
from __future__ import annotations

from pops.model.provider_pack import ComponentKey, ProviderPack


def runtime_input_pack(pack, field, space):
    """Select the whole declared field surface, refusing computed or missing producers."""
    declaration = field.declaration_ref if field.is_instance else field
    owner = str(declaration.owner_path.canonical())
    rows = []
    for component in space.components:
        key = ComponentKey(owner, "field", space.name, component)
        entry = pack.lookup(key)
        if entry.producer != "runtime_input":
            raise ValueError(
                "input_fields requires runtime_input for every claimed component; "
                "%s/%s is produced by %r" % (key.space, component, entry.producer))
        rows.append((key, pack.contract(key), entry))
    return ProviderPack(rows, capacity=pack.capacity)


def input_fields(program, state, *, for_rate, name=None):
    """Delegate observation to the Program that owns the state and schedule."""
    return program.input_fields(state, for_rate=for_rate, name=name)
