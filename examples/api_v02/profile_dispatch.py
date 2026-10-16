"""Examples-only construction dispatch, independent of the PoPS core engine.

The selected equation profiles are the defaults. Historical and companion
profiles are available by explicit name. ``require_complete`` checks the
reviewed equation-component manifest for the selected profile; it does not
demand a universal theorem or synthesize any native execution evidence.
"""

from __future__ import annotations

from dataclasses import replace
import importlib
import inspect
import json
from pathlib import Path

from .common import CaseSpec, MissingCapabilityError
from .profile_contracts import (
    ComponentBinding,
    ImplementationManifest,
    ProfileSelection,
    named_profiles,
    selected_profile,
)

_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = Path(__file__).with_name("profile_implementations.json")


def reviewed_manifest(identifier):
    """Read a versioned review; never replace stored hashes with current hashes."""
    contract = selected_profile(identifier)
    data = json.loads(_MANIFEST.read_text())
    if data.get("schema") != "pops-api02-reviewed-components-v1":
        raise ValueError("unsupported reviewed-component manifest schema")
    row = data["profiles"].get(contract.key)
    if row is None:
        raise MissingCapabilityError(
            contract.identifier,
            required=tuple(component.identifier for component in contract.required_components),
            profile=contract.key,
        )
    return ImplementationManifest(
        contract.key,
        tuple(ComponentBinding(**binding) for binding in row["bindings"]),
        tuple(tuple(pair) for pair in row["dependencies"]),
    )


def resolve_builder(identifier, *, profile=None):
    """Resolve an explicit callable and its selected defaults, without running it."""
    contract = selected_profile(identifier)
    name = contract.variant if profile is None else str(profile)
    choices = {item.name: item for item in named_profiles(contract.identifier)}
    if name not in choices:
        raise ValueError(
            f"unknown {contract.identifier} profile {name!r}; choose " + ", ".join(choices)
        )
    choice = choices[name]
    module, symbol = choice.builder.symbol.split(":")
    builder = getattr(importlib.import_module(module), symbol)
    return contract, choice, builder


def build_profile(
    identifier, *, profile=None, require_complete=False, implementation_manifest=None, **kwargs
):
    """Build the chosen Python equations and retain the actual call parameters.

    ``require_complete=True`` means all equation/composition components of the
    selected contract were reviewed against these exact source bytes. A custom
    parameter choice still requires its own numerical receipt. In particular,
    construction and source coverage never imply convergence, positivity or
    a successful C++ run.
    """
    contract, choice, builder = resolve_builder(identifier, profile=profile)
    parameters = choice.builder.kwargs()
    # Model identity is not a tunable numerical parameter of the shared builder.
    if "model" in parameters and kwargs.get("model", parameters["model"]) != parameters["model"]:
        raise ValueError("the model parameter cannot change the selected corpus identifier")
    parameters.update(kwargs)
    bound = inspect.signature(builder).bind(**parameters)
    bound.apply_defaults()
    manifest = None
    if require_complete:
        if choice.name != contract.variant:
            raise MissingCapabilityError(
                contract.identifier,
                required=("the selected equation contract " + contract.key,),
                profile=choice.name + " is a separately named composition",
            )
        manifest = (
            reviewed_manifest(contract.identifier)
            if implementation_manifest is None
            else implementation_manifest
        )
        contract.require_components(manifest, source_root=_ROOT, parameters=bound.arguments)
    elif implementation_manifest is not None:
        raise ValueError("an implementation manifest requires require_complete=True")
    spec = builder(**bound.arguments)
    if not isinstance(spec, CaseSpec):
        raise TypeError("a corpus profile builder must return its actual CaseSpec")
    selection = ProfileSelection(
        contract.identifier,
        choice.name,
        choice.builder.symbol,
        tuple(bound.arguments.items()),
        tuple(sorted(kwargs)),
        manifest,
    )
    return replace(spec, selection=selection)
