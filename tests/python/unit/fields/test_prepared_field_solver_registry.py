"""Extension tests for the backend-agnostic prepared field-solver protocol."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from pops.fields._prepared_field_solver_registry import (
    PreparedFieldSolverFacts,
    PreparedFieldSolverProvider,
    PreparedFieldSolverResolution,
    PreparedFieldSolverUsePolicy,
    prepared_field_solver_binding_from_data,
    prepared_field_solver_binding_from_descriptor,
    register_prepared_field_solver_provider,
)
from pops.fields.providers import PreparedFieldSolver


def _resolve(options, facts, where):
    del where
    return PreparedFieldSolverResolution(
        {
            "route": "example.native-field-solver",
            "schema": "example.native-field-solver.options@1",
            "options": {"scale": options["scale"]},
        },
        {
            "provider_id": "example.topology",
            "topology_identity": facts.layout["topology_identity"],
        },
    )


def _validate(use, where):
    if use.facts.target != "example.runtime@1":
        raise ValueError("%s example provider target mismatch" % where)
    if use.facts.hierarchy.get("policy_id") != "tests.field-hierarchy.coupled-graph":
        raise ValueError("%s example provider hierarchy-policy mismatch" % where)


def _install(context, binding):
    context.install(binding)


_PROVIDER = register_prepared_field_solver_provider(PreparedFieldSolverProvider(
    provider_id="tests.field-solver.example",
    version=1,
    resolver_id="tests.field-solver.example.resolve@1",
    installer_id="tests.field-solver.example.install@1",
    use_policy=PreparedFieldSolverUsePolicy(
        "tests.field-solver.example.use",
        1,
        {
            "target": "example.runtime@1",
            "hierarchy_policy": "tests.field-hierarchy.coupled-graph@3",
        },
        _validate,
    ),
    resolver=_resolve,
    native_installer=_install,
))


def _facts(target: str = "example.runtime@1") -> PreparedFieldSolverFacts:
    return PreparedFieldSolverFacts(
        target=target,
        operator={"identity": "example.operator@1"},
        layout={"topology_identity": "example.topology-instance@1"},
        hierarchy={
            "policy_id": "tests.field-hierarchy.coupled-graph",
            "interface_version": 3,
            "option_schema": "tests.field-hierarchy.coupled-graph.options@2",
            "options": {"overlap": 2},
        },
        boundary={"identity": "example.boundary@1"},
        nonlinear=False,
    )


@dataclass
class _Context:
    installed: object | None = None

    def install(self, binding) -> None:
        self.installed = binding


def test_extension_target_and_contract_cross_the_generic_registry_unchanged() -> None:
    descriptor = PreparedFieldSolver(_PROVIDER, scale=2.5)
    binding = prepared_field_solver_binding_from_descriptor(
        descriptor, facts=_facts(), where="example field"
    )

    restored = prepared_field_solver_binding_from_data(binding.to_data())
    assert restored.identity == binding.identity
    assert restored.facts.target == "example.runtime@1"
    assert restored.facts.hierarchy["policy_id"] == (
        "tests.field-hierarchy.coupled-graph"
    )
    assert restored.resolution.native_contract["route"] == "example.native-field-solver"

    context = _Context()
    _PROVIDER.install(context, restored)
    assert context.installed is restored


def test_provider_owns_target_policy_instead_of_the_protocol_core() -> None:
    descriptor = PreparedFieldSolver(_PROVIDER, scale=1.0)
    with pytest.raises(ValueError, match="target mismatch"):
        prepared_field_solver_binding_from_descriptor(
            descriptor, facts=_facts("another.runtime@7"), where="example field"
        )


def test_deserialized_resolution_is_replayed_by_its_registered_provider() -> None:
    binding = prepared_field_solver_binding_from_descriptor(
        PreparedFieldSolver(_PROVIDER, scale=1.0),
        facts=_facts(),
        where="example field",
    )
    forged = type(binding).create(
        provider=_PROVIDER,
        options=binding.options,
        facts=binding.facts,
        resolution=PreparedFieldSolverResolution(
            {"route": "forged", "schema": "forged@1", "options": {"scale": 1.0}},
            binding.resolution.topology_contract,
        ),
    )
    with pytest.raises(ValueError, match="nondeterministic"):
        prepared_field_solver_binding_from_data(forged.to_data())
