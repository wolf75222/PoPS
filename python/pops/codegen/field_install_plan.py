"""Immutable resolved field-install artifact and provider-owned runtime binding."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pops.codegen._artifact_freeze import freeze_artifact_value
from pops.codegen.lowering_coverage import LoweringCoverageReport
from pops.fields._identity import field_identity, strict_field_data
from pops.fields._prepared_field_lowering_registry import (
    PreparedFieldRuntimeInstallContext,
    prepared_field_lowering_binding_from_data,
    prepared_field_lowering_provider_from_identity,
)
from pops.fields._prepared_field_nullspace_registry import (
    prepared_field_nullspace_binding_from_data,
)
from pops.fields._prepared_field_solver_registry import (
    prepared_field_solver_binding_from_data,
)
from pops.fields.discretization import (
    FieldDiscretizationProtocol,
    field_discretization_data,
    require_field_discretization,
)
from pops.fields.operator import FieldOperator
from pops.identity import Identity, canonical_bytes


def native_plain_data(value: Any) -> Any:
    """Detach a recursively frozen native carrier into canonical ordinary containers."""
    if isinstance(value, Mapping):
        return {key: native_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native_plain_data(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((native_plain_data(item) for item in value), key=repr)
    return value


@dataclass(frozen=True, slots=True)
class ResolvedFieldInstallPlan:
    """Complete field semantics plus their exact authenticated provider lowering."""

    name: str
    operator: FieldOperator
    discretization: FieldDiscretizationProtocol
    target: str
    rhs_providers: tuple[Any, ...]
    native_options: Any
    coverage: LoweringCoverageReport
    nonlinear_provider: Any
    identity: Identity

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise TypeError("ResolvedFieldInstallPlan name must be non-empty")
        if self.operator.name != self.name:
            raise ValueError("field install name disagrees with FieldOperator")
        require_field_discretization(
            self.discretization, where="resolved field install discretization"
        )
        if type(self.target) is not str or not self.target:
            raise TypeError("field install target must be a non-empty exact identity")
        from pops.model import Handle
        if not self.rhs_providers or any(
            not isinstance(provider, Handle)
            or not provider.is_resolved
            or provider.kind != "field_operator"
            for provider in self.rhs_providers
        ):
            raise TypeError("field install requires canonical field-operator RHS providers")
        if not isinstance(self.coverage, LoweringCoverageReport):
            raise TypeError("field install requires exact lowering coverage")
        if not isinstance(self.native_options, Mapping):
            raise TypeError("resolved field install native_options must be a mapping")
        current_protocol = getattr(
            self.discretization.method, "_prepared_field_lowering", None
        )
        if not callable(current_protocol):
            raise TypeError("resolved field method lost its prepared lowering provider")
        current = current_protocol()
        lowering_binding = prepared_field_lowering_binding_from_data(
            self.native_options.get("method_provider")
        )
        provider_native = native_plain_data(lowering_binding.resolution.native_options)
        if any(
            key not in self.native_options
            or native_plain_data(self.native_options[key]) != value
            for key, value in provider_native.items()
        ):
            raise ValueError("resolved field lowering native contract changed")
        if (
            not isinstance(current, tuple)
            or len(current) != 2
            or current[0].authority() != native_plain_data(lowering_binding.provider)
            or native_plain_data(current[1]) != native_plain_data(lowering_binding.options)
        ):
            raise ValueError("resolved field method provider selection changed")
        solver_binding = prepared_field_solver_binding_from_data(
            self.native_options.get("solver_provider")
        )
        if solver_binding.facts != lowering_binding.resolution.solver_facts:
            raise ValueError("resolved field solver facts changed after method lowering")
        if solver_binding.facts.target != self.target:
            raise ValueError("resolved field target changed after method lowering")
        nullspace_binding = prepared_field_nullspace_binding_from_data(
            self.native_options.get("nullspace_provider")
        )
        if nullspace_binding.facts != lowering_binding.resolution.nullspace_facts:
            raise ValueError("resolved field nullspace facts changed after method lowering")
        nonlinear_manifest = self.native_options.get("nonlinear")
        if nonlinear_manifest is None:
            if self.nonlinear_provider is not None:
                raise ValueError("field install retains an undeclared nonlinear provider")
        else:
            validate = getattr(self.nonlinear_provider, "__post_init__", None)
            install = getattr(self.nonlinear_provider, "install", None)
            to_data = getattr(self.nonlinear_provider, "to_data", None)
            if not callable(validate) or not callable(install) or not callable(to_data):
                raise TypeError(
                    "field nonlinear provider does not implement the prepared protocol"
                )
            validate()
            if to_data() != nonlinear_manifest:
                raise ValueError(
                    "field nonlinear provider disagrees with its canonical manifest"
                )
        object.__setattr__(
            self, "native_options", freeze_artifact_value(dict(self.native_options))
        )
        expected = field_identity(
            "resolved-field-install", self.to_data(include_identity=False)
        )
        if self.identity != expected:
            raise ValueError("resolved field install identity is not canonical")

    def to_data(self, *, include_identity: bool = True) -> dict[str, Any]:
        data = {
            "schema_version": 1,
            "name": self.name,
            "operator": self.operator.to_data(),
            "discretization": field_discretization_data(
                self.discretization, where="resolved field install discretization"
            ),
            "target": self.target,
            "rhs_providers": [
                provider.canonical_identity() for provider in self.rhs_providers
            ],
            "native_options": self.native_install_data(),
            "coverage": self.coverage.to_data(),
        }
        if include_identity:
            data["identity"] = self.identity.token
        return data

    def native_install_data(self) -> dict[str, Any]:
        data = native_plain_data(self.native_options)
        if not isinstance(data, dict):
            raise TypeError("resolved field native install data must be a dict")
        return data

    def provider_parameter_handles(self, consumer: str) -> tuple[Any, ...]:
        """Return one provider-owned parameter pack for an opaque consumer identity."""
        if type(consumer) is not str or not consumer:
            raise TypeError("field parameter consumer must be a non-empty exact string")
        binding = prepared_field_lowering_binding_from_data(
            self.native_options["method_provider"]
        )
        provider = prepared_field_lowering_provider_from_identity(binding.provider)
        packs = provider.parameter_handles(binding, self.operator, self.discretization)
        if not isinstance(packs, Mapping) or any(
            type(name) is not str or not name or type(handles) is not tuple
            for name, handles in packs.items()
        ):
            raise TypeError(
                "field lowering parameter resolver must return string-to-tuple packs"
            )
        return packs.get(consumer, ())

    def bind_native_options(self, params: Mapping[Any, Any]) -> dict[str, Any]:
        """Bind runtime values through the authenticated method provider."""
        if not isinstance(params, Mapping):
            raise TypeError("field native bind parameters must be a mapping")
        binding = prepared_field_lowering_binding_from_data(
            self.native_options["method_provider"]
        )
        provider = prepared_field_lowering_provider_from_identity(binding.provider)
        result = provider.prepare_bound_options(
            binding, self.operator, self.discretization, params
        )
        plain = native_plain_data(freeze_artifact_value(dict(result)))
        if not isinstance(plain, dict):
            raise TypeError("field lowering runtime binder lost its mapping root")
        return plain

    def install_runtime(
        self,
        context: PreparedFieldRuntimeInstallContext,
        params: Mapping[Any, Any],
    ) -> None:
        """Delegate the complete runtime install to the authenticated method provider."""
        if type(context) is not PreparedFieldRuntimeInstallContext:
            raise TypeError("field runtime install requires an exact prepared context")
        if context.target != self.target:
            raise ValueError("field runtime install target changed after resolve")
        expected_slot = self.native_options.get("provider_slot")
        if context.slot != expected_slot:
            raise ValueError("field runtime provider slot changed after resolve")
        if not isinstance(params, Mapping):
            raise TypeError("field runtime bind parameters must be a mapping")
        binding = prepared_field_lowering_binding_from_data(
            self.native_options["method_provider"]
        )
        provider = prepared_field_lowering_provider_from_identity(binding.provider)
        provider.install_runtime(
            binding, context, self.operator, self.discretization, params
        )

    def component_bindings(self) -> tuple[dict[str, Any], ...]:
        binding = prepared_field_solver_binding_from_data(
            self.native_options["solver_provider"]
        )
        return tuple(
            native_plain_data(component)
            for component in binding.resolution.component_bindings
        )

    def require_component_inputs(self, components: tuple[Any, ...]) -> None:
        from pops.external import CompiledComponentArtifact, ExternalComponent

        by_id = {}
        for component in components:
            if type(component) is ExternalComponent:
                component_id = component.component_manifest.component_id
                manifest = component.component_manifest.manifest_digest.token
                interface = canonical_bytes(
                    strict_field_data(component.component_type.interface.to_data())
                )
                source_package = component.package_identity.token
                parameters = canonical_bytes(
                    strict_field_data(component.to_data()["parameters"])
                )
            elif type(component) is CompiledComponentArtifact:
                component.verify()
                component_id = component.component_id
                manifest = component.component_manifest.token
                interface = canonical_bytes(
                    strict_field_data(component.interface.to_data())
                )
                source_package = (
                    None
                    if component.source_package is None
                    else component.source_package.token
                )
                parameters = None
            else:
                continue
            by_id[component_id] = (manifest, interface, source_package, parameters)
        for binding in self.component_bindings():
            component_id = binding["component_id"]
            actual = by_id.get(component_id)
            if actual is None:
                raise ValueError(
                    "field %r requires exact component %r in resolve(components=)"
                    % (self.name, component_id)
                )
            expected = (
                binding["component_manifest_identity"],
                canonical_bytes(strict_field_data(binding["native_interface"])),
                binding["source_package_identity"],
            )
            if actual[:3] != expected or (
                actual[3] is not None
                and actual[3] != canonical_bytes(
                    strict_field_data(binding["parameters"])
                )
            ):
                raise ValueError(
                    "field %r component %r changed source package, manifest, native "
                    "interface identity, or parameters" % (self.name, component_id)
                )


__all__ = ["ResolvedFieldInstallPlan", "native_plain_data"]
