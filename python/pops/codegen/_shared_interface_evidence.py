"""Private resolve-issued evidence for shared-interface Program lowering."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pops.identity import Identity, make_identity


_EVIDENCE_ISSUER = object()


@dataclass(frozen=True, slots=True, init=False)
class _ResolvedSharedInterfaceCodegenEvidence:
    """Nominal proof bound to one exact resolved plan and Program graph.

    Construction is intentionally unavailable.  ``pops.resolve`` records the canonical capability
    on its immutable plan; the private compiler route issues this value only after re-verifying that
    plan.  Public low-level emitters never accept this type or a boolean substitute.
    """

    plan_identity: Identity
    program_graph_hash: str
    target: str
    layout_plan_id: str
    hierarchy_identity: Identity
    interfaces: tuple[tuple[str, Identity], ...]
    identity: Identity

    def __new__(cls):
        raise TypeError(
            "shared-interface codegen evidence is issued only from an exact resolved plan"
        )

    @classmethod
    def _issue(cls, issuer: object) -> _ResolvedSharedInterfaceCodegenEvidence:
        if issuer is not _EVIDENCE_ISSUER:
            raise TypeError("shared-interface codegen evidence issuer is invalid")
        return object.__new__(cls)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "plan_identity": self.plan_identity.to_data(),
            "program_graph_hash": self.program_graph_hash,
            "target": self.target,
            "layout_plan_id": self.layout_plan_id,
            "hierarchy_identity": self.hierarchy_identity.to_data(),
            "interfaces": [
                {"qualified_id": name, "identity": identity.to_data()}
                for name, identity in self.interfaces
            ],
        }

    def require(self, program: Any, *, target: str) -> None:
        """Authenticate this proof against the exact detached Program being lowered."""
        from pops.time import Program

        if type(program) is not Program:
            raise TypeError("shared-interface codegen evidence requires an exact Program")
        if target != self.target or target != "amr_system":
            raise ValueError("shared-interface codegen evidence changed its resolved target")
        if program.to_graph().graph_hash != self.program_graph_hash:
            raise ValueError("shared-interface codegen evidence belongs to another Program graph")
        if self.identity != make_identity(
            "resolved-shared-interface-codegen", self._payload()
        ):
            raise ValueError("shared-interface codegen evidence identity verification failed")


def _issue_shared_interface_codegen_evidence(
    plan: Any,
) -> _ResolvedSharedInterfaceCodegenEvidence | None:
    """Issue the nominal compiler proof from one exact, verified resolve result."""
    from pops.codegen._plans import ResolvedSimulationPlan

    if type(plan) is not ResolvedSimulationPlan:
        raise TypeError("shared-interface codegen evidence requires a resolved simulation plan")
    plan.verify()
    capabilities = plan.capabilities.get("shared_interfaces")
    if not isinstance(capabilities, Mapping) or set(capabilities) != {
        "implicit_jacvec_pair"
    }:
        raise TypeError("resolved shared-interface codegen evidence is not canonical")
    required = capabilities["implicit_jacvec_pair"]
    if type(required) is not bool:
        raise TypeError("resolved shared-interface implicit-JVP evidence must be an exact bool")
    if not required:
        return None
    if plan.target != "amr_system" or len(plan.layout_plan.layouts) != 1:
        raise ValueError(
            "shared-interface implicit-JVP evidence requires one AMR runtime layout"
        )
    hierarchy = plan.resolved_hierarchy
    hierarchy_identity = getattr(hierarchy, "identity", None)
    if type(hierarchy_identity) is not Identity:
        raise TypeError("shared-interface codegen evidence lost its resolved hierarchy")

    declarations: dict[str, Identity] = {}
    for block in plan.blocks:
        numerics = block.numerics
        for boundary in (() if numerics is None else numerics.boundaries):
            for interface in getattr(boundary, "interfaces", ()):
                name = getattr(interface, "qualified_id", None)
                canonical = getattr(interface, "canonical_identity", None)
                if not isinstance(name, str) or not name or not callable(canonical):
                    raise TypeError(
                        "shared-interface codegen evidence found an invalid declaration"
                    )
                identity = make_identity("shared-interface-declaration", canonical())
                previous = declarations.setdefault(name, identity)
                if previous != identity:
                    raise ValueError(
                        "shared-interface codegen evidence found competing declarations"
                    )
    if len(declarations) != 1:
        raise ValueError(
            "shared-interface implicit-JVP evidence requires one exact interface declaration"
        )

    program_graph_hash = plan.time.to_graph().graph_hash
    if not isinstance(program_graph_hash, str) or not program_graph_hash:
        raise TypeError("shared-interface codegen evidence lost the Program graph identity")
    evidence = _ResolvedSharedInterfaceCodegenEvidence._issue(_EVIDENCE_ISSUER)
    object.__setattr__(evidence, "plan_identity", Identity.from_data(plan.plan_identity.to_data()))
    object.__setattr__(evidence, "program_graph_hash", program_graph_hash)
    object.__setattr__(evidence, "target", plan.target)
    object.__setattr__(evidence, "layout_plan_id", plan.layout_plan.qualified_id)
    object.__setattr__(
        evidence, "hierarchy_identity", Identity.from_data(hierarchy_identity.to_data())
    )
    object.__setattr__(
        evidence,
        "interfaces",
        tuple(
            (name, Identity.from_data(identity.to_data()))
            for name, identity in sorted(declarations.items())
        ),
    )
    object.__setattr__(
        evidence,
        "identity",
        make_identity("resolved-shared-interface-codegen", evidence._payload()),
    )
    evidence.require(plan.time, target=plan.target)
    return evidence


__all__: list[str] = []
