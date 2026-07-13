"""pops.mesh.layouts -- how a mesh is organised for execution (Spec 5 sec.5.10).

A layout is NOT a backend and NOT a compile target: it says what mesh STRUCTURE the
runtime must materialise. ``Uniform`` is a single-level mesh; ``AMR`` is an adaptively
refined hierarchy whose policies come from :mod:`pops.mesh.amr`. Spec 5 (sec.8.5)
replaces ``target="amr_system"`` / ``AmrSystemTarget()`` with ``layout=AMR(...)``.

These are inert descriptors: they declare requirements / capabilities and answer
``available(context)`` so an unsupported route is refused before the runtime is touched.
"""
from __future__ import annotations

from typing import Any

from .._descriptor import Availability, MeshDescriptor
from ..amr import IgnoreAMRCriteria, NATIVE_RATIOS
from ...descriptors_report import RequirementSet, CapabilitySet
from pops.params.use_sites import ParamUse, resolve_param_use
from pops.runtime_environment import validate_amr_refinement_ratio


_LAYOUT_REPORT_SCHEMA_VERSION = 1


def _availability_dict(status: Any) -> dict:
    return {
        "status": status.status,
        "ok": status.ok,
        "reason": status.reason,
        "missing": list(status.missing),
        "alternatives": list(status.alternatives),
    }


def _native_layout_report(features: Any) -> dict:
    from pops._capabilities import native_capability_report

    report = native_capability_report()
    wanted = set(features)
    return {
        "schema_version": report.schema_version,
        "abi_version": report.abi_version,
        "target": report.target,
        "abi_key": report.abi_key,
        "platform": report.platform,
        "routes": [row.to_dict() for row in report.routes if row.feature in wanted],
    }


def _layout_inspect_dict(layout: Any, *, native_features: Any, amr_report: Any = None) -> dict:
    status = layout.available()
    info = {
        "schema_version": _LAYOUT_REPORT_SCHEMA_VERSION,
        "report_type": "layout_inspection",
        "name": layout.name,
        "category": layout.category,
        "native_id": layout.native_id,
        "options": layout.options(),
        "requirements": layout.requirements().to_dict(),
        "capabilities": layout.capabilities().to_dict(),
        "available": _availability_dict(status),
        "native_capabilities": _native_layout_report(native_features),
    }
    limitations = [
        {"feature": row["route_id"], "status": row["status"], "reason": row["reason"]}
        for row in info["native_capabilities"]["routes"]
        if row["status"] != "available"
    ]
    if not status.ok:
        limitations.append({
            "feature": "layout:%s" % info["capabilities"].get("layout", layout.name),
            "status": status.status,
            "reason": status.reason,
        })
    info["limitations"] = limitations
    if amr_report is not None:
        info["amr_report"] = amr_report.to_dict()
    return info


class Uniform(MeshDescriptor):
    """A single-level (uniform) mesh layout.

    ``refine=`` is NOT a supported single-level feature: a :class:`~pops.mesh.amr.Refine` /
    :class:`~pops.mesh.amr.TagUnion` attached here has no level to refine onto and would be a
    silently-ignored criterion if it were just dropped (Spec 5 sec.8.6 / ADC-589 / ADC-555). It
    is carried on the descriptor ONLY so :meth:`pops.Case.validate` can see it and refuse the
    problem by default; the explicit escape is ``ignore_amr=pops.mesh.amr.IgnoreAMRCriteria()``.
    """

    category = "layout"

    def __init__(self, mesh: Any, embedded_boundary: Any = None, refine: Any = None,
                 ignore_amr: Any = None) -> None:
        if ignore_amr is not None and not isinstance(ignore_amr, IgnoreAMRCriteria):
            raise TypeError(
                "Uniform(ignore_amr=...) accepts only the typed "
                "pops.mesh.amr.IgnoreAMRCriteria() marker, got %r; the escape must be "
                "the explicit descriptor, never a truthy value" % (ignore_amr,))
        self.mesh = mesh
        self.embedded_boundary = embedded_boundary
        self.refine = refine
        self.ignore_amr = ignore_amr

    def options(self) -> dict:
        opt = {"mesh": self.mesh.name}
        if self.embedded_boundary is not None:
            opt["embedded_boundary"] = self.embedded_boundary.name
        if self.refine is not None:
            opt["refine"] = self.refine.name
            opt["ignore_amr"] = self.ignore_amr is not None
        return opt

    def capabilities(self) -> Any:
        return CapabilitySet({"layout": "uniform", "levels": 1, "supports_amr": False})

    def resolve_for_case(self, resolver: Any) -> Uniform:
        """Authenticate optional declaration leaves through the common layout protocol."""
        if not callable(resolver):
            raise TypeError("Uniform.resolve_for_case requires a callable Handle resolver")
        refine = self.refine
        if refine is not None:
            protocol = getattr(refine, "resolve_references", None)
            if not callable(protocol):
                raise TypeError("Uniform.refine must implement resolve_references(resolver)")
            refine = protocol(resolver)
        return type(self)(
            mesh=self.mesh,
            embedded_boundary=self.embedded_boundary,
            refine=refine,
            ignore_amr=self.ignore_amr,
        )

    def validate(self, context: Any = None) -> bool:
        if self.refine is not None and self.ignore_amr is None:
            raise ValueError(
                "Uniform layout cannot consume AMR refinement criteria; remove the criterion, "
                "choose AMR, or declare IgnoreAMRCriteria() explicitly"
            )
        if self.refine is not None:
            self.refine.validate(context)
        return super().validate(context)

    def inspect(self) -> dict:
        from pops import inspect_amr

        return _layout_inspect_dict(
            self,
            native_features=("layout:Uniform", "layout:AMR", "mesh:2d_storage_arithmetic"),
            amr_report=inspect_amr(self))


class AMR(MeshDescriptor):
    """An adaptively refined mesh layout (Spec 5 sec.5.10 / sec.8.5).

    ``AMR(base=mesh, max_levels=2, ratio=2, regrid=RegridEvery(20),
    patches=PatchLayout(...), refine=TagUnion(...), nesting=ProperNesting(...),
    checkpoint=CheckpointPolicy(...))``.

    The resolved hierarchy carries any positive level count.  Ratio support is a transfer-kernel
    capability (currently ratio 2); resource policy, not a hardcoded DSL constant, limits depth.
    """

    category = "layout"

    def __init__(self, base: Any, max_levels: Any = 2, ratio: Any = 2, regrid: Any = None,
                 patches: Any = None, refine: Any = None, nesting: Any = None,
                 checkpoint: Any = None, output: Any = None, clustering: Any = None) -> None:
        self.base = base
        self.max_levels = int(resolve_param_use(
            max_levels, ParamUse.AMR_HIERARCHY, where="AMR(max_levels=)"))
        self.ratio = int(resolve_param_use(
            ratio, ParamUse.AMR_HIERARCHY, where="AMR(ratio=)"))
        self.regrid = regrid
        self.patches = patches
        self.refine = refine
        self.nesting = nesting
        self.checkpoint = checkpoint
        self.output = output
        # ADC-616: optional pops.mesh.amr.PatchClustering(...) tuning the Berger-Rigoutsos layout.
        # None -> the native ClusterParams default (bit-identical).
        self.clustering = clustering

    def options(self) -> dict:
        return {"base": self.base.name, "max_levels": self.max_levels, "ratio": self.ratio,
                "regrid": self.regrid.name if self.regrid else None,
                "refine": self.refine.name if self.refine else None}

    def capabilities(self) -> Any:
        base_capabilities = self.base.capabilities().to_dict()
        return CapabilitySet({"layout": "amr", "max_levels": self.max_levels,
                              "ratio": self.ratio, "dim": base_capabilities.get("dim"),
                              "supports_amr": True})

    def requirements(self) -> Any:
        return RequirementSet({"amr_runtime": True,
                               "reflux": True,
                               "tag_reduction": True})

    def resolve_for_case(self, resolver: Any) -> AMR:
        """Authenticate every declaration-bearing policy through one descriptor protocol."""
        if not callable(resolver):
            raise TypeError("AMR.resolve_for_case requires a callable Handle resolver")

        def resolved(value: Any) -> Any:
            if value is None:
                return None
            protocol = getattr(value, "resolve_references", None)
            return protocol(resolver) if callable(protocol) else value

        refine = resolved(self.refine)
        if refine is not None and not getattr(refine, "references_authenticated", False):
            raise ValueError("AMR refinement references were not authenticated")
        return type(self)(
            base=self.base,
            max_levels=self.max_levels,
            ratio=self.ratio,
            regrid=self.regrid,
            patches=self.patches,
            refine=refine,
            nesting=self.nesting,
            checkpoint=self.checkpoint,
            output=resolved(self.output),
            clustering=self.clustering,
        )

    def available(self, context: Any = None) -> Any:
        if self.ratio not in NATIVE_RATIOS:
            return Availability.no(
                "AMR(ratio=%d) is not supported by the current native AMR route "
                "(supported ratios: %s)" % (self.ratio, ", ".join(map(str, NATIVE_RATIOS))),
                alternatives=["AMR(ratio=2)"])
        return Availability.yes()

    def validate(self, context: Any = None) -> Any:
        if self.max_levels < 1:
            raise ValueError("AMR: max_levels must be >= 1")
        validate_amr_refinement_ratio(self.ratio, where="AMR")
        # Validate the attached policies, then the route availability.
        for policy in (self.regrid, self.patches, self.refine, self.nesting,
                       self.checkpoint, self.output, self.clustering):
            if policy is not None and hasattr(policy, "validate"):
                policy.validate(context)
        return super().validate(context)

    def inspect(self) -> dict:
        from pops import inspect_amr

        return _layout_inspect_dict(
            self,
            native_features=("layout:AMR", "amr:refinement_ratio",
                             "mesh:2d_storage_arithmetic"),
            amr_report=inspect_amr(self))


from ..layout_plan import (  # noqa: E402
    LayoutAssignment,
    LayoutHandle,
    LayoutLevel,
    LayoutMappingProvider,
    LayoutMappingRequirement,
    LayoutPlan,
    LayoutPlanBuilder,
    NormalizedLayout,
    ResolvedLayoutMapping,
    normalize_layout,
    normalize_layout_plan,
)


__all__ = [
    "Uniform", "AMR", "LayoutAssignment", "LayoutHandle", "LayoutLevel",
    "LayoutMappingProvider", "LayoutMappingRequirement", "LayoutPlan", "LayoutPlanBuilder",
    "NormalizedLayout", "ResolvedLayoutMapping", "normalize_layout", "normalize_layout_plan",
]
