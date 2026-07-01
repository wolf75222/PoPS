"""pops.mesh.layouts -- how a mesh is organised for execution (Spec 5 sec.5.10).

A layout is NOT a backend and NOT a compile target: it says what mesh STRUCTURE the
runtime must materialise. ``Uniform`` is a single-level mesh; ``AMR`` is an adaptively
refined hierarchy whose policies come from :mod:`pops.mesh.amr`. Spec 5 (sec.8.5)
replaces ``target="amr_system"`` / ``AmrSystemTarget()`` with ``layout=AMR(...)``.

These are inert descriptors: they declare requirements / capabilities and answer
``available(context)`` so an unsupported route is refused before the runtime is touched.
"""
from .._descriptor import Availability, MeshDescriptor
from ..amr import NATIVE_MAX_LEVELS, NATIVE_RATIOS
from pops.runtime_environment import validate_amr_refinement_ratio


_LAYOUT_REPORT_SCHEMA_VERSION = 1


def _availability_dict(status):
    return {
        "status": status.status,
        "ok": status.ok,
        "reason": status.reason,
        "missing": list(status.missing),
        "alternatives": list(status.alternatives),
    }


def _native_layout_report(features):
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


def _layout_inspect_dict(layout, *, native_features, amr_report=None):
    status = layout.available()
    info = {
        "schema_version": _LAYOUT_REPORT_SCHEMA_VERSION,
        "report_type": "layout_inspection",
        "name": layout.name,
        "category": layout.category,
        "native_id": layout.native_id,
        "options": layout.options(),
        "requirements": layout.requirements(),
        "capabilities": layout.capabilities(),
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
    """A single-level (uniform) mesh layout."""

    category = "layout"

    def __init__(self, mesh, embedded_boundary=None):
        self.mesh = mesh
        self.embedded_boundary = embedded_boundary

    def options(self):
        opt = {"mesh": self.mesh.name}
        if self.embedded_boundary is not None:
            opt["embedded_boundary"] = self.embedded_boundary.name
        return opt

    def capabilities(self):
        return {"layout": "uniform", "levels": 1, "supports_amr": False}

    def inspect(self):
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

    Declares its limitations explicitly: the current native AMR route supports
    ``max_levels <= 2`` at ``ratio == 2`` (see :data:`pops.mesh.amr.NATIVE_MAX_LEVELS` /
    ``NATIVE_RATIOS``); a request beyond that is refused by :meth:`available` /
    :meth:`validate` with a clear message instead of being silently clamped.
    """

    category = "layout"

    def __init__(self, base, max_levels=2, ratio=2, regrid=None, patches=None,
                 refine=None, nesting=None, checkpoint=None, output=None):
        self.base = base
        self.max_levels = int(max_levels)
        self.ratio = int(ratio)
        self.regrid = regrid
        self.patches = patches
        self.refine = refine
        self.nesting = nesting
        self.checkpoint = checkpoint
        self.output = output

    def options(self):
        return {"base": self.base.name, "max_levels": self.max_levels, "ratio": self.ratio,
                "regrid": self.regrid.name if self.regrid else None,
                "refine": self.refine.name if self.refine else None}

    def capabilities(self):
        return {"layout": "amr", "max_levels": self.max_levels, "ratio": self.ratio,
                "supports_amr": True}

    def requirements(self):
        return {"amr_runtime": True,
                "reflux": True,
                "tag_reduction": True}

    def available(self, context=None):
        if self.max_levels > NATIVE_MAX_LEVELS:
            return Availability.no(
                "AMR(max_levels=%d) is not supported by the current native AMR route "
                "(supports max_levels=%d)" % (self.max_levels, NATIVE_MAX_LEVELS),
                alternatives=["AMR(max_levels=%d)" % NATIVE_MAX_LEVELS])
        if self.ratio not in NATIVE_RATIOS:
            return Availability.no(
                "AMR(ratio=%d) is not supported by the current native AMR route "
                "(supported ratios: %s)" % (self.ratio, ", ".join(map(str, NATIVE_RATIOS))),
                alternatives=["AMR(ratio=2)"])
        return Availability.yes()

    def validate(self, context=None):
        if self.max_levels < 1:
            raise ValueError("AMR: max_levels must be >= 1")
        validate_amr_refinement_ratio(self.ratio, where="AMR")
        # Validate the attached policies, then the route availability.
        for policy in (self.regrid, self.patches, self.refine, self.nesting,
                       self.checkpoint, self.output):
            if policy is not None and hasattr(policy, "validate"):
                policy.validate(context)
        return super().validate(context)

    def inspect(self):
        from pops import inspect_amr

        return _layout_inspect_dict(
            self,
            native_features=("layout:AMR", "amr:refinement_ratio",
                             "mesh:2d_storage_arithmetic"),
            amr_report=inspect_amr(self))


__all__ = ["Uniform", "AMR"]
