"""Public mesh-organization descriptors.

A layout owns mesh structure only.  Physics, numerical methods, temporal programs and runtime
controls remain separate authorities.
"""
from __future__ import annotations

import json
from typing import Any

from pops.descriptors_report import CapabilitySet, RequirementSet
from pops.descriptors import Availability
from pops.mesh._descriptor import MeshDescriptor
from pops.mesh._layout_plan_contracts import NormalizedGeometry
from pops.amr import IgnoreAMRCriteria, PatchLayout


_LAYOUT_REPORT_SCHEMA_VERSION = 1


def _detached_json(value: Any) -> Any:
    """Detach canonical data even when providers expose immutable Mapping/tuple containers."""
    from pops._frozen_data import thaw_data

    return json.loads(json.dumps(
        thaw_data(value), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ))


def _availability_dict(status: Any) -> dict[str, Any]:
    return {
        "status": status.status,
        "ok": status.ok,
        "reason": status.reason,
        "missing": list(status.missing),
        "alternatives": list(status.alternatives),
    }


def _native_layout_report(features: Any) -> dict[str, Any]:
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


def _layout_inspect_dict(layout: Any, *, native_features: Any, amr_report: Any = None) -> dict[str, Any]:
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


def _delegated_geometry(value: Any, *, where: str) -> NormalizedGeometry:
    projection = getattr(value, "normalized_geometry", None)
    if not callable(projection):
        raise TypeError("%s must implement normalized_geometry()" % where)
    result = projection()
    if type(result) is not NormalizedGeometry:
        raise TypeError("%s normalized_geometry() must return an exact NormalizedGeometry" % where)
    return result


class Uniform(MeshDescriptor):
    """A single-level layout; AMR criteria need an explicit ignore marker."""

    category = "layout"

    def __init__(self, mesh: Any, embedded_boundary: Any = None, refine: Any = None,
                 ignore_amr: Any = None) -> None:
        if ignore_amr is not None and not isinstance(ignore_amr, IgnoreAMRCriteria):
            raise TypeError(
                "Uniform(ignore_amr=...) accepts only the typed "
                "pops.amr.IgnoreAMRCriteria() marker, got %r; the escape must be "
                "the explicit descriptor, never a truthy value" % (ignore_amr,))
        self.mesh = mesh
        self.embedded_boundary = embedded_boundary
        self.refine = refine
        self.ignore_amr = ignore_amr
        self._embedded_boundary_plan = (
            None if embedded_boundary is None else self._resolve_embedded_boundary()
        )

    def options(self) -> dict[str, Any]:
        options = {"mesh": self.mesh.name}
        if self._embedded_boundary_plan is not None:
            options["embedded_boundary"] = self._normalized_embedded_boundary()
        if self.refine is not None:
            options["refine"] = self.refine.name
            options["ignore_amr"] = self.ignore_amr is not None
        return options

    def _normalized_embedded_boundary(self) -> dict[str, Any]:
        """Return detached signed runtime data captured when this layout was authored."""
        if self._embedded_boundary_plan is None:
            raise ValueError("Uniform layout has no embedded boundary")
        return _detached_json(self._embedded_boundary_plan)

    def _resolve_embedded_boundary(self) -> dict[str, Any]:
        """Resolve an extension geometry exactly once into deterministic signed data."""
        from pops.mesh.geometry import Disc, EmbeddedBoundary
        from pops.mesh.masks import lower_transport_mask, transport_mask_thresholds

        embedded = self.embedded_boundary
        if not isinstance(embedded, EmbeddedBoundary):
            raise TypeError(
                "Uniform.embedded_boundary must be a pops.mesh.geometry.EmbeddedBoundary"
            )
        frame = getattr(self.mesh, "frame", None)

        def project() -> dict[str, Any]:
            from pops.boundary.embedded import lower_embedded_boundary_flux

            level_set = embedded.level_set(frame)
            mode = lower_transport_mask(embedded.transport)
            if mode == "cutcell" and type(embedded.domain) is not Disc:
                raise NotImplementedError(
                    "CutCell is not a generic LevelSet route: arbitrary analytic/CSG geometry "
                    "requires true face apertures, cell-intersection volumes and a typed wall-flux "
                    "provider. Use Staircase for generic embedded geometry until that complete "
                    "native route exists."
                )
            thresholds = transport_mask_thresholds(embedded.transport)
            if not isinstance(thresholds, dict) or any(
                key not in {"kappa_min", "face_open_eps", "cut_theta_min"}
                for key in thresholds
            ):
                raise TypeError(
                    "embedded transport thresholds must use only kappa_min, "
                    "face_open_eps and cut_theta_min"
                )
            return {
                "schema_version": 1,
                "level_set": level_set.to_data(),
                "boundary": {"provider": lower_embedded_boundary_flux(embedded.boundary)},
                "transport": {
                    "mode": mode,
                    "kappa_min": thresholds.get("kappa_min", 0.0),
                    "face_open_eps": thresholds.get("face_open_eps", 0.0),
                    "cut_theta_min": thresholds.get("cut_theta_min", 0.0),
                },
            }

        first = project()
        second = project()
        if first != second:
            raise ValueError(
                "embedded geometry and transport providers must lower deterministically"
            )
        return _detached_json(first)

    def semantic_data(self) -> dict[str, Any]:
        return {
            "kind": "uniform",
            "mesh": self.mesh,
            "embedded_boundary": (
                None if self._embedded_boundary_plan is None
                else self._normalized_embedded_boundary()
            ),
            "refinement": self.refine,
            "ignore_amr": self.ignore_amr is not None,
        }

    def normalized_geometry(self) -> NormalizedGeometry:
        return _delegated_geometry(self.mesh, where="Uniform.mesh")

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet({
            "layout": "uniform",
            "levels": 1,
            "supports_amr": False,
            "transition_ratios": [],
        })

    def resolve_for_case(self, resolver: Any) -> Uniform:
        if not callable(resolver):
            raise TypeError("Uniform.resolve_for_case requires a callable Handle resolver")
        refine = self.refine
        if refine is not None:
            protocol = getattr(refine, "resolve_references", None)
            if not callable(protocol):
                raise TypeError("Uniform.refine must implement resolve_references(resolver)")
            refine = protocol(resolver)
        return type(self)._from_captured_plan(
            mesh=self.mesh,
            embedded_boundary_plan=self._embedded_boundary_plan,
            refine=refine,
            ignore_amr=self.ignore_amr,
        )

    @classmethod
    def _from_captured_plan(
        cls,
        *,
        mesh: Any,
        embedded_boundary_plan: Any,
        refine: Any,
        ignore_amr: Any,
    ) -> Uniform:
        """Build a resolved layout without consulting the authoring EB provider again."""

        result = object.__new__(cls)
        object.__setattr__(result, "mesh", mesh)
        # The resolved descriptor owns only the detached signed plan.  Keeping the live provider
        # here would make a later validation or copy capable of re-entering mutable Python code.
        object.__setattr__(result, "embedded_boundary", None)
        object.__setattr__(result, "refine", refine)
        object.__setattr__(result, "ignore_amr", ignore_amr)
        captured = None
        if embedded_boundary_plan is not None:
            captured = _detached_json(embedded_boundary_plan)
        object.__setattr__(result, "_embedded_boundary_plan", captured)
        return result

    def validate(self, context: Any = None) -> bool:
        if self._embedded_boundary_plan is not None:
            # Reparse the detached data rather than recalling Geometry.level_set() or any
            # TransportMask extension.  Bind performs the same strict schema authentication.
            from pops.mesh.geometry import LevelSet

            captured = self._normalized_embedded_boundary()
            if set(captured) != {"schema_version", "level_set", "boundary", "transport"} \
                    or captured.get("schema_version") != 1:
                raise TypeError("captured Uniform embedded-boundary plan is malformed")
            LevelSet.from_data(captured["level_set"])
            if captured["boundary"] != {"provider": "zero_flux"}:
                raise TypeError("captured Uniform embedded boundary flux is unsupported")
            transport = captured["transport"]
            if not isinstance(transport, dict) or set(transport) != {
                "mode", "kappa_min", "face_open_eps", "cut_theta_min",
            }:
                raise TypeError("captured Uniform embedded transport plan is malformed")
        if self.refine is not None and self.ignore_amr is None:
            raise ValueError(
                "Uniform layout cannot consume AMR refinement criteria; remove the criterion, "
                "choose AMR, or declare IgnoreAMRCriteria() explicitly"
            )
        if self.refine is not None:
            self.refine.validate(context)
        return super().validate(context)

    def inspect(self) -> dict[str, Any]:
        from pops._capabilities_inspect import _layout_amr_report

        return _layout_inspect_dict(
            self,
            native_features=("layout:Uniform", "layout:AMR", "mesh:2d_storage_arithmetic"),
            amr_report=_layout_amr_report(self),
        )

    def _amr_report(self) -> Any:
        from pops._capabilities_inspect import AmrReport, _native_amr_context

        native_depth, native_ratios, _ = _native_amr_context()
        return AmrReport(
            layout="uniform", max_levels=1, ratio=1,
            native_max_levels=native_depth, native_ratios=native_ratios,
            available="yes",
            limitations=["a Uniform layout is single-level: no refinement, regrid or reflux"],
            requirements={}, policies=[],
        )


_AMR_AUTHORITY_PROTOCOLS = {
    "hierarchy": ("to_data",),
    "patch_layout": ("to_data",),
    "tagging": ("inspect", "resolve_references", "resolve"),
    "regrid": ("to_data",),
    "transfer": ("inspect", "resolve_references", "resolve"),
    "execution": ("to_data", "runtime_execution_data"),
}

_AMR_PROVIDER_PROTOCOL = (
    "inspect", "resolve_references", "lower_amr_provider",
)


def _authority_data(value: Any, slot: str) -> dict[str, Any]:
    """Authenticate one small AMR authority protocol without naming an implementation class."""
    for method in _AMR_AUTHORITY_PROTOCOLS[slot]:
        if not callable(getattr(value, method, None)):
            raise TypeError("AMR.%s must implement %s()" % (slot, method))
    projection = value.inspect if slot in {"tagging", "transfer"} else value.to_data
    data = projection()
    if not isinstance(data, dict):
        raise TypeError("AMR.%s identity projection must be a mapping" % slot)
    authority_type = data.get("authority_type")
    if not isinstance(authority_type, str) or not authority_type:
        raise ValueError("AMR.%s identity must authenticate authority_type" % slot)
    try:
        json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("AMR.%s identity must be strict JSON data" % slot) from exc
    return data


def _provider_data(value: Any, slot: str) -> dict[str, Any]:
    """Authenticate one open AMR provider through its small immutable protocol."""
    for method in _AMR_PROVIDER_PROTOCOL:
        if not callable(getattr(value, method, None)):
            raise TypeError("AMR.%s must implement %s()" % (slot, method))
    first, second = value.inspect(), value.inspect()
    if not isinstance(first, dict) or first != second \
            or first.get("schema_version") != 1 \
            or not isinstance(first.get("provider_type"), str) \
            or not isinstance(first.get("provider_identity"), str):
        raise TypeError("AMR.%s must expose one deterministic canonical provider identity" % slot)
    try:
        json.dumps(first, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError("AMR.%s provider identity must be strict JSON data" % slot) from exc
    return first


def _patch_layout_data(value: Any) -> dict[str, Any]:
    """Authenticate the backend-neutral coarse-patch authority through its data protocol."""
    data = _authority_data(value, "patch_layout")
    if data != _authority_data(value, "patch_layout"):
        raise TypeError("AMR.patch_layout to_data() must be deterministic")
    expected = {
        "schema_version", "authority_type", "distribute_coarse", "coarse_max_grid",
    }
    if set(data) != expected \
            or type(data.get("schema_version")) is not int \
            or data["schema_version"] != 1 \
            or data.get("authority_type") != "amr_patch_layout" \
            or type(data.get("distribute_coarse")) is not bool:
        raise TypeError("AMR.patch_layout must expose the exact amr_patch_layout schema-v1")
    coarse_max_grid = data["coarse_max_grid"]
    if coarse_max_grid is not None:
        if type(coarse_max_grid) is not int:
            raise TypeError("AMR.patch_layout coarse_max_grid must be None or an exact integer")
        if coarse_max_grid < 1:
            raise ValueError("AMR.patch_layout coarse_max_grid must be positive when provided")
    return data


def _load_balance_data(value: Any) -> dict[str, Any]:
    """Authenticate one open load-balance provider without naming its implementation."""
    from pops.amr._load_balance_contract import load_balance_provider_data

    return load_balance_provider_data(value)


class AMR(MeshDescriptor):
    """One complete adaptive-layout authority.

    Accuracy, formal order and halo depth are deliberately absent: they are derived from the
    selected numerical and transfer providers during resolution.
    """

    category = "layout"

    def __init__(
        self,
        *,
        grid: Any,
        hierarchy: Any,
        tagging: Any,
        regrid: Any,
        transfer: Any,
        execution: Any,
        patch_layout: Any = None,
        load_balance: Any = None,
        tagger: Any = None,
        clustering: Any = None,
    ) -> None:
        # Structural snapshots consume ``options()``.  Keeping authorities private prevents the
        # generic snapshotter from recursively treating Schedule implementation helpers as public
        # authoring state in addition to that canonical projection.
        self._grid = grid
        self._hierarchy = hierarchy
        self._tagging = tagging
        self._regrid = regrid
        self._transfer = transfer
        self._execution = execution
        self._patch_layout = PatchLayout() if patch_layout is None else patch_layout
        if load_balance is None or tagger is None or clustering is None:
            from pops.lib.amr import BergerRigoutsos, SpaceFillingCurve, SymbolicTagger

            load_balance = SpaceFillingCurve() if load_balance is None else load_balance
            tagger = SymbolicTagger() if tagger is None else tagger
            clustering = BergerRigoutsos() if clustering is None else clustering
        self._load_balance = load_balance
        self._tagger = tagger
        self._clustering = clustering

    @property
    def grid(self) -> Any:
        return self._grid

    @property
    def hierarchy(self) -> Any:
        return self._hierarchy

    @property
    def tagging(self) -> Any:
        return self._tagging

    @property
    def regrid(self) -> Any:
        return self._regrid

    @property
    def transfer(self) -> Any:
        return self._transfer

    @property
    def execution(self) -> Any:
        return self._execution

    @property
    def patch_layout(self) -> Any:
        return self._patch_layout

    @property
    def load_balance(self) -> Any:
        return self._load_balance

    @property
    def tagger(self) -> Any:
        return self._tagger

    @property
    def clustering(self) -> Any:
        return self._clustering

    def _validate_authorities(self) -> None:
        authorities = {
            "hierarchy": self.hierarchy, "tagging": self.tagging,
            "regrid": self.regrid, "transfer": self.transfer,
            "execution": self.execution,
        }
        data = {slot: _authority_data(value, slot) for slot, value in authorities.items()}
        data["patch_layout"] = _patch_layout_data(self.patch_layout)
        _load_balance_data(self.load_balance)
        _provider_data(self.tagger, "tagger")
        _provider_data(self.clustering, "clustering")
        for method in ("validate", "capabilities", "requirements", "options", "to_dict"):
            if not callable(getattr(self.grid, method, None)):
                raise TypeError("AMR.grid must implement %s()" % method)
        hierarchy = data["hierarchy"]
        levels = hierarchy.get("max_levels")
        ratios = hierarchy.get("ratios")
        if isinstance(levels, bool) or not isinstance(levels, int) or levels < 1:
            raise ValueError("AMR hierarchy requires at least one level")
        if not isinstance(ratios, list) or len(ratios) != levels - 1 or any(
                isinstance(ratio, bool) or not isinstance(ratio, int) or ratio < 2
                for ratio in ratios):
            raise ValueError("AMR hierarchy identity must preserve every transition ratio")

    def requirements(self) -> RequirementSet:
        return RequirementSet({
            "amr_runtime": True,
            "reflux": True,
            "transactional_regrid": True,
            "tag_reduction": True,
        })

    def capabilities(self) -> CapabilitySet:
        self._validate_authorities()
        grid = self.grid.capabilities().to_dict()
        hierarchy = _authority_data(self.hierarchy, "hierarchy")
        execution = _authority_data(self.execution, "execution")
        return CapabilitySet({
            "layout": "amr",
            "supports_amr": True,
            "dim": grid.get("dim"),
            "max_levels": hierarchy["max_levels"],
            "transition_ratios": list(hierarchy["ratios"]),
            "execution": execution["mode"],
        })

    def options(self) -> dict[str, Any]:
        self._validate_authorities()
        return {
            "grid": self.grid.to_dict(),
            "hierarchy": self.hierarchy.to_data(),
            "tagging": self.tagging.inspect(),
            "regrid": self.regrid.to_data(),
            "transfer": self.transfer.inspect(),
            "execution": self.execution.to_data(),
            "patch_layout": _patch_layout_data(self.patch_layout),
            "load_balance": _load_balance_data(self.load_balance),
            "tagger": self.tagger.inspect(),
            "clustering": self.clustering.inspect(),
        }

    def _summary(self) -> str:
        """Keep descriptor printing structural; detailed authority data belongs to ``inspect``."""
        hierarchy = _authority_data(self.hierarchy, "hierarchy")
        execution = _authority_data(self.execution, "execution")
        grid_name = getattr(self.grid, "name", type(self.grid).__name__)
        return "grid=%s, max_levels=%s, transition_ratios=%s, execution=%s" % (
            grid_name,
            hierarchy["max_levels"],
            hierarchy["ratios"],
            execution["mode"],
        )

    def semantic_data(self) -> dict[str, Any]:
        """Scientific adaptive structure without backend, ABI or runtime availability facts."""
        return {"kind": "amr", **self.options()}

    def available(self, context: Any = None) -> Availability:
        del context
        self._validate_authorities()
        # Provider availability is resolved later. The algorithm-neutral descriptor preserves the
        # exact plan even when the installed native provider will fail closed before bind.
        return Availability.yes()

    def validate(self, context: Any = None) -> bool:
        self._validate_authorities()
        self.grid.validate(context)
        return super().validate(context)

    def resolve_for_case(self, resolver: Any) -> AMR:
        """Return a detached descriptor with every declaration Handle authenticated."""
        if not callable(resolver):
            raise TypeError("AMR.resolve_for_case requires a callable Handle resolver")

        def resolved(value: Any) -> Any:
            if getattr(value, "is_resolved", False):
                return value
            return resolver(value)

        return type(self)(
            grid=self.grid,
            hierarchy=self.hierarchy,
            tagging=self.tagging.resolve_references(resolved),
            regrid=self.regrid,
            transfer=self.transfer.resolve_references(resolved),
            execution=self.execution,
            patch_layout=self.patch_layout,
            load_balance=self.load_balance,
            tagger=self.tagger.resolve_references(resolved),
            clustering=self.clustering.resolve_references(resolved),
        )

    def resolve_amr_authorities(self, context: Any) -> Any:
        """Resolve via the same open layout protocol available to extension descriptors."""
        from pops.amr._resolution import resolve_amr_authorities

        return resolve_amr_authorities(
            hierarchy=self.hierarchy,
            tagging=self.tagging,
            regrid=self.regrid,
            transfer=self.transfer,
            execution=self.execution,
            patch_layout=self.patch_layout,
            load_balance=self.load_balance,
            tagger=self.tagger,
            clustering=self.clustering,
            context=context,
        )

    def runtime_layout_data(self) -> dict[str, Any]:
        """Project exact geometry/cadence/execution facts for a runtime provider."""
        self._validate_authorities()
        return {
            "schema_version": 1,
            "layout_type": "adaptive_cartesian",
            "grid": self.grid.to_dict(),
            "regrid": self.regrid.to_data(),
            "execution": self.execution.to_data(),
            "load_balance": _load_balance_data(self.load_balance),
        }

    def normalized_geometry(self) -> NormalizedGeometry:
        return _delegated_geometry(self.grid, where="AMR.grid")

    def inspect(self) -> dict[str, Any]:
        from pops._capabilities_inspect import _layout_amr_report

        return _layout_inspect_dict(
            self,
            native_features=("layout:AMR", "amr:refinement_ratio", "mesh:2d_storage_arithmetic"),
            amr_report=_layout_amr_report(self),
        )

    def _amr_report(self) -> Any:
        from pops._capabilities_inspect import AmrReport, _native_amr_context

        native_depth, native_ratios, native_note = _native_amr_context()
        status = self.available()
        hierarchy = _authority_data(self.hierarchy, "hierarchy")
        ratios = tuple(hierarchy["ratios"])
        ratio = ratios[0] if len(set(ratios)) == 1 else None
        limitations = [native_note]
        if not status.ok and status.reason:
            limitations.append(status.reason)
        return AmrReport(
            layout="amr",
            max_levels=hierarchy["max_levels"],
            ratio=ratio,
            native_max_levels=native_depth,
            native_ratios=native_ratios,
            available=status.status,
            limitations=limitations,
            requirements=self.requirements().to_dict(),
            policies=[],
        )


__all__ = ["AMR", "Uniform"]
