"""Ready elliptic field-solver providers built on the generic field protocol.

This is the only Python module that interprets GeometricMG, CompositeFAC and FFT option schemas.
The registry, field compiler and runtime installers consume opaque authenticated bindings.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any


_MG_KEYS = {
    "rel_tol", "abs_tol", "max_cycles", "min_coarse", "pre_smooth",
    "post_smooth", "bottom_sweeps", "coarse_threshold",
}
_FAC_KEYS = {
    "max_iters", "fine_sweeps", "rel_tol", "abs_tol", "coarse_rel_tol",
    "coarse_abs_tol", "coarse_cycles", "verbose",
}
_FAC_DEFAULTS = {
    "max_iters": 30,
    "fine_sweeps": 400,
    "rel_tol": 1.0e-9,
    "abs_tol": 0.0,
    "coarse_rel_tol": 1.0e-12,
    "coarse_abs_tol": 0.0,
    "coarse_cycles": 100,
    "verbose": False,
}
_COMPOSITE_HIERARCHY_POLICY = "pops.field-hierarchy.composite"
_LEVEL_LOCAL_HIERARCHY_POLICY = "pops.field-hierarchy.level-local"


def _hierarchy_policy_identity(facts: Any, *, where: str) -> str:
    authority = facts.hierarchy
    expected = {"policy_id", "interface_version", "option_schema", "options"}
    if not isinstance(authority, Mapping) or set(authority) != expected:
        raise ValueError("%s hierarchy policy authority has an invalid shape" % where)
    policy_id = authority.get("policy_id")
    if type(policy_id) is not str or not policy_id:
        raise TypeError("%s hierarchy policy requires an exact identity" % where)
    if (
        type(authority.get("interface_version")) is not int
        or authority["interface_version"] < 1
    ):
        raise TypeError("%s hierarchy policy requires a positive exact interface version" % where)
    if type(authority.get("option_schema")) is not str or not authority["option_schema"]:
        raise TypeError("%s hierarchy policy requires an exact option schema" % where)
    if not isinstance(authority.get("options"), Mapping):
        raise TypeError("%s hierarchy policy options must be a mapping" % where)
    if policy_id in (_COMPOSITE_HIERARCHY_POLICY, _LEVEL_LOCAL_HIERARCHY_POLICY) and (
        authority["interface_version"] != 1
        or authority["option_schema"] != "pops.field-hierarchy.options.empty@1"
        or dict(authority["options"])
    ):
        raise ValueError("%s builtin hierarchy policy authority is inconsistent" % where)
    return policy_id


def _native_real(value: Any, *, where: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s must be a finite real" % where)
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result == 0.0):
        raise ValueError("%s is outside its exact native domain" % where)
    return result


def _native_int(value: Any, *, where: str, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError("%s must be an exact integer" % where)
    if value < minimum or value > (1 << 31) - 1:
        raise ValueError("%s is outside its exact native domain" % where)
    return value


def _resolved_mg_options(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _MG_KEYS:
        raise TypeError("%s must contain the exact geometric-MG option schema" % where)
    return {
        "rel_tol": _native_real(value["rel_tol"], where=where + ".rel_tol", positive=True),
        "abs_tol": _native_real(value["abs_tol"], where=where + ".abs_tol"),
        "max_cycles": _native_int(value["max_cycles"], where=where + ".max_cycles", minimum=1),
        "min_coarse": _native_int(value["min_coarse"], where=where + ".min_coarse", minimum=1),
        "pre_smooth": _native_int(value["pre_smooth"], where=where + ".pre_smooth", minimum=0),
        "post_smooth": _native_int(
            value["post_smooth"], where=where + ".post_smooth", minimum=0
        ),
        "bottom_sweeps": _native_int(
            value["bottom_sweeps"], where=where + ".bottom_sweeps", minimum=0
        ),
        "coarse_threshold": _native_int(
            value["coarse_threshold"], where=where + ".coarse_threshold", minimum=0
        ),
    }


def _resolved_fac_options(value: Any, *, where: str) -> dict[str, Any]:
    if value is None:
        return dict(_FAC_DEFAULTS)
    if not isinstance(value, Mapping) or set(value) != _FAC_KEYS:
        raise TypeError("%s must contain the exact composite-FAC option schema" % where)
    result = dict(_FAC_DEFAULTS)
    for name in ("max_iters", "fine_sweeps", "coarse_cycles"):
        if value[name] is not None:
            result[name] = _native_int(
                value[name], where="%s.%s" % (where, name), minimum=1
            )
    for name in ("rel_tol", "coarse_rel_tol"):
        if value[name] is not None:
            lowered = _native_real(
                value[name], where="%s.%s" % (where, name), positive=True
            )
            if lowered >= 1.0:
                raise ValueError("%s.%s must be in (0, 1)" % (where, name))
            result[name] = lowered
    for name in ("abs_tol", "coarse_abs_tol"):
        if value[name] is not None:
            result[name] = _native_real(value[name], where="%s.%s" % (where, name))
    if type(value["verbose"]) is not bool:
        raise TypeError("%s.verbose must be an exact bool" % where)
    result["verbose"] = value["verbose"]
    return result


def _resolution(native: Mapping[str, Any], topology: Mapping[str, Any]) -> Any:
    from pops.fields._prepared_field_solver_registry import PreparedFieldSolverResolution

    return PreparedFieldSolverResolution(native, topology)


def _builtin_topology(facts: Any) -> dict[str, Any]:
    topology_identity = facts.layout.get("topology_identity")
    if type(topology_identity) is not str or not topology_identity:
        raise ValueError("prepared field solver layout lost its exact topology identity")
    return {
        "provider_id": "pops.field-topology.rectangular-cell-graph",
        "version": 1,
        "topology_identity": topology_identity,
    }


def _geometric_mg_resolver(
    options: Mapping[str, Any], facts: Any, where: str,
) -> Any:
    if set(options) != {"mg", "fac"}:
        raise TypeError("%s geometric-MG provider options have an invalid shape" % where)
    mg = _resolved_mg_options(options["mg"], where=where + ".mg")
    fac_authored = options["fac"]
    if facts.target == "system":
        if fac_authored is not None:
            raise ValueError("%s composite-FAC options require an AMR hierarchy" % where)
        native = {
            "factory_route": "geometric_mg",
            "schema_identity": "pops.system.geometric-mg-options@1",
            "options": mg,
        }
    elif facts.target == "amr_system":
        fac = _resolved_fac_options(fac_authored, where=where + ".fac")
        native = {
            "factory_route": "geometric_mg",
            "schema_identity": "pops.amr.field-solver-options.geometric-mg@1",
            "options": {
                **{"mg.%s" % key: value for key, value in mg.items()},
                **{"fac.%s" % key: value for key, value in fac.items()},
            },
        }
    else:
        raise ValueError(
            "%s geometric-MG provider does not implement target %r"
            % (where, facts.target)
        )
    return _resolution(native, _builtin_topology(facts))


def _validate_geometric_mg(use: Any, where: str) -> None:
    facts = use.facts
    if use.options.get("fac") is not None and (
        facts.target != "amr_system"
        or _hierarchy_policy_identity(facts, where=where)
        != _COMPOSITE_HIERARCHY_POLICY
        or facts.layout.get("levels", 0) < 2
    ):
        raise ValueError(
            "%s authored CompositeFAC requires a composite multi-level AMR backend" % where
        )


def _install_configured(context: Any, binding: Any) -> None:
    context.install_configured(binding)


def _fft_resolver(options: Mapping[str, Any], facts: Any, where: str) -> Any:
    if set(options) != {"spectral"} or type(options["spectral"]) is not bool:
        raise TypeError("%s FFT provider requires one exact spectral bool" % where)
    return _resolution(
        {
            "factory_route": "fft_spectral" if options["spectral"] else "fft",
            "schema_identity": "pops.system.fft-options@1",
            "options": {"spectral": options["spectral"]},
        },
        _builtin_topology(facts),
    )


def _validate_fft(use: Any, where: str) -> None:
    facts = use.facts
    if facts.target != "system" or facts.layout.get("kind") != "uniform":
        raise ValueError("%s FFT provider requires a single uniform System layout" % where)
    if (
        _hierarchy_policy_identity(facts, where=where)
        != _LEVEL_LOCAL_HIERARCHY_POLICY
    ):
        raise ValueError("%s FFT provider requires a level-local hierarchy" % where)
    if facts.operator.get("screened"):
        raise ValueError("%s FFT provider does not implement a screened operator" % where)
    if facts.layout.get("embedded_boundary"):
        raise ValueError("%s FFT provider requires a full-material topology" % where)
    faces = facts.boundary.get("faces")
    if not isinstance(faces, tuple) or not faces or any(
        not isinstance(face, Mapping) or face.get("type") != "periodic" for face in faces
    ):
        raise ValueError("%s FFT provider requires fully periodic boundaries" % where)
    if facts.boundary.get("dynamic") or facts.boundary.get("dependent"):
        raise ValueError("%s FFT provider requires an immutable boundary contract" % where)
    if facts.nonlinear:
        raise ValueError("%s FFT provider cannot serve a nonlinear outer solve" % where)
    cells = facts.layout.get("cells")
    if (
        not isinstance(cells, tuple)
        or not cells
        or any(type(value) is not int or value < 1 or value & (value - 1) for value in cells)
    ):
        raise ValueError("%s FFT provider requires a power-of-two cell count on every axis" % where)


def _register_ready_providers() -> tuple[Any, Any]:
    # Lazy import preserves the solvers authoring layer's import-time DAG.  Concrete provider
    # registration occurs only when a descriptor is prepared for field lowering.
    from pops.fields._prepared_field_solver_registry import (
        PreparedFieldSolverProvider as Provider,
        PreparedFieldSolverUsePolicy as UsePolicy,
        register_prepared_field_solver_provider as register,
    )

    geometric = register(Provider(
        provider_id="pops.field-solver.geometric-mg",
        version=1,
        resolver_id="pops.field-solver.geometric-mg.resolve@1",
        installer_id="pops.field-solver.geometric-mg.install@1",
        use_policy=UsePolicy(
            "pops.field-solver.geometric-mg.use",
            1,
            {
                "targets": ("system", "amr_system"),
                "operators": ("poisson", "screened-poisson"),
                "hierarchy_policies": (
                    "pops.field-hierarchy.level-local@1",
                    "pops.field-hierarchy.composite@1",
                ),
            },
            _validate_geometric_mg,
        ),
        resolver=_geometric_mg_resolver,
        native_installer=_install_configured,
    ))
    fft = register(Provider(
        provider_id="pops.field-solver.fft",
        version=1,
        resolver_id="pops.field-solver.fft.resolve@1",
        installer_id="pops.field-solver.fft.install@1",
        use_policy=UsePolicy(
            "pops.field-solver.fft.use",
            1,
            {
                "targets": ("system",),
                "layout": "uniform-power-of-two",
                "boundary": "fully-periodic",
                "operator": "poisson",
            },
            _validate_fft,
        ),
        resolver=_fft_resolver,
        native_installer=_install_configured,
    ))
    return geometric, fft


_GEOMETRIC_MG_PROVIDER, _FFT_PROVIDER = _register_ready_providers()


def geometric_mg_field_solver_provider() -> Any:
    return _GEOMETRIC_MG_PROVIDER


def fft_field_solver_provider() -> Any:
    return _FFT_PROVIDER


__all__ = ["fft_field_solver_provider", "geometric_mg_field_solver_provider"]
