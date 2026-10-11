"""Ready elliptic field-solver providers built on the generic field protocol.

This is the only Python module that interprets CartesianCG, GeometricMG, CompositeFAC and FFT
option schemas.
The registry, field compiler and runtime installers consume opaque authenticated bindings.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any


_MG_KEYS = {
    "rel_tol",
    "abs_tol",
    "max_cycles",
    "min_coarse",
    "pre_smooth",
    "post_smooth",
    "bottom_sweeps",
    "coarse_threshold",
}
_CARTESIAN_CG_KEYS = {"rel_tol", "abs_tol", "max_iterations"}
_FAC_KEYS = {
    "max_iters",
    "fine_sweeps",
    "rel_tol",
    "abs_tol",
    "coarse_rel_tol",
    "coarse_abs_tol",
    "coarse_cycles",
    "verbose",
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
    if type(authority.get("interface_version")) is not int or authority["interface_version"] < 1:
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
        "post_smooth": _native_int(value["post_smooth"], where=where + ".post_smooth", minimum=0),
        "bottom_sweeps": _native_int(
            value["bottom_sweeps"], where=where + ".bottom_sweeps", minimum=0
        ),
        "coarse_threshold": _native_int(
            value["coarse_threshold"], where=where + ".coarse_threshold", minimum=0
        ),
    }


def _resolved_cartesian_cg_options(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _CARTESIAN_CG_KEYS:
        raise TypeError("%s must contain the exact Cartesian-CG option schema" % where)
    return {
        "rel_tol": _native_real(value["rel_tol"], where=where + ".rel_tol", positive=True),
        "abs_tol": _native_real(value["abs_tol"], where=where + ".abs_tol"),
        "max_iterations": _native_int(
            value["max_iterations"], where=where + ".max_iterations", minimum=1
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
            result[name] = _native_int(value[name], where="%s.%s" % (where, name), minimum=1)
    for name in ("rel_tol", "coarse_rel_tol"):
        if value[name] is not None:
            lowered = _native_real(value[name], where="%s.%s" % (where, name), positive=True)
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
    options: Mapping[str, Any],
    facts: Any,
    where: str,
) -> Any:
    if set(options) != {"mg", "fac"}:
        raise TypeError("%s geometric-MG provider options have an invalid shape" % where)
    mg = _resolved_mg_options(options["mg"], where=where + ".mg")
    fac_authored = options["fac"]
    if facts.target == "system":
        raise ValueError(
            "%s GeometricMG is reserved for the native MG/FAC AMR route; "
            "use pops.solvers.elliptic.CartesianCG() for a uniform System" % where
        )
    if facts.target == "amr_system":
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
            "%s geometric-MG provider does not implement target %r" % (where, facts.target)
        )
    return _resolution(native, _builtin_topology(facts))


def _cartesian_cg_resolver(
    options: Mapping[str, Any],
    facts: Any,
    where: str,
) -> Any:
    if facts.target != "system":
        raise ValueError(
            "%s CartesianCG implements only a uniform System; use GeometricMG() on AMR" % where
        )
    native = {
        "factory_route": "cartesian_cg",
        "schema_identity": "pops.system.cartesian-cg-options@1",
        "options": _resolved_cartesian_cg_options(options, where=where),
    }
    return _resolution(native, _builtin_topology(facts))


def _validate_cartesian_cg(use: Any, where: str) -> None:
    facts = use.facts
    if facts.target != "system" or facts.layout.get("kind") != "uniform":
        raise ValueError("%s CartesianCG requires one uniform System layout" % where)
    if facts.layout.get("levels") != 1 or facts.layout.get("adaptive"):
        raise ValueError("%s CartesianCG requires exactly one non-adaptive level" % where)
    if _hierarchy_policy_identity(facts, where=where) != _LEVEL_LOCAL_HIERARCHY_POLICY:
        raise ValueError("%s CartesianCG requires a level-local hierarchy" % where)
    if facts.operator.get("principal") != "scalar-laplacian" or facts.operator.get("screened"):
        raise ValueError("%s CartesianCG implements only constant-coefficient Poisson" % where)
    if facts.layout.get("embedded_boundary"):
        raise ValueError("%s CartesianCG requires a full-material Cartesian topology" % where)
    if facts.boundary.get("iterate_dependent") and not facts.nonlinear:
        raise ValueError(
            "%s iterate-dependent Cartesian boundary requires a prepared Newton-Krylov outer "
            "solve" % where
        )


def _validate_geometric_mg(use: Any, where: str) -> None:
    facts = use.facts
    hierarchy = _hierarchy_policy_identity(facts, where=where)
    levels = facts.layout.get("levels", 0)
    if use.options.get("fac") is not None and (
        facts.target != "amr_system" or hierarchy != _COMPOSITE_HIERARCHY_POLICY or levels < 2
    ):
        raise ValueError(
            "%s authored CompositeFAC requires a composite multi-level AMR backend" % where
        )
    if (
        facts.target == "amr_system"
        and levels > 1
        and hierarchy == _LEVEL_LOCAL_HIERARCHY_POLICY
        and facts.boundary.get("iterate_dependent")
    ):
        raise ValueError(
            "%s iterate-dependent multilevel AMR boundaries have no qualified nonlinear "
            "transaction" % where
        )


def _install_configured(context: Any, binding: Any) -> None:
    context.install_configured(binding)


def _red_black_geometric_mg_resolver(options: Mapping[str, Any], facts: Any, where: str) -> Any:
    if set(options) != {"mg", "field_domain", "cycle", "fac"}:
        raise TypeError("%s requires the exact staircase-MG option schema" % where)
    mg = _resolved_mg_options(options["mg"], where=where + ".mg")
    domain = options["field_domain"]
    cycle = options["cycle"]
    policy = cycle.get("residual_evaluation") if isinstance(cycle, Mapping) else None
    keys = {"schema_version", "smoother", "constant_injection", "pre_smooth_bottom"}
    version = 1
    if policy is not None:
        keys.add("residual_evaluation")
        version = 2
        if not isinstance(policy, Mapping) or set(policy) != {"policy", "version"} or policy["policy"] != "binary64-five-point-gamma-bound" or type(policy["version"]) is not int or policy["version"] != 1:
            raise ValueError("%s requires the exact binary64 residual evaluation policy" % where)
    if not isinstance(cycle, Mapping) or set(cycle) != keys or type(cycle["schema_version"]) is not int or cycle["schema_version"] != version or cycle["smoother"] != "red-black-gauss-seidel" or type(cycle["constant_injection"]) is not bool or type(cycle["pre_smooth_bottom"]) is not bool:
        raise ValueError("%s requires an exact red-black V-cycle authority" % where)
    if domain is None:
        captured = None
        centre = ()
    else:
        captured, centre = _capture_staircase_domain(domain, where=where)
    native = {
        "factory_route": "red_black_geometric_mg",
        "schema_identity": "pops.system.red-black-geometric-mg-options@%d" % version,
        "options": {
            **mg, "smoother": 1, "prolongation": int(cycle["constant_injection"]),
            "pre_smooth_bottom": cycle["pre_smooth_bottom"],
            "domain_dimension": len(centre), "domain_radius": 0.0 if captured is None else captured.radius,
            **{"domain_centre_%d" % axis: centre[axis] if axis < len(centre) else 0.0 for axis in range(3)},
        },
    }
    if policy is not None:
        native["options"].update({"roundoff_policy": 1, "roundoff_policy_version": 1})
    if facts.target == "amr_system":
        defaults = {
            "max_iters": mg["max_cycles"], "fine_sweeps": mg["pre_smooth"] + mg["post_smooth"],
            "rel_tol": mg["rel_tol"], "abs_tol": mg["abs_tol"],
            "coarse_rel_tol": mg["rel_tol"], "coarse_abs_tol": mg["abs_tol"],
            "coarse_cycles": mg["max_cycles"], "verbose": False,
        }
        if options["fac"] is None:
            fac = _resolved_fac_options(defaults, where=where + ".fac")
        else:
            authored = options["fac"]
            if not isinstance(authored, Mapping) or set(authored) != _FAC_KEYS:
                raise TypeError("%s requires the exact FAC option schema" % where)
            fac = _resolved_fac_options({name: defaults[name] if value is None else value
                                         for name, value in authored.items()}, where=where + ".fac")
        native["schema_identity"] = "pops.amr.field-solver-options.red-black-geometric-mg@%d" % version
        native["options"].update({"fac.%s" % name: value for name, value in fac.items()})
    elif facts.target != "system":
        raise ValueError("%s explicit red-black MG does not implement this runtime target" % where)
    return _resolution(native, _builtin_topology(facts))


def _capture_staircase_domain(domain: Any, *, where: str) -> Any:
    from pops.fields.domain import CellCentreBall
    if not isinstance(domain, Mapping) or set(domain) != {
        "schema_version", "region", "centre", "radius", "sampling", "exterior"
    }:
        raise TypeError("%s requires an exact field-domain authority" % where)
    centre = domain["centre"]
    if not isinstance(centre, tuple):
        raise TypeError("%s field-domain coordinates lost immutable binding" % where)
    captured = CellCentreBall(centre=centre, radius=domain["radius"])
    canonical = captured.to_data()
    canonical["centre"] = tuple(canonical["centre"])
    if type(domain["schema_version"]) is not int or dict(domain) != canonical:
        raise ValueError("%s field-domain sampling/exterior authority is inconsistent" % where)
    return captured, centre


def _validate_red_black_geometric_mg(use: Any, where: str) -> None:
    facts = use.facts
    hierarchy = _hierarchy_policy_identity(facts, where=where)
    if facts.target == "system":
        if facts.layout.get("kind") != "uniform" or facts.layout.get("levels") != 1 or hierarchy != _LEVEL_LOCAL_HIERARCHY_POLICY:
            raise ValueError("%s uniform red-black MG requires one level-local field layout" % where)
        if use.options["fac"] is not None:
            raise ValueError("%s a uniform red-black MG cannot carry an AMR FAC contract" % where)
    elif facts.target == "amr_system":
        if facts.layout.get("kind") != "amr" or (facts.layout.get("levels", 0) > 1 and hierarchy != _COMPOSITE_HIERARCHY_POLICY):
            raise ValueError("%s sparse multilevel field domains require an explicitly composite FAC authority" % where)
    else:
        raise ValueError("%s red-black MG has no authority for this runtime target" % where)
    if facts.layout.get("embedded_boundary"):
        raise ValueError("%s field-only conductor must not reuse a transport embedded boundary" % where)
    if facts.operator.get("screened"):
        raise ValueError("%s field-only staircase MG implements constant Poisson only" % where)
    cells = facts.layout.get("cells")
    domain = use.options["field_domain"]
    if not isinstance(cells, tuple) or (domain is not None and len(cells) != len(domain["centre"])):
        raise ValueError("%s field-domain coordinate rank differs from its exact storage" % where)
    if facts.boundary.get("iterate_dependent"):
        raise ValueError("%s staircase conductor has no dynamic-boundary authority" % where)


def _register_red_black_geometric_mg_field_solver_provider() -> Any:
    from pops.fields._prepared_field_solver_registry import (
        PreparedFieldSolverProvider as Provider, PreparedFieldSolverUsePolicy as UsePolicy,
        register_prepared_field_solver_provider as register,
    )
    return register(Provider(
        provider_id="pops.field-solver.red-black-geometric-mg", version=1,
        resolver_id="pops.field-solver.red-black-geometric-mg.resolve@1",
        installer_id="pops.field-solver.red-black-geometric-mg.install@1",
        use_policy=UsePolicy("pops.field-solver.red-black-geometric-mg.use", 1, {
            "targets": ("system", "amr_system"), "operators": ("poisson",),
            "algorithm": "red-black-gs.explicit-prolongation-and-bottom-contract",
            "composite_algorithm": "partitioned-FAC.quadratic-CF.flux-matching.initial-residual-scale",
            "field_domain": "strict-cell-centre-ball.zero-exterior.resample-per-level",
            "transport_topology": "full-material-unchanged",
        }, _validate_red_black_geometric_mg),
        resolver=_red_black_geometric_mg_resolver, native_installer=_install_configured,
    ))


def _fft_resolver(options: Mapping[str, Any], facts: Any, where: str) -> Any:
    if options:
        raise TypeError("%s FFT provider has an exact empty option schema" % where)
    return _resolution(
        {
            "factory_route": "fft",
            "schema_identity": "pops.system.fft-discrete-exact-rank-options.empty@2",
            "options": {},
        },
        _builtin_topology(facts),
    )


def _validate_fft(use: Any, where: str) -> None:
    facts = use.facts
    if facts.target != "system" or facts.layout.get("kind") != "uniform":
        raise ValueError("%s FFT provider requires a single uniform System layout" % where)
    if _hierarchy_policy_identity(facts, where=where) != _LEVEL_LOCAL_HIERARCHY_POLICY:
        raise ValueError("%s FFT provider requires a level-local hierarchy" % where)
    if facts.operator.get("screened"):
        raise ValueError("%s FFT provider does not implement a screened operator" % where)
    if facts.layout.get("embedded_boundary"):
        raise ValueError("%s FFT provider requires a full-material topology" % where)
    cells = facts.layout.get("cells")
    if not isinstance(cells, tuple) or len(cells) not in (1, 2, 3):
        dimension = len(cells) if isinstance(cells, tuple) else "unknown"
        raise ValueError(
            "%s FFT concrete backend requires Cartesian Dim in {1,2,3}; got Dim=%s"
            % (where, dimension)
        )
    if any(type(value) is not int or value < 1 for value in cells):
        raise ValueError("%s FFT provider requires a positive cell count on every axis" % where)
    faces = facts.boundary.get("faces")
    if (
        not isinstance(faces, tuple)
        or not faces
        or any(not isinstance(face, Mapping) or face.get("type") != "periodic" for face in faces)
    ):
        raise ValueError("%s FFT provider requires fully periodic boundaries" % where)
    if facts.boundary.get("dynamic") or facts.boundary.get("dependent"):
        raise ValueError("%s FFT provider requires an immutable boundary contract" % where)
    if facts.nonlinear:
        raise ValueError("%s FFT provider cannot serve a nonlinear outer solve" % where)


def _register_ready_providers() -> tuple[Any, Any, Any]:
    # Lazy import preserves the solvers authoring layer's import-time DAG.  Concrete provider
    # registration occurs only when a descriptor is prepared for field lowering.
    from pops.fields._prepared_field_solver_registry import (
        PreparedFieldSolverProvider as Provider,
        PreparedFieldSolverUsePolicy as UsePolicy,
        register_prepared_field_solver_provider as register,
    )

    cartesian_cg = register(
        Provider(
            provider_id="pops.field-solver.cartesian-cg",
            version=1,
            resolver_id="pops.field-solver.cartesian-cg.resolve@1",
            installer_id="pops.field-solver.cartesian-cg.install@1",
            use_policy=UsePolicy(
                "pops.field-solver.cartesian-cg.use",
                1,
                {
                    "targets": ("system",),
                    "layout": "uniform",
                    "operators": ("poisson",),
                    "boundary": "static-cartesian",
                },
                _validate_cartesian_cg,
            ),
            resolver=_cartesian_cg_resolver,
            native_installer=_install_configured,
        )
    )
    geometric = register(
        Provider(
            provider_id="pops.field-solver.geometric-mg",
            version=1,
            resolver_id="pops.field-solver.geometric-mg.resolve@1",
            installer_id="pops.field-solver.geometric-mg.install@1",
            use_policy=UsePolicy(
                "pops.field-solver.geometric-mg.use",
                1,
                {
                    "targets": ("amr_system",),
                    "operators": ("poisson", "screened-poisson"),
                    "hierarchy_policies": (
                        "pops.field-hierarchy.level-local@1",
                        "pops.field-hierarchy.composite@1",
                    ),
                    "amr_boundary_dependencies": (
                        "level-qualified-state@1",
                        "level-qualified-field@1",
                        "logical-timepoint@1",
                    ),
                },
                _validate_geometric_mg,
            ),
            resolver=_geometric_mg_resolver,
            native_installer=_install_configured,
        )
    )
    fft = register(
        Provider(
            provider_id="pops.field-solver.fft",
            version=3,
            resolver_id="pops.field-solver.fft.resolve@3",
            installer_id="pops.field-solver.fft.install@3",
            use_policy=UsePolicy(
                "pops.field-solver.fft.use",
                3,
                {
                    "targets": ("system",),
                    "layout": "uniform-exact-rank-cartesian",
                    "boundary": "fully-periodic",
                    "operator": "discrete-cartesian-poisson",
                },
                _validate_fft,
            ),
            resolver=_fft_resolver,
            native_installer=_install_configured,
        )
    )
    return cartesian_cg, geometric, fft


_CARTESIAN_CG_PROVIDER, _GEOMETRIC_MG_PROVIDER, _FFT_PROVIDER = _register_ready_providers()
_RED_BLACK_GEOMETRIC_MG_PROVIDER = _register_red_black_geometric_mg_field_solver_provider()


def cartesian_cg_field_solver_provider() -> Any:
    return _CARTESIAN_CG_PROVIDER


def geometric_mg_field_solver_provider() -> Any:
    return _GEOMETRIC_MG_PROVIDER


def red_black_geometric_mg_field_solver_provider() -> Any:
    return _RED_BLACK_GEOMETRIC_MG_PROVIDER


def fft_field_solver_provider() -> Any:
    return _FFT_PROVIDER


__all__ = [
    "cartesian_cg_field_solver_provider",
    "fft_field_solver_provider",
    "geometric_mg_field_solver_provider",
    "red_black_geometric_mg_field_solver_provider",
]
