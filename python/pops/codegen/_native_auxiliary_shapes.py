"""One exact halo shape authority for native providers and Program consumers."""
from typing import Any


def native_auxiliary_halos(model: Any) -> dict[tuple[str, str, str, str], int]:
    routes = getattr(model, "_auxiliary_provider_routes", None)
    flux_plan = getattr(model, "_component_flux_consumer_plan", None)
    if routes is None or flux_plan is None:
        raise ValueError("auxiliary shapes require exact typed routes and physical-flux reads")

    def key_tuple(key):
        return (key.owner_qid, key.space_kind, key.space_name, key.component)

    typed_routes = {key_tuple(key): route for key, route in routes.items()}
    # FV evaluates providers at the two neighboring cells of each physical face.
    halos = {tuple(row["key"][name] for name in
                   ("owner_qid", "space_kind", "space_name", "component")): 1
             for row in flux_plan}
    for key, route in typed_routes.items():
        boundary = route.get("boundary")
        halos[key] = max(halos.get(key, 0), 0 if boundary is None else boundary.width)
    # A derived launch evaluates its complete output image, so its prerequisites
    # must carry that same image, including all demanded halo cells.
    changed = True
    while changed:
        changed = False
        for key, route in typed_routes.items():
            width = halos.get(key, 0)
            for dependency in route.get("dependencies", ()):
                dependency_key = key_tuple(dependency)
                if width > halos.get(dependency_key, 0):
                    halos[dependency_key] = width
                    changed = True
    return halos


def auxiliary_shape_cpp(width: int) -> str:
    if type(width) is not int or width < 0:
        raise ValueError("auxiliary halo width must be a non-negative integer")
    return (
        "Shape{pops::kNativeDimension, 1, [] { pops::Index<pops::kNativeDimension> halo{}; "
        "for (int axis = 0; axis < pops::kNativeDimension; ++axis) halo[axis] = %d; return halo; }()}"
        % width)
