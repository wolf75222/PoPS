"""Pure shape predicates over retained physical balance occurrences."""

def diffusion_balance_supported(view):
    if view is None or not view.accumulation.is_identity or not any(
            row.kind == "diffusion" for row in view.occurrences):
        return False
    return all(row.kind in {"diffusion", "source", "flux"} for row in view.occurrences)


def fitted_balance_supported(view):
    return (view is not None and view.accumulation.is_identity
        and sum(row.kind=="drift" for row in view.occurrences)==1
        and sum(row.kind=="diffusion" for row in view.occurrences)==1
        and all(row.kind in {"drift","diffusion","source"} for row in view.occurrences))
