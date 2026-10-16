"""Authenticate a finite partition of a retained equation into executable views.

This does not select numerical methods. Each returned view still requires its
own exact DiscretizationPlan binding during resolution. Overlapping declaration
views are ambiguous: they never become extra additive physical contributions.
"""
from __future__ import annotations


def balance_composition(registry, operator):
    """Return disjoint registered proper views covering an unsupported balance.

    The initial implicit composition accepts one unambiguous declared partition.
    It deliberately does not search for a preferred partition among overlaps.
    """
    from pops._ir.balance import BalanceView, source_balance_supported
    from pops.numerics.nonconservative import path_balance_supported
    from pops.numerics.diffusion import diffusion_balance_supported
    from pops.numerics.scharfetter_gummel import fitted_balance_supported

    view=operator.lowering.get("physical_balance")
    if not isinstance(view,BalanceView) or not operator.lowering.get("native_unsupported"):
        return ()
    required=set(view.ordinals)
    if not required:
        return ()
    parts=[]
    for candidate in registry:
        part=candidate.lowering.get("physical_balance")
        if (candidate is operator or not isinstance(part,BalanceView)
                or part.balance != view.balance or not set(part.ordinals)<required
                or not part.ordinals):
            continue
        supported=(source_balance_supported(part) or path_balance_supported(part)
                   or diffusion_balance_supported(part) or fitted_balance_supported(part)
                   or not candidate.lowering.get("native_unsupported"))
        if supported:
            parts.append(candidate)
    covered=set()
    for part in parts:
        ordinals=set(part.lowering["physical_balance"].ordinals)
        if covered & ordinals:
            return ()
        covered.update(ordinals)
    if covered != required:
        return ()
    return tuple(sorted(parts,key=lambda part:min(part.lowering["physical_balance"].ordinals)))
