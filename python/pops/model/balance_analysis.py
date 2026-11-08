"""Pure shape predicates over retained physical balance occurrences, and their coverage refusal.

``CoverageNotExact`` (D5) is the refusal of a plan that covers an occurrence twice or not at all;
``occurrence_text`` names an occurrence as that refusal names it.
"""
from pops._ir.refusals import CoverageNotExact


def occurrence_text(occurrence):
    """An occurrence of a balance as a refusal names it: what it is and the line that wrote it."""
    payload = occurrence.payload
    name = getattr(payload, "name", None) or getattr(payload, "local_id", None) or repr(payload)
    text = "%s[%d] %s %s x%s" % (occurrence.balance.local_id, occurrence.ordinal, occurrence.kind,
                                 name, occurrence.coefficient)
    if occurrence.written is not None:
        text += " (inline, written x%s)" % (occurrence.written,)
    origin = occurrence.origin
    return text if origin is None else "%s, written at %s:%d" % (text, origin.file, origin.line)


def diffusion_balance_supported(view):
    if view is None or not view.accumulation.is_identity or not any(
            row.kind in {"diffusion", "coupled_gradient"} for row in view.occurrences):
        return False
    return all(row.kind in {"diffusion", "coupled_gradient", "source", "flux"}
               for row in view.occurrences)


def fitted_balance_supported(view):
    return (view is not None and view.accumulation.is_identity
        and sum(row.kind=="drift" for row in view.occurrences)==1
        and sum(row.kind=="diffusion" for row in view.occurrences)==1
        and all(row.kind in {"drift","diffusion","source"} for row in view.occurrences))


__all__ = ["CoverageNotExact", "diffusion_balance_supported", "fitted_balance_supported",
           "occurrence_text"]
