"""PF-09 in-memory rank loads.

Uniform 16/16/16/16 versus a rank-0 hotspot of 40 cells. Optional
``run_native`` refuses unless a public standalone balancer exists.
"""
from __future__ import annotations

import inspect
import time
from pathlib import Path

from verification.pops_verify.case_authoring import load_sibling_module

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))
_v15 = load_sibling_module(Path(__file__).resolve().parents[1] / "_v15.py")

PUBLIC_BALANCER_REFUSAL = "public standalone load balancer is not available"
_BALANCER_MODULES = ("pops.layouts", "pops.lib.amr", "pops.amr")
_BALANCER_NAMES = (
    "Balancer",
    "LoadBalancer",
    "Redistribute",
    "rebalance",
    "balance_boxes",
    "redistribute",
)
_BALANCER_METHODS = ("rebalance", "balance", "redistribute")


class NativeUnavailable(RuntimeError):
    """Raised when a public standalone balancer cannot be timed."""


def uniform_counts(n_cells=_exact.N_CELLS, n_ranks=_exact.N_RANKS):
    """Return the even 64-cell / 4-rank occupancy."""
    return _exact.uniform_counts(n_cells, n_ranks)


def hotspot_counts(
    n_cells=_exact.N_CELLS,
    n_ranks=_exact.N_RANKS,
    hotspot=_exact.HOTSPOT_RANK0,
):
    """Return the rank-0 hotspot occupancy."""
    return _exact.hotspot_counts(n_cells, n_ranks, hotspot)


def uniform_stats(n_cells=_exact.N_CELLS, n_ranks=_exact.N_RANKS) -> dict:
    """Return max/mean/CV of the uniform occupancy."""
    return _exact.load_stats(uniform_counts(n_cells, n_ranks))


def hotspot_stats(
    n_cells=_exact.N_CELLS,
    n_ranks=_exact.N_RANKS,
    hotspot=_exact.HOTSPOT_RANK0,
) -> dict:
    """Return max/mean/CV of the hotspot occupancy."""
    return _exact.load_stats(hotspot_counts(n_cells, n_ranks, hotspot))


def public_balancer():
    """Return a public standalone balancer callable, or ``None``.

    ``pops.lib.amr`` exports SpaceFillingCurve / Knapsack descriptors.
    Those are AMR authoring values. ``layouts.AMR.load_balance`` is a
    property, not a callable that redistributes occupancy.
    """
    import importlib

    for module_name in _BALANCER_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for name in _BALANCER_NAMES:
            candidate = getattr(module, name, None)
            if candidate is None:
                continue
            if inspect.isfunction(candidate):
                return candidate
            if inspect.isclass(candidate):
                for method_name in _BALANCER_METHODS:
                    method = getattr(candidate, method_name, None)
                    if callable(method):
                        return method
    return None


def refuse_public_balancer() -> str:
    """Return why a native load-balance timer cannot run."""
    if public_balancer() is None:
        return PUBLIC_BALANCER_REFUSAL
    return "public standalone load balancer cannot be driven without inventing ranks"



def official_authority() -> dict:
    """PF-09 is absent from benchmarks/manifest.toml."""
    return _v15.official_authority("PF-09")


def run_native(*args, request=None, **kwargs):
    """PF timed work belongs to benchmarks/manifest.toml, not a sibling wrap."""
    from verification.pops_verify.official_benchmark import OfficialBenchmarkUnavailable

    try:
        return _v15.run_mapped_or_refuse("PF-09", request)
    except OfficialBenchmarkUnavailable as exc:
        raise NativeUnavailable(str(exc)) from exc

