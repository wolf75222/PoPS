"""CF2 (spec 0.3, section 16.2): the architecture test over both trees of the frozen engine.

This module turns the occurrences of ``scan.py`` into exception-list entries, each with the Linear
issue that takes the site over. The issue is a proposal derived from sections 18.3 and 18.4 of the
spec by the first matching rule below; a slice that moves a site to another issue edits the rule,
not the list.
"""
from __future__ import annotations

import pathlib

from . import ratchet, scan

EXCEPTIONS = scan.HERE / "cf2_exceptions.txt"

_BOUNDARY = frozenset(term.label for term in scan.VOCABULARY if term.category == "boundary")
_INTERPOLATION = frozenset(term.label for term in scan.VOCABULARY
                           if term.category == "interpolation") | {"monotonized"}

# (issue, path entries or None, terms or None, reason). The first rule that matches both its paths
# and its terms names the issue. A path entry ending with "/" is a directory.
OWNER_RULES = (
    ("N9", ("include/pops/numerics/elliptic/", "python/pops/solvers/",
            "include/pops/runtime/amr/amr_tensor_elliptic.hpp",
            "include/pops/runtime/amr/hierarchy_tensor_solver_provider.hpp",
            "include/pops/runtime/amr/tensor_coarse_gmres.hpp",
            "include/pops/runtime/amr/tensor_composite_fac.hpp"), None,
     "IR24: native solvers (Krylov, multigrid, FFT, polar and Poisson, AMR tensor), Phase 2"),
    ("G06", ("include/pops/numerics/fv/path_flux.hpp", "include/pops/amr/reflux/",
             "include/pops/coupling/source/coupled_source_program.hpp",
             "python/pops/physics/multispecies.py"), None,
     "IR24 path-conservative flux, reflux ledger; C44 coupled-source bytecode, Phase 2"),
    ("G11", ("include/pops/amr/transfer/", "include/pops/runtime/amr/bootstrap_transfer_builtins.hpp"),
     None, "C43: AMR transfer kernels, Phase 2"),
    ("G11", None, _INTERPOLATION, "C43: named coarse-fine and temporal interpolations, Phase 2"),
    ("N12", ("include/pops/amr/tagging/",), None, "C36: the tag-graph VM, Phase 1"),
    ("G04", ("include/pops/runtime/analytic/",), None,
     "C44: the analytic-expression VM, listed under G04 only (R10-02), Phase 2"),
    ("G04", None, _BOUNDARY, "named boundary families, Phase 2"),
    ("G05", None, frozenset({"scharfetter", "gummel"}), "Scharfetter-Gummel, Phase 2"),
    ("G10", None, frozenset({"laplacian", "curl", "poisson"}),
     "field operators and solves (C7, C28), Phase 2"),
    ("N3", None, None,
     "mesh catalogue and physics out of the core: fluxes, reconstructions, limiters, roles, "
     "the \"p\" switch, gamma, route switches, closures and models (C3, C19-C21, C28-C30), Phase 1"),
)


def owner_of(path: str, term: str) -> str:
    for issue, paths, terms, _ in OWNER_RULES:
        if paths is not None and not scan.covered(path, paths):
            continue
        if terms is not None and term not in terms:
            continue
        return issue
    raise AssertionError("OWNER_RULES has no default")  # the last rule matches everything


def current_entries(root: pathlib.Path = scan.REPO_ROOT, engine=None):
    """(entries for every CF2 site of the checkout, generic-sense entries that matched)."""
    occurrences, used_senses = scan.scan_cf2(root, engine)
    entries = [ratchet.Entry(o.path, o.line, o.term, owner_of(o.path, o.term), o.excerpt)
               for o in occurrences]
    return entries, used_senses
