"""CF2 (spec 0.3, section 16.2): the architecture test over both trees of the frozen engine.

This module turns the occurrences of ``scan.py`` into exception-list entries, each with the Linear
issue that takes the site over, by the first matching rule below. The rules follow sections 18.3 and
18.4 of the spec, with the three points settled on 29/10 (adjudication, section 17): N9 keeps the
native-solver sites until Phase 2, except a named field operator (G10) and a named boundary family
(G04), which go to their mechanism even inside a native solver. The owner of each site stays with N2
and its audit (same section); the rules for the terms that the early audit of N2 added (29/10) are
its own. A slice that moves a site to another issue edits the rule, not the list.
"""
from __future__ import annotations

import pathlib

from . import ratchet, scan

EXCEPTIONS = scan.HERE / "cf2_exceptions.txt"

def _labels(category: str) -> frozenset:
    return frozenset(label for label, (kind, _) in scan.TERM_INFO.items() if kind == category)


_BOUNDARY = _labels("boundary")
_INTERPOLATION = _labels("interpolation") | {"monotonized"}
_APPROXIMATION = _labels("approximation")
_FIELD_OPERATORS = frozenset({"laplacian", "curl", "poisson", "gradient", "divergence"})
_NAMED_DIFFERENCES = frozenset({"gradient", "divergence"})

# The native solvers that IR24 moves under N9 (the files of the solver entries of
# cf0_native_placement.toml, by directory where a directory holds only solvers).
_NATIVE_SOLVERS = ("include/pops/numerics/elliptic/", "python/pops/solvers/",
                   "include/pops/runtime/amr/amr_tensor_elliptic.hpp",
                   "include/pops/runtime/amr/hierarchy_tensor_solver_provider.hpp",
                   "include/pops/runtime/amr/tensor_coarse_gmres.hpp",
                   "include/pops/runtime/amr/tensor_composite_fac.hpp")
# The closures of the moment hierarchies, where gaussian is a closure (C29).
_MOMENTS = ("python/pops/moments/", "include/pops/numerics/moments/")
# The refinement criteria outside the tag-graph VM itself (C36).
_TAGGING = ("include/pops/runtime/amr/prepared_tagging_execution.hpp",
            "include/pops/runtime/dynamic/component_consumers.hpp",
            "python/pops/numerics/indicator_stencils.py", "python/pops/mesh/_amr/tagging_graph.py",
            "python/pops/mesh/_amr/tagging_resolution.py", "python/pops/mesh/_amr/__init__.py",
            "python/pops/amr/_resolution.py")
# Diffusion, which G05 takes over with TensorDiffusion (section 18.4).
_DIFFUSION = ("include/pops/numerics/diffusion/", "python/pops/numerics/diffusion.py",
              "python/pops/physics/diffusion.py", "python/pops/codegen/program_emit_diffusion.py",
              "python/pops/time/_program/diffusion.py", "python/pops/physics/_board_rate.py")
# The finite-volume route of the mesh catalogue: face-flux divergences, bricks, reconstructions.
_MESH_ROUTE = ("include/pops/numerics/spatial/nd/finite_volume.hpp", "include/pops/numerics/fv/",
               "include/pops/physics/bricks/", "python/pops/numerics/named_flux.py",
               "python/pops/numerics/reconstruction/", "python/pops/numerics/spatial.py")

# (issue, path entries or None, terms or None, reason). The first rule that matches both its paths
# and its terms names the issue. A path entry ending with "/" is a directory.
OWNER_RULES = (
    ("G06", ("include/pops/numerics/fv/path_flux.hpp", "include/pops/amr/reflux/",
             "include/pops/numerics/time/amr/reflux/amr_interface_flux_ledger.hpp",
             "include/pops/numerics/time/amr/reflux/amr_reflux_mf.hpp",
             "include/pops/coupling/source/coupled_source_program.hpp",
             "python/pops/physics/multispecies.py"), None,
     "IR24 path-conservative flux (removed under PF14, never a substitution: FC6), reflux ledger; "
     "C44 coupled-source bytecode, Phase 2"),
    ("G11", ("include/pops/amr/transfer/", "include/pops/runtime/amr/bootstrap_transfer_builtins.hpp"),
     None, "C43: AMR transfer kernels, Phase 2"),
    ("N9", _NATIVE_SOLVERS, _INTERPOLATION | _APPROXIMATION,
     "the interpolations of the native solvers stay with them: 18.4 moves only field operators and "
     "boundary families out of N9, Phase 2"),
    ("G12", None, _APPROXIMATION, "T7: named approximations of a history, Phase 2"),
    ("G11", None, _INTERPOLATION, "C43: named coarse-fine and temporal interpolations, Phase 2"),
    ("N3", _MOMENTS, frozenset({"gaussian"}),
     "the Gaussian closures of the moment hierarchies (C29), Phase 1"),
    ("G07", None, frozenset({"gaussian"}),
     "the gaussian_field initial-data route of the analytic VM, which G07 replaces (R10-02, C44, "
     "CN12), Phase 2"),
    ("N12", ("include/pops/amr/tagging/",), None, "C36: the tag-graph VM, Phase 1"),
    ("N12", None, _labels("tagging"), "C36: the gradient predicates of the refinement criteria, "
     "Phase 1"),
    ("N12", _TAGGING, _FIELD_OPERATORS, "C36: field operators of the refinement criteria, Phase 1"),
    ("G04", ("include/pops/runtime/analytic/",), None,
     "C44: the analytic-expression VM, in boundary states and level sets (R10-02), Phase 2"),
    ("G04", None, _BOUNDARY, "named boundary families, also inside the native solvers (18.4), Phase 2"),
    ("G05", None, frozenset({"scharfetter", "gummel"}), "Scharfetter-Gummel, Phase 2"),
    ("G05", _DIFFUSION, _FIELD_OPERATORS, "the field operators of diffusion and TensorDiffusion "
     "(18.4), Phase 2"),
    ("N3", _MESH_ROUTE, _NAMED_DIFFERENCES,
     "face-flux divergences, bricks and reconstruction slopes of the mesh route, Phase 1"),
    ("G10", None, _FIELD_OPERATORS,
     "field operators and solves (C7, C28), also inside the native solvers (18.4), Phase 2"),
    ("N9", _NATIVE_SOLVERS, None,
     "IR24: the other knowledge of the native solvers (Krylov, multigrid, FFT, polar, AMR tensor), "
     "kept until Phase 2 (18.3)"),
    ("N8", None, _labels("controller"),
     "named controllers: controllers are library functions (section 15, 5.11), compiled step "
     "control, Phase 1"),
    ("N9", None, frozenset({"field-solver-route"}),
     "native solvers selected by name, replaced by the binding of SV12, Phase 2"),
    ("N3", None, None,
     "mesh catalogue and physics out of the core: fluxes, reconstructions, limiters, roles, "
     "the \"p\" switch, gamma, route switches, closures, models and their constants (C3, "
     "C19-C21, C28-C30), Phase 1"),
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
