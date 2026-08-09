"""The live AMR runtime view (Spec 5 sec.8.12 ``sim.amr``).

:class:`AmrRuntimeView` is the runtime-bound handle returned by ``AmrSystem.amr``. It is the only
part of :mod:`pops.runtime.amr` tied to a live system: it reads the already-built box accessors
(``patch_bounds`` / ``patch_boxes`` / ``coarse_local_boxes`` / ``coarse_total_boxes``) and the
static config the ``AmrSystem`` retained, then packages them into the inert report value classes.

It is INERT: it builds nothing, allocates nothing, steps no clock. The native box accessors trigger
only the same lazy box build that ``n_patches()`` does; before the hierarchy is built (no block
added / no first step) they raise, and the view reports an unbuilt hierarchy instead of forcing a
build. A measure the native build cannot answer (per-level ghost depth, composite Poisson) is
reported as honestly unavailable.
"""

from __future__ import annotations

import math
from typing import Any

from pops.runtime.amr._reports import (
    PatchReport,
    RegridReport,
    GhostReport,
    RefluxReport,
    CheckpointReport,
    HierarchySnapshot,
    RuntimeInspection,
)


# The NATIVE C++ engine raises this substring before the first block is installed / the hierarchy
# is built; we treat it as "not built yet" rather than letting it propagate (the report is inert,
# never a build). This is a native-message MATCHER, not a user remedy -- it must stay byte-for-byte
# in sync with the C++ side ("AmrSystem : call add_block first"), so it is not reworded here; a bound
# simulation is populated by pops.bind(compiled, ...) before its amr view reports a built hierarchy.
_NOT_BUILT = "call add_block first"


class AmrRuntimeView:
    """Inert, live-bound inspection handle for an :class:`AmrSystem` (Spec 5 sec.8.12).

    Construct via ``sim.amr`` (do not instantiate directly). Every method returns one of the inert
    report value classes (:class:`PatchReport`, :class:`RegridReport`, :class:`GhostReport`,
    :class:`RefluxReport`, :class:`CheckpointReport`, :class:`HierarchySnapshot`).
    """

    def __init__(self, amr_system: Any) -> None:
        # Bound to the AmrSystem facade (not the raw _AmrSystem): we read its box helpers, its
        # block registry and the config snapshot it retained.
        self._sim = amr_system

    # --- live box readers (guard the pre-build RuntimeError) -----------------
    def _is_built(self) -> Any:
        """True once the native hierarchy has been built (a block added; boxes available)."""
        try:
            self._sim._s.n_levels()
            return True
        except RuntimeError as exc:
            if _NOT_BUILT in str(exc):
                return False
            raise

    def _coarse_boxes(self) -> Any:
        """(local, total) coarse base box counts, or (None, None) before the build."""
        try:
            return (int(self._sim.coarse_local_boxes()), int(self._sim.coarse_total_boxes()))
        except RuntimeError as exc:
            if _NOT_BUILT in str(exc):
                return (None, None)
            raise

    def _block_names(self) -> Any:
        try:
            return list(self._sim._s.block_names())
        except RuntimeError as exc:
            if _NOT_BUILT in str(exc):
                return []
            raise

    def _per_level(self) -> Any:
        """Per-level patch census from exact ranked index and physical bounds."""
        s = self._sim._s
        n_levels = int(s.n_levels())
        base_shape = tuple(int(value) for value in s.spatial_shape())
        # Level 0: the coarse base covers the whole domain; report it as one covering box.
        levels = {
            0: {
                "level": 0,
                "n_patches": 1,
                "cells": math.prod(base_shape),
                "boxes": [],
                "physical_bounds": [],
            }
        }
        for lvl in range(1, n_levels):
            levels[lvl] = {
                "level": lvl,
                "n_patches": 0,
                "cells": 0,
                "boxes": [],
                "physical_bounds": [],
            }
        physical_bounds = self._sim.patch_bounds()
        # Both projections are parallel (one physical tuple per index box); strict=True
        # asserts that invariant rather than silently truncating to the shorter.
        for (level, lower, upper), physical in zip(s.patch_boxes(), physical_bounds, strict=True):
            level = int(level)
            lower = tuple(int(value) for value in lower)
            upper = tuple(int(value) for value in upper)
            if len(lower) != len(base_shape) or len(upper) != len(base_shape):
                raise ValueError("native AMR patch bounds differ from the runtime spatial rank")
            entry = levels.setdefault(
                level,
                {
                    "level": level,
                    "n_patches": 0,
                    "cells": 0,
                    "boxes": [],
                    "physical_bounds": [],
                },
            )
            entry["n_patches"] += 1
            entry["cells"] += math.prod(
                high - low + 1 for low, high in zip(lower, upper, strict=True)
            )
            entry["boxes"].append((lower, upper))
            entry["physical_bounds"].append(tuple(float(value) for value in physical))
        return [levels[k] for k in sorted(levels)]

    # --- the sec.8.12 reports ------------------------------------------------
    def patch_table(self) -> Any:
        """Return a :class:`PatchReport` of the live patches + coarse box distribution."""
        coarse_local, coarse_total = self._coarse_boxes()
        bounds = (self._sim._lower, self._sim._upper)
        if not self._is_built():
            return PatchReport(
                built=False,
                n_levels=None,
                base_cells=self._sim._shape,
                domain_bounds=bounds,
                per_level=[],
                coarse_local_boxes=coarse_local,
                coarse_total_boxes=coarse_total,
            )
        return PatchReport(
            built=True,
            n_levels=int(self._sim._s.n_levels()),
            base_cells=tuple(int(value) for value in self._sim._s.spatial_shape()),
            domain_bounds=bounds,
            per_level=self._per_level(),
            coarse_local_boxes=coarse_local,
            coarse_total_boxes=coarse_total,
        )

    def explain_regrid(self) -> Any:
        """Return the regrid policy plus live completed-regrid/topology counters."""
        regrid_every = int(self._sim._regrid_every)
        frozen = regrid_every == 0
        regrid_count = int(self._sim._s.checkpoint_regrid_count())
        topology_epoch = int(self._sim._s.checkpoint_topology_epoch())
        # The union-of-tags criteria shape, as the multi-block route documents it (the exact
        # threshold lives on the native model; we name the criteria, not a fabricated number).
        # Refinement is declared by the structured AMR layout's tagging authority.
        criteria = [
            "per-block variable threshold (AMR tagging predicate on variable/role; "
            "default the density, component 0)",
            "grad phi (AMR tagging predicate; multi-block only, disabled when the threshold <= 0)",
        ]
        notes = []
        if frozen:
            notes.append(
                "regrid_every == 0: the hierarchy is built once and frozen "
                "(bit-identical, no dynamic regrid)."
            )
        else:
            notes.append("a cell is tagged when ANY criterion fires (cell-by-cell OR).")
        return RegridReport(
            regrid_every=regrid_every,
            frozen=frozen,
            regrid_count=regrid_count,
            topology_epoch=topology_epoch,
            criteria=criteria,
            notes=notes,
        )

    def explain_ghosts(self) -> Any:
        """Return a :class:`GhostReport`; the per-level ghost depth is honestly unavailable."""
        return GhostReport(
            per_level_depth=None,
            requirement_note=(
                "the reconstruction stencil sets the ghost depth "
                "(minmod / vanleer / mc / superbee -> 1, weno5 -> 3); the coarse-fine fine ghosts "
                "are re-derived per path on the AMR transport."
            ),
            notes=["per-level ghost depth is not exposed by this native build."],
        )

    def explain_reflux(self) -> Any:
        """Return a :class:`RefluxReport` of the route reflux requirement."""
        return RefluxReport(
            enabled=True,
            per_stage=None,
            notes=[
                "coarse-fine flux refluxing is a native AMR route requirement (the AMR layout "
                "descriptor reports reflux=True); it runs on the single-block coupler path."
            ],
        )

    def explain_checkpoint(self) -> Any:
        """Return a :class:`CheckpointReport` of the live system's restartability (sec.8.11)."""
        constraints = [
            "same bound composition and compiled Program",
            "authenticated v5 accepted-state payload",
            "same recorded patch boxes and refinement topology under RestoreRecordedHierarchy()",
        ]
        violations = []
        try:
            n_blocks = int(self._sim._s.n_blocks())
        except RuntimeError:
            n_blocks = len(self._block_names())
        if n_blocks == 0:
            violations.append("no block is installed, so there is no accepted hierarchy to persist")
        notes = [
            "v5 persists every block/level, owner-rank map, aux and qualified field warm start.",
            "active regridding is supported: topology epoch, regrid count, exact level clocks, "
            "history identities and lagged conservative flux publications are restored.",
            "bit_identical=True requires the recorded MPI rank count and owner-rank map.",
            "the default non-bit-identical route may rematerialize hierarchy ownership and the "
            "rank-owned Program accepted state onto a different rank count only when every persisted "
            "history ring is Dense; selective history replay remains same-rank.",
            "RegridOnRestart() is an explicit weaker continuation for one AMR layout with an "
            "artifact-backed Program and unchanged MPI cardinality: it restores the exact accepted "
            "state first, then performs one scientific tag/regrid at the restored clock.",
            "RegridOnRestart() supports serial and exact-MPI-world rematerializable "
            "depth-preserving shared-interface flux groups at unchanged MPI cardinality; it still "
            "refuses Uniform and multi-layout runtimes, active-depth changes, unsupported "
            "non-finest replacements at depth greater than two, elliptic field providers, and "
            "bootstrap staggered caches.",
            "Its phase-local history consensus fingerprints are cold-restart collectives whose "
            "memory and communication cost scales with every dense history slot; they prove "
            "all-rank agreement per hierarchy, not bitwise equality across interpolation.",
        ]
        return CheckpointReport(
            restartable=not violations, constraints=constraints, violations=violations, notes=notes
        )

    def hierarchy_snapshot(self) -> Any:
        """Return the native config envelope composed with the live patch table."""
        # Config envelope from the inert authoring report (native max_levels / ratio / limitations).
        from pops._capabilities_inspect import _native_amr_envelope

        envelope = _native_amr_envelope().to_dict()
        regrid_every = int(self._sim._regrid_every)
        patch_table = self.patch_table()
        return HierarchySnapshot(
            blocks=self._block_names(),
            max_levels=envelope["max_levels"],
            ratio=envelope["ratio"],
            regrid_every=regrid_every,
            frozen=regrid_every == 0,
            patch_table=patch_table,
            limitations=envelope["limitations"],
            config_available=envelope["available"],
        )

    def inspect(self) -> Any:
        """Return the unified AMR runtime inspection report (Spec 5 sec.8.12, ADC-589/ADC-555).

        Composes the four things a single ``sim.amr.inspect()`` call must answer: the
        :class:`HierarchySnapshot` (config envelope + live patches), the live
        :class:`PatchReport` again as its own key, the :class:`RegridReport` (cadence + union-tag
        criteria), and the capability ``limitations`` (the non-available native-route rows from
        the native capability report, from the same source as :mod:`pops.problem`'s route matrix,
        reads). This REPLACES the pre-ADC-589 shape (bare ``HierarchySnapshot``); the snapshot
        itself is unchanged and still reachable directly via :meth:`hierarchy_snapshot`.
        """
        from pops._capabilities import native_capability_report

        limitations = [
            row.to_dict() for row in native_capability_report().routes if row.status != "available"
        ]
        return RuntimeInspection(
            hierarchy=self.hierarchy_snapshot(),
            patches=self.patch_table(),
            regrid=self.explain_regrid(),
            limitations=limitations,
        )

    def __repr__(self) -> Any:
        return "AmrRuntimeView(blocks=%r)" % (self._block_names(),)

    def __str__(self) -> Any:
        """Short, array-free handle summary (Spec 5 sec.12.1)."""
        built = "built" if self._is_built() else "not built"
        return "AmrRuntimeView(blocks=%s, hierarchy %s)" % (self._block_names(), built)
