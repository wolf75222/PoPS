"""AMR output / checkpoint run-loop driver (Spec 5 sec.5.14 / addendum C.1, ADC-542).

The AMR sibling of :mod:`pops.runtime._output_driver`. It fires the typed
:class:`pops.output.OutputPolicy` / :class:`pops.output.CheckpointPolicy` descriptors on an
``AmrSystem`` at their cadence, HONORING the typed level policy (``AllLevels`` / ``CoarseOnly`` /
``SelectedLevels``) over the AMR hierarchy -- the per-level writes the base plan deferred (N2) are a
capability here, not a refusal. The cadence interpreter is SHARED with the uniform driver
(``policy_due``), so ``on_start`` / ``on_end`` / ``when`` land on AMR too.

Level resolution is LATE-BOUND (validated per fire, not per compile): under active regridding
``n_levels`` varies, so a :class:`SelectedLevels` naming a level >= the live count is refused with a
verbatim message naming the live count at the moment of the write.
"""
from pops.runtime._output_driver import policy_due


def _reject_output(code, message, *, evidence=None, actions=()):
    from pops._report import DiagnosticError, ReportTree

    raise DiagnosticError(ReportTree(
        phase="runtime", severity="error", code="runtime.output.%s" % code,
        message=message, source="amr_output", evidence=evidence or {}, actions=actions))


def resolve_levels(sim, levels):
    """The level index set a typed level policy selects on @p sim's live hierarchy.

    ``AllLevels`` -> every level; ``CoarseOnly`` -> ``[0]``; ``SelectedLevels(*ks)`` -> the validated
    subset. A selected level >= the LIVE ``n_levels()`` is refused verbatim (late-bound: the count
    varies under regridding). ``None`` defaults to all levels.
    """
    n_levels = int(sim.n_levels())
    name = type(levels).__name__ if levels is not None else "AllLevels"
    if name == "CoarseOnly":
        return [0]
    if name == "SelectedLevels":
        ks = list(getattr(levels, "levels", ()))
        for k in ks:
            if k < 0 or k >= n_levels:
                _reject_output(
                    "selected_level_out_of_range",
                    "OutputPolicy(levels=SelectedLevels(%s)): level %d is out of range for the live "
                    "AMR hierarchy (n_levels=%d at this write). Under active regridding the level "
                    "count varies; select a level in [0, %d)."
                    % (", ".join(str(x) for x in ks), k, n_levels, n_levels),
                    evidence={"selected_levels": ks, "level": k, "n_levels": n_levels})
        return ks
    return list(range(n_levels))  # AllLevels (and the default)


def fire_amr_output_policies(sim, policies, step, output_dir, last_step=None):
    """Fire every DUE AMR output / checkpoint policy at macro-step @p step (the run-loop hook).

    For each :class:`OutputPolicy` whose cadence is due, resolve the level set (:func:`resolve_levels`)
    and write it; for each :class:`CheckpointPolicy` whose cadence is due, checkpoint (the AMR v3
    checkpoint honors the whole hierarchy). Returns the list of written paths. A policy type the driver
    does not recognise raises rather than being silently skipped.
    """
    import os

    written = []
    for policy in policies:
        cat = getattr(policy, "category", None)
        if cat == "output_policy":
            if not policy_due(policy.cadence, step, last_step=last_step, sim=sim):
                continue
            levels = resolve_levels(sim, getattr(policy, "levels", None))
            prefix = os.path.join(output_dir, getattr(policy, "prefix", None) or "output")
            written.append(_write_amr(sim, prefix, policy, step, levels))
        elif cat == "checkpoint_policy":
            if not policy_due(policy.cadence, step, last_step=last_step, sim=sim):
                continue
            prefix = os.path.join(output_dir, getattr(policy, "prefix", None) or "checkpoint")
            written.append(sim.checkpoint("%s_%06d" % (prefix, step)))
        else:
            _reject_output(
                "unsupported_policy",
                "output policy must be a pops.output.OutputPolicy / CheckpointPolicy "
                "(got category %r)" % (cat,), evidence={"category": cat})
    return written


def _write_amr(sim, prefix, policy, step, levels):
    """Write the AMR state for the resolved @p levels via the typed format.

    A ``Plotfile()`` format writes the AMReX plotfile layout (:mod:`pops.runtime._plotfile_writer`);
    ``vtk`` keeps the existing coarse visualization writer; every other format (npz default, the HDF5
    token included) writes the PER-LEVEL npz: every selected level's full per-block state arrays plus
    the shared phi -- the same per-level accessors the v3 checkpoint gathers, so AllLevels /
    SelectedLevels actually emit every selected level's data, not just level 0 + footprints.
    """
    fmt = getattr(policy, "format", None)
    fmt_name = type(fmt).__name__ if fmt is not None else "npz"
    if fmt_name == "HDF5":
        _reject_output(
            "hdf5_not_lowered",
            "AMR OutputPolicy(format=HDF5()) has no HDF5 writer; refusing the historical NPZ "
            "substitution. Use Plotfile() or the default NPZ route.",
            actions=("use Plotfile() or the exact NPZ route",))
    if getattr(policy, "fields", None):
        _reject_output(
            "field_subset_not_lowered",
            "AMR OutputPolicy(fields=...) cannot yet preserve the exact qualified field subset; "
            "refusing to widen it silently to every block state and phi")
    if getattr(policy, "diagnostics", None):
        _reject_output(
            "diagnostics_not_lowered",
            "AMR OutputPolicy(diagnostics=...) has no output-diagnostic lowering; refusing to "
            "drop the requested diagnostics")
    if getattr(policy, "require_parallel", False):
        _reject_output(
            "parallel_writer_unavailable",
            "AMR OutputPolicy(require_parallel=True) cannot be honored by the rank-0 NPZ/plotfile "
            "writers; refusing to report a parallel route")
    if fmt_name == "Plotfile":
        from pops.runtime._plotfile_writer import write_plotfile
        return write_plotfile(sim, prefix, step=step, levels=levels)
    if fmt_name == "VTK" or (isinstance(fmt, str) and fmt == "vtk"):
        if type(getattr(policy, "levels", None)).__name__ not in {"AllLevels", "NoneType"}:
            _reject_output(
                "vtk_level_subset_not_lowered",
                "AMR VTK output cannot preserve a selected level subset; use Plotfile/NPZ")
        return sim.write(prefix, format="vtk", step=step)
    if fmt is not None and not (isinstance(fmt, str) and fmt == "npz"):
        _reject_output(
            "format_not_lowered",
            "AMR output format %r has no exact lowering" % fmt_name)
    return _write_amr_levels_npz(sim, prefix, step, levels)


def _write_amr_levels_npz(sim, prefix, step, levels):
    """Per-level npz payload: state_<block>_<k> + phi_<k> for every selected level @p k.

    Uses the SAME per-level accessors the AMR checkpoint gathers (block_level_state / level_state +
    level_potential; the _global collective variants under np>1, every rank calls, rank 0 writes), so
    the emitted arrays are bit-identical to the engine's per-level state. Also carries t / n / the
    selected level list / patch_boxes so a reader can reconstruct the geometry.
    """
    import os
    import numpy as np
    from pops import _pops

    gather = _pops.n_ranks() != 1
    multi = sim.n_blocks() != 1
    names = list(sim.block_names())
    pb = sim.patch_boxes()
    out = {"t": sim.time(), "n": sim.nx(),
           "levels": np.asarray(sorted(levels), dtype=np.int64),
           "patch_boxes": (np.asarray(pb, dtype=np.int64) if pb
                           else np.zeros((0, 5), dtype=np.int64))}
    for k in sorted(levels):
        for b in names:
            if multi:
                st = sim.block_level_state_global(b, k) if gather else sim.block_level_state(b, k)
            else:
                st = sim.level_state_global(k) if gather else sim.level_state(k)
            out["state_%s_%d" % (b, k)] = np.asarray(st, dtype=np.float64)
        out["phi_%d" % k] = np.asarray(
            sim.level_potential_global(k) if gather else sim.level_potential(k), dtype=np.float64)
    suffix = ("_%06d" % int(step)) if step is not None else ""
    target = prefix + suffix + ".npz"
    if _pops.my_rank() != 0:
        return target  # only rank 0 writes (the gathers already ran on every rank)
    tmp = target + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **out)
    os.replace(tmp, target)
    return target


__all__ = ["resolve_levels", "fire_amr_output_policies"]
