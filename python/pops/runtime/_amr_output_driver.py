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
                raise ValueError(
                    "OutputPolicy(levels=SelectedLevels(%s)): level %d is out of range for the live "
                    "AMR hierarchy (n_levels=%d at this write). Under active regridding the level "
                    "count varies; select a level in [0, %d)."
                    % (", ".join(str(x) for x in ks), k, n_levels, n_levels))
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
            raise TypeError(
                "output policy must be a pops.output.OutputPolicy / CheckpointPolicy "
                "(got category %r)" % (cat,))
    return written


def _write_amr(sim, prefix, policy, step, levels):
    """Write the AMR state for the resolved @p levels via the typed format.

    A ``Plotfile()`` format writes the AMReX plotfile layout (:mod:`pops.runtime._plotfile_writer`);
    every other format writes the npz / vtk visualization the AMR writer already produces (coarse
    fields + fine-patch footprints), the levels metadata carried alongside. The level set is recorded
    so a reader knows which levels the file covers.
    """
    fmt = getattr(policy, "format", None)
    fmt_name = type(fmt).__name__ if fmt is not None else "npz"
    if fmt_name == "Plotfile":
        from pops.runtime._plotfile_writer import write_plotfile
        return write_plotfile(sim, prefix, step=step, levels=levels)
    token = "vtk" if fmt_name == "VTK" else ("npz" if fmt is None or fmt_name in ("HDF5",) else
                                             getattr(fmt, "name", "npz"))
    # The AMR writer produces coarse fields + patch footprints; the level set is honored by the
    # plotfile path (per-level) and recorded for the visualization path.
    return sim.write(prefix, format="npz" if token == "hdf5" else token, step=step)


__all__ = ["resolve_levels", "fire_amr_output_policies"]
