"""Output / checkpoint run-loop driver (Spec 5 sec.5.14 / case C4, ADC-509).

Wires the typed :class:`pops.output.OutputPolicy` / :class:`pops.output.CheckpointPolicy`
descriptors to the EXISTING :mod:`pops.runtime._system_io` writers (``System.write`` /
``System.checkpoint``). It is a pure-Python run-loop hook: at each macro-step the driver asks
each policy's cadence whether it is DUE and, if so, calls the matching writer. There is NO new
writer subsystem and NO C++ codegen here -- output is a runtime concern, not a kernel concern.

SCOPE: the Uniform / single-level ``System`` path ONLY. The level selection
(``AllLevels`` / ``CoarseOnly`` / ``SelectedLevels``) is a NO-OP on a single-level System --
there is one level to write -- and AMR per-level writes are the Spec 6 epic ADC-511. ``Plotfile``
has no ``System`` writer (it is the AMReX per-level format), so a ``Plotfile`` output policy is
the one precise reject here, naming ADC-511; every other policy lowers.
"""


def policy_due(cadence, step):
    """True when @p cadence is DUE at macro-step @p step (1-based: the count after a step).

    Accepts the typed schedule objects (``pops.time.schedule.every(N)`` / ``always()``) and the
    integer shorthand ``cadence=N`` (== ``every(N)``). ``None`` means every step (the default).
    An ``every(N)`` fires when ``step % N == 0`` (and ``step > 0``); ``always`` fires every step;
    a schedule kind with no run-loop meaning here (``when`` / ``on_start`` / ``on_end`` /
    ``subcycle``) raises rather than silently never firing.
    """
    if step <= 0:
        return False
    if cadence is None:
        return True
    if isinstance(cadence, bool):  # guard: bool is an int subclass, never a cadence
        raise TypeError("output cadence must be an int interval or a pops.time schedule, got a bool")
    if isinstance(cadence, int):
        return step % cadence == 0
    kind = getattr(cadence, "kind", None)
    if kind == "always":
        return True
    if kind == "every":
        n = int(cadence.params.get("n", 1))
        return step % n == 0
    raise NotImplementedError(
        "output cadence schedule kind %r is not honored by the Uniform run loop; use every(N) "
        "or an int interval (when/on_start/on_end/subcycle output is a follow-up)." % (kind,))


def _format_token(fmt):
    """Map a typed output format (``HDF5()`` / ``Plotfile()``) to a ``System.write`` format string.

    ``HDF5`` -> ``"hdf5"``; ``None`` (no explicit format) -> ``"npz"`` (the dependency-free
    default). A string is passed through (back-compat with ``format="npz"`` authoring). ``Plotfile``
    has no ``System`` writer -- it is the AMReX per-level format -- so it raises a precise
    ``NotImplementedError`` naming the AMR epic, NOT a reject of the whole output surface.
    """
    if fmt is None:
        return "npz"
    if isinstance(fmt, str):
        return fmt
    name = type(fmt).__name__
    if name == "HDF5":
        return "hdf5"
    if name == "Plotfile":
        raise NotImplementedError(
            "OutputPolicy(format=Plotfile()) has no Uniform System writer: Plotfile is the AMReX "
            "per-level format, deferred to the AMR output epic ADC-511. Use HDF5() or the npz "
            "default for a Uniform System.")
    # An unknown typed format: defer to its declared name so System.write rejects it precisely.
    return getattr(fmt, "name", name)


def _hdf5_parallel(fmt):
    """True when the typed format requests the parallel-HDF5 path (``HDF5(parallel=True)``)."""
    return bool(getattr(fmt, "parallel", False))


def _field_names(fields):
    """Resolve an OutputPolicy ``fields=[...]`` list to the block-name subset ``write`` wants.

    Each entry is a block name string or a typed handle carrying ``.name`` (a field / state
    handle). An empty list means "all blocks" -> ``None`` (``write``'s all-fields sentinel).
    """
    if not fields:
        return None
    names = []
    for f in fields:
        if isinstance(f, str):
            names.append(f)
        else:
            name = getattr(f, "name", None)
            if name is not None:
                names.append(str(name))
    return names or None


def fire_output_policies(sim, policies, step, output_dir):
    """Fire every DUE output / checkpoint policy at macro-step @p step (the run-loop hook).

    For each :class:`pops.output.OutputPolicy` whose cadence is due, call the existing
    ``sim.write(prefix, format=, step=, fields=, parallel=)``; for each
    :class:`pops.output.CheckpointPolicy` whose cadence is due, call the existing
    ``sim.checkpoint(prefix)``. @p output_dir is the directory the files land in (the run supplies
    it); each policy writes ``<output_dir>/<sim-or-policy-prefix>``. Returns the list of written
    paths (useful for tests / logging).

    The level selection on a policy is a documented NO-OP on a single-level System (one level to
    write); AMR per-level filtering is ADC-511. A policy type the driver does not recognise raises
    rather than being silently skipped.
    """
    import os

    written = []
    for policy in policies:
        cat = getattr(policy, "category", None)
        if cat == "output_policy":
            if not policy_due(policy.cadence, step):
                continue
            fmt = _format_token(policy.format)
            prefix = os.path.join(output_dir, getattr(policy, "prefix", None) or "output")
            written.append(sim.write(
                prefix, format=fmt, step=step, fields=_field_names(policy.fields),
                parallel=_hdf5_parallel(policy.format)))
        elif cat == "checkpoint_policy":
            if not policy_due(policy.cadence, step):
                continue
            prefix = os.path.join(output_dir, getattr(policy, "prefix", None) or "checkpoint")
            # The v1 checkpoint numbers nothing itself; suffix the step so a cadence keeps a history
            # of restartable points rather than overwriting one file.
            written.append(sim.checkpoint("%s_%06d" % (prefix, step)))
        else:
            raise TypeError(
                "output policy must be a pops.output.OutputPolicy / CheckpointPolicy "
                "(got category %r)" % (cat,))
    return written


__all__ = ["policy_due", "fire_output_policies"]
