"""Output / checkpoint run-loop driver (Spec 5 sec.5.14 / case C4, ADC-509).

Wires the typed :class:`pops.output.OutputPolicy` / :class:`pops.output.CheckpointPolicy`
descriptors to the EXISTING :mod:`pops.runtime._system_io` writers (``System.write`` /
``System.checkpoint``). It is a pure-Python run-loop hook: at each macro-step the driver asks
each policy's cadence whether it is DUE and, if so, calls the matching writer. There is NO new
writer subsystem and NO C++ codegen here -- output is a runtime concern, not a kernel concern.

The level selection (``AllLevels`` / ``CoarseOnly`` / ``SelectedLevels``) is a NO-OP on a
single-level ``System``. On ``AmrSystem`` the existing AMR writer emits the coarse fields plus patch
footprints; exact level filtering still belongs to that writer.
"""


def policy_due(cadence, step, phase="step"):
    """True when @p cadence is DUE at macro-step @p step.

    Accepts the typed schedule objects (``pops.time.schedule.every(N)`` / ``always()``) and the
    integer shorthand ``cadence=N`` (== ``every(N)``). ``phase`` is ``"start"`` before the first
    step, ``"step"`` after each macro-step, or ``"end"`` after the run loop. ``None`` means every
    completed step (the default). An ``every(N)`` fires when ``step % N == 0`` (and ``step > 0``);
    ``always`` fires every completed step; ``on_start`` / ``on_end`` fire at their named phases.
    A schedule kind with no run-loop meaning here (``when`` / ``subcycle``) raises rather than
    silently never firing.
    """
    if phase not in ("start", "step", "end"):
        raise ValueError("output phase must be 'start', 'step' or 'end' (got %r)" % (phase,))
    if cadence is None:
        return phase == "step" and step > 0
    if isinstance(cadence, bool):  # guard: bool is an int subclass, never a cadence
        raise TypeError("output cadence must be an int interval or a pops.time schedule, got a bool")
    if isinstance(cadence, int):
        return phase == "step" and step > 0 and step % cadence == 0
    kind = getattr(cadence, "kind", None)
    if kind == "always":
        return phase == "step" and step > 0
    if kind == "every":
        n = int(cadence.params.get("n", 1))
        return phase == "step" and step > 0 and step % n == 0
    if kind == "on_start":
        return phase == "start"
    if kind == "on_end":
        return phase == "end"
    raise NotImplementedError(
        "output cadence schedule kind %r is not honored by the run loop; use every(N), "
        "always(), on_start(), on_end() or an int interval (when/subcycle output needs a "
        "runtime condition/sub-runner)." % (kind,))


def _format_token(fmt):
    """Map a typed output format (``HDF5()``) to a ``System.write`` format string.

    ``HDF5`` -> ``"hdf5"``; ``None`` (no explicit format) -> ``"npz"`` (the dependency-free
    default). Public policies reject strings before this seam; this helper rejects them too.
    """
    from pops.descriptors import reject_string_selector
    if fmt is None:
        return "npz"
    if isinstance(fmt, str):
        reject_string_selector(fmt, "format", "pops.output.NPZ() / VTK() / HDF5()")
    token = getattr(fmt, "native_token", None)
    if token is None or getattr(fmt, "category", None) != "output_format":
        raise TypeError(
            "output format must be a pops.output format descriptor "
            "(NPZ() / VTK() / HDF5()), got %r" % type(fmt).__name__)
    return token


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


def _write_output(sim, prefix, fmt, step, fields, parallel):
    """Call ``sim.write`` with a typed format descriptor through the signature the runtime exposes.

    ``System.write`` accepts ``fields=`` / ``parallel=``. ``AmrSystem.write`` writes the AMR coarse
    visualization plus patch footprints and intentionally has no fields/parallel parameters.
    """
    import inspect

    params = inspect.signature(sim.write).parameters
    if "fields" in params or "parallel" in params:
        return sim.write(prefix, format=fmt, step=step, fields=fields, parallel=parallel)
    return sim.write(prefix, format=fmt, step=step)


def fire_output_policies(sim, policies, step, output_dir, phase="step"):
    """Fire every DUE output / checkpoint policy at macro-step @p step (the run-loop hook).

    For each :class:`pops.output.OutputPolicy` whose cadence is due, call the existing
    ``sim.write(prefix, format=, step=, fields=, parallel=)``; for each
    :class:`pops.output.CheckpointPolicy` whose cadence is due, call the existing
    ``sim.checkpoint(prefix)``. @p phase is ``"start"``, ``"step"`` or ``"end"`` and controls
    ``on_start`` / ``on_end`` schedules. @p output_dir is the directory the files land in (the run
    supplies it); each policy writes ``<output_dir>/<sim-or-policy-prefix>``. Returns the list of
    written paths (useful for tests / logging).

    The level selection on a policy is a documented NO-OP on a single-level System (one level to
    write); AMR per-level filtering is ADC-511. A policy type the driver does not recognise raises
    rather than being silently skipped.
    """
    import os

    written = []
    for policy in policies:
        cat = getattr(policy, "category", None)
        if cat == "output_policy":
            if not policy_due(policy.cadence, step, phase=phase):
                continue
            prefix = os.path.join(output_dir, getattr(policy, "prefix", None) or "output")
            written.append(_write_output(
                sim, prefix, policy.format, step, _field_names(policy.fields),
                _hdf5_parallel(policy.format)))
        elif cat == "checkpoint_policy":
            if not policy_due(policy.cadence, step, phase=phase):
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
