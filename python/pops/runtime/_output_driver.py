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
from __future__ import annotations

from typing import Any


def policy_due(cadence: Any, step: Any, last_step: Any = None, sim: Any = None) -> Any:
    """True when @p cadence is DUE at macro-step @p step (1-based: the count after a step).

    Accepts the typed schedule objects (``pops.time.schedule.every(N)`` / ``always()`` /
    ``on_start()`` / ``on_end()`` / ``when(cond)``) and the integer shorthand ``cadence=N``
    (== ``every(N)``). ``None`` means every step (the default). Behaviors:

    - ``every(N)`` / int ``N``: fires when ``step % N == 0`` (and ``step > 0``).
    - ``always``: fires every step.
    - ``on_start``: fires at step 1 only.
    - ``on_end``: fires when ``step == last_step`` (the LAST step actually taken -- decided in the
      run loop, not pre-guessed). Never fires when @p last_step is unknown (``None``): honest silence
      rather than a wrong fire.
    - ``when(cond)``: fires when the condition holds this step. A CALLABLE ``cond(sim, step)`` is
      evaluated (a non-bool return is rejected); a STRING ``cond`` names a recorded program
      diagnostic and fires iff ``sim.program_diagnostic(cond) != 0.0`` (the native record_scalar map
      is the runtime-condition seam; a missing name raises the existing fail-loud lookup). A Program
      Bool IR value as ``cond`` has no host evaluation seam by design and is refused, the message
      naming the recorded-scalar bridge.

    ``subcycle`` stays refused: subcycles happen INSIDE the native macro step and are invisible to
    this run-loop hook, so an "every subcycle" IO cadence has no meaning at this tier.
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
    if kind == "on_start":
        return step == 1
    if kind == "on_end":
        return last_step is not None and step == last_step
    if kind == "when":
        return _when_due(cadence, step, sim)
    raise NotImplementedError(
        "output cadence schedule kind %r is not honored by the run loop; use every(N), an int "
        "interval, always(), on_start(), on_end() or when(cond). subcycle output is refused: "
        "subcycles are internal to the native macro step and invisible to the run-loop IO hook."
        % (kind,))


def _when_due(cadence, step, sim):
    """Evaluate a ``when(cond)`` cadence at macro-step @p step (see :func:`policy_due`)."""
    cond = cadence.params.get("cond", None)
    if callable(cond):
        result = cond(sim, step)
        if not isinstance(result, bool):
            raise TypeError(
                "when(cond) callable must return a bool (got %r); the run loop fires the policy "
                "iff the callable returns True this step." % (type(result).__name__,))
        return result
    if isinstance(cond, str):
        if sim is None:
            return False
        return sim.program_diagnostic(cond) != 0.0
    raise NotImplementedError(
        "when(cond) supports a callable cond(sim, step) or a string naming a recorded program "
        "diagnostic (fires iff sim.program_diagnostic(name) != 0.0); a Program Bool IR value has no "
        "host evaluation seam -- record it with P.record_scalar and use the recorded-scalar name.")


def _format_token(fmt: Any) -> Any:
    """Map a typed output format (``HDF5()`` / ``Plotfile()``) to a ``System.write`` format string.

    ``HDF5`` -> ``"hdf5"``; ``None`` (no explicit format) -> ``"npz"`` (the dependency-free
    default). A string is passed through (back-compat with ``format="npz"`` authoring). ``Plotfile``
    -> ``"plotfile"`` (the driver routes it to :mod:`pops.runtime._plotfile_writer`, which writes a
    single-level plotfile on a Uniform System -- the former refusal is DELETED, ADC-542 addendum C.1).
    """
    if fmt is None:
        return "npz"
    if isinstance(fmt, str):
        return fmt
    name = type(fmt).__name__
    if name == "HDF5":
        return "hdf5"
    if name == "Plotfile":
        return "plotfile"
    # An unknown typed format: defer to its declared name so System.write rejects it precisely.
    return getattr(fmt, "name", name)


def _hdf5_parallel(fmt: Any) -> Any:
    """True when the typed format requests the parallel-HDF5 path (``HDF5(parallel=True)``)."""
    return bool(getattr(fmt, "parallel", False))


def _field_names(fields: Any) -> Any:
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


def fire_output_policies(sim: Any, policies: Any, step: Any, output_dir: Any,
                         last_step: Any = None) -> Any:
    """Fire every DUE output / checkpoint policy at macro-step @p step (the run-loop hook).

    For each :class:`pops.output.OutputPolicy` whose cadence is due, call the existing
    ``sim.write(prefix, format=, step=, fields=, parallel=)``; for each
    :class:`pops.output.CheckpointPolicy` whose cadence is due, call the existing
    ``sim.checkpoint(prefix)``. @p output_dir is the directory the files land in (the run supplies
    it); each policy writes ``<output_dir>/<sim-or-policy-prefix>``. @p last_step is the LAST step the
    run will take (for ``on_end`` cadences); the run loop supplies it when known. Returns the list of
    written paths (useful for tests / logging).

    The level selection on a policy is a documented NO-OP on a single-level System (one level to
    write); AMR per-level filtering rides the AMR output driver. A policy type the driver does not
    recognise raises rather than being silently skipped.
    """
    import os

    written = []
    for policy in policies:
        cat = getattr(policy, "category", None)
        if cat == "output_policy":
            if not policy_due(policy.cadence, step, last_step=last_step, sim=sim):
                continue
            fmt = _format_token(policy.format)
            prefix = os.path.join(output_dir, getattr(policy, "prefix", None) or "output")
            if fmt == "plotfile":
                from pops.runtime._plotfile_writer import write_plotfile
                written.append(write_plotfile(sim, prefix, step=step))
            else:
                written.append(sim.write(
                    prefix, format=fmt, step=step, fields=_field_names(policy.fields),
                    parallel=_hdf5_parallel(policy.format)))
        elif cat == "checkpoint_policy":
            if not policy_due(policy.cadence, step, last_step=last_step, sim=sim):
                continue
            prefix = os.path.join(output_dir, getattr(policy, "prefix", None) or "checkpoint")
            # The checkpoint numbers nothing itself; suffix the step so a cadence keeps a history
            # of restartable points rather than overwriting one file.
            written.append(sim.checkpoint("%s_%06d" % (prefix, step)))
        else:
            raise TypeError(
                "output policy must be a pops.output.OutputPolicy / CheckpointPolicy "
                "(got category %r)" % (cat,))
    return written


__all__ = ["policy_due", "fire_output_policies"]
