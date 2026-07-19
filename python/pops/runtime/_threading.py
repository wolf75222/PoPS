"""Internal environment preparation and inspection for the compiled Kokkos runtime.

The compute backend is COMPILED into _pops. ``set_threads`` configures a thread-based Kokkos
execution space such as OpenMP; it never changes Serial into OpenMP and never changes a CPU build
into CUDA/HIP. At runtime, Kokkos initializes LAZILY at the creation of the 1st System/AmrSystem
and reads the thread environment at that exact moment.
The final public API selects resources before bind and also honors the standard thread environment.
This private module owns the implementation re-exported as :func:`pops.set_threads`; its module path
is not a second public runtime surface and importing it never initializes Kokkos or loads ``_pops``.

``_first_system_built`` is the shared mutable flag : read here and by
``doctor``, and WRITTEN by ``System.__init__`` / ``AmrSystem.__init__`` via
``_threading._first_system_built = True`` (a module attribute, not a cross-file ``global`` rebind).
All readers/writers live in ``pops.runtime``, so the flag never leaks across layers.
"""
from __future__ import annotations

from typing import Any

_first_system_built = False

# POPS_THREADS supplies the low-level helper's default count. An explicit argument wins;
# the env only supplies the default. Transparent coercion: an
# unparseable or non-positive value is ignored (falls back to os.cpu_count()), never raised.
_THREADS_ENV_VAR = "POPS_THREADS"


def _threads_from_env() -> Any:
    """Resolve a positive thread count from ``POPS_THREADS``, or None when unset/unusable.

    Returns None (so the caller falls back to ``os.cpu_count()``) when the variable is unset,
    blank, non-integer, or < 1. This mirrors the lenient parsing used elsewhere (POPS_PROFILE,
    POPS_FOREACH_SERIAL_THRESHOLD): a bad value is ignored, not rejected.
    """
    import os
    raw = os.environ.get(_THREADS_ENV_VAR)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw.strip())
    except ValueError:
        return None
    return value if value >= 1 else None


def has_kokkos() -> Any:
    """True if _pops was compiled with Kokkos, False if it was built without Kokkos.

    None if the module is too old to expose the info (attribute __has_kokkos__ absent)."""
    from pops import _pops
    return getattr(_pops, "__has_kokkos__", None)


def set_threads(n: Any = None) -> None:
    """Prepare thread environment before native initialization.

    Equivalent to exporting OMP_NUM_THREADS=n before launching Python, but without touching the shell. Has
    an effect only for a thread-based Kokkos backend such as OpenMP, and MUST be called BEFORE the
    1st System/AmrSystem (Kokkos initializes lazily at that moment and reads the setting once) :

        import pops
        pops.set_threads(8)

    With no argument the default is taken from ``POPS_THREADS`` (a positive integer); an explicit
    ``n`` ALWAYS wins, and an unset / unparseable env value falls back to ``os.cpu_count()``.

    A module built without Kokkos or a late call is flagged by a warning (without raising)."""
    import os
    import warnings
    if n is None:                       # default : POPS_THREADS, else all logical cores
        n = _threads_from_env()
        if n is None:
            n = os.cpu_count() or 1
    n = int(n)
    if n < 1:
        raise ValueError("thread count must be >= 1")
    # Source of truth : the REAL state of the Kokkos runtime (covers ALL lazy init paths --
    # System, AmrSystem, DSL .so, direct use of _pops). The Python flag stays the fallback for
    # an old module without the binding.
    from pops import _pops
    _kokkos_started = getattr(_pops, "kokkos_is_initialized", lambda: _first_system_built)()
    if _kokkos_started or _first_system_built:
        warnings.warn(
            "pops.set_threads() was called after native initialization; the request has no effect",
            RuntimeWarning, stacklevel=2)
        return
    if has_kokkos() is False:
        warnings.warn(
            "pops.set_threads() cannot affect a module built without Kokkos; the request is "
            "ignored at compute time.", RuntimeWarning, stacklevel=2)
    # We write the env even in case of doubt (harmless) : a DSL .so with backend='production' compiled with
    # Kokkos will also read OMP_NUM_THREADS at its initialization.
    # We set TWO variables to be agnostic to the backend that Kokkos was compiled with :
    #   - OMP_NUM_THREADS  : read by the OpenMP device (usual case) ;
    #   - KOKKOS_NUM_THREADS : read by Kokkos::initialize whatever the device (OpenMP OR Threads),
    #     useful if the installed Kokkos (e.g. conda-forge) uses the Threads backend and not OpenMP.
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["KOKKOS_NUM_THREADS"] = str(n)
    # OMP_PROC_BIND=false ONLY on macOS (avoids libomp warnings/oversubscription on
    # dev Macs). On Linux/cluster we impose NOTHING : disabling affinity there would degrade
    # NUMA scaling, and a SLURM job that exports OMP_PROC_BIND=close/spread stays in control (setdefault
    # would not override it anyway).
    import sys as _s
    if _s.platform == "darwin":
        os.environ.setdefault("OMP_PROC_BIND", "false")


def parallel_info() -> Any:
    """Parallelism state : compiled backend, current OMP_NUM_THREADS, Kokkos init already done."""
    import os
    return {
        "has_kokkos": has_kokkos(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "first_system_built": _first_system_built,
    }
