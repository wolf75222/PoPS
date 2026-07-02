"""Shared initial-condition fixtures for the Python test tree.

Importable both under pytest (REPO_ROOT is on sys.path via the rootdir) and from a
process-isolated script (conftest._process_pythonpath puts REPO_ROOT on PYTHONPATH).

The duplicated ``_bubble`` / ``initial_state`` copies were NOT all the same shape, so the divergent
variants are kept as distinct, explicitly named helpers here rather than collapsed into one -- a
single ``_bubble`` would silently change the physics of whichever group did not match. Call sites
alias the one they need (``from tests.python.support.initial_states import bubble_amr as _bubble``).
"""

from __future__ import annotations

import numpy as np


def bubble_amr(n: int) -> np.ndarray:
    """A centered density bubble, amplitude 0.5, width 0.02 (the AMR/production copies).

    Canonical copy of the ``_bubble`` used by the AMR hybrid / spec5-e2e / production-amr tests.
    """
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    return (1.0 + 0.5 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)).reshape(-1)


def bubble_offset(n: int) -> np.ndarray:
    """An off-center density bubble, amplitude 0.4, sharp width, xy indexing (the bind/freeze copies).

    Distinct from :func:`bubble_amr`: centered at (0.4, 0.5) with a sharper ``exp(-50 * r^2)`` profile.
    Canonical copy of the ``_bubble`` used by the bind-adapters / freeze-lifecycle tests.
    """
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    return (1.0 + 0.4 * np.exp(-50.0 * ((X - 0.4) ** 2 + (Y - 0.5) ** 2))).reshape(-1)


def euler_bubble_state(n: int, gamma: float) -> list:
    """A 4-var Euler state: a centered density bubble at rest, energy ``1/(gamma-1)`` (flat).

    Canonical copy of the ``initial_state(n)`` used by the DSL compile-cache tests. Parameterized by
    @p gamma because the copies differed only there (a named ``GAMMA`` vs a literal ``1.6667``).
    Returns a flat Python list (component-major), the shape those tests feed to the native install.
    """
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    U = np.zeros((4, n, n))
    U[0] = 1.0 + 0.3 * np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.02)
    U[3] = 1.0 / (gamma - 1.0)
    return U.reshape(-1).tolist()
