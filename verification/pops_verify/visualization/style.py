"""Headless Matplotlib style for Phase 8 (plan §40.3, §40.4)."""
from __future__ import annotations

from typing import Any

RENDERER_VERSION = "pops.verification.visuals.v1"
RENDERER_SCRIPT = "scripts/render_verification_visuals.py"
HASH_SALT = "pops-verification-visuals-v1"

_RC: dict[str, Any] = {
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "font.size": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "svg.hashsalt": HASH_SALT,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.prop_cycle": None,
}


def configure_matplotlib() -> Any:
    import os

    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from cycler import cycler

    rc = dict(_RC)
    rc["axes.prop_cycle"] = cycler(
        color=["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#000000"]
    )
    plt.rcParams.update(rc)
    return plt
