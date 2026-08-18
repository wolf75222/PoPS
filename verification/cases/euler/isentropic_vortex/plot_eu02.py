#!/usr/bin/env python3
"""REAL EU-02 plots from native fields. Writes only under build/verification.

Refuses fixture labels. Requires an EvidenceBundle or snapshots.npz from a
campaign dump. Shared colour scales, units, SHA, and leaf captions.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

_CASE_DIR = Path(__file__).resolve().parent
_REPO = Path(__file__).resolve().parents[4]


def _sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() or "unknown"


def _load_siblings():
    from verification.pops_verify.case_authoring import load_sibling_module

    return (
        load_sibling_module(_CASE_DIR / "exact.py"),
        load_sibling_module(_CASE_DIR / "run.py"),
        load_sibling_module(_CASE_DIR / "analyze.py"),
    )


def _caption(sha: str, leaf: str) -> str:
    return f"campaign result | EU-02 | SHA {sha} | leaf {leaf}"


def _save(figure, path: Path, caption: str):
    figure.text(0.01, 0.01, caption, fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140, bbox_inches="tight")


def plot_bundle(series_dir: str | Path, build_dir: str | Path | None = None) -> dict:
    """Render REAL contours, quiver, radial cuts, trajectory, and contact sheet."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from verification.pops_verify.evidence_bundle import EvidenceBundle

    exact, run, analyze = _load_siblings()
    series = Path(series_dir)
    bundle = EvidenceBundle(series)
    sha = _sha()
    leaf = str(bundle.records[0]["leaf_sha256"])[:12]
    caption = _caption(sha, leaf)
    dest = Path(build_dir or (_REPO / "build" / "verification" / "EU-02" / "real_plots"))
    dest.mkdir(parents=True, exist_ok=True)
    campaign = analyze.campaign_from_evidence(series)
    finest = int(campaign["resolutions"][-1])
    t_end = float(campaign["t_end"])
    packed = campaign["fields"][finest]
    primitives = run.conserved_to_primitives(packed)
    oracle = run.average_primitives(finest, t_end)
    x, y, width = run.cell_centers(finest)
    vorticity = analyze.vorticity_from_velocity(primitives["u"], primitives["v"], width)
    vorticity_exact = exact.exact_vorticity(x, y, t_end)
    rho_lim = [
        float(min(oracle["rho"].min(), primitives["rho"].min())),
        float(max(oracle["rho"].max(), primitives["rho"].max())),
    ]
    p_lim = [
        float(min(oracle["p"].min(), primitives["p"].min())),
        float(max(oracle["p"].max(), primitives["p"].max())),
    ]
    w_peak = float(max(abs(vorticity).max(), abs(vorticity_exact).max(), 1.0e-16))
    fields = {
        "rho": (oracle["rho"], primitives["rho"], rho_lim, "rho"),
        "p": (oracle["p"], primitives["p"], p_lim, "p"),
        "vorticity": (vorticity_exact, vorticity, [-w_peak, w_peak], "vorticity"),
    }
    written = []
    for name, (exact_f, num_f, limits, unit) in fields.items():
        figure, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), constrained_layout=True)
        for axis, field, title, cmap in zip(
            axes,
            (exact_f, num_f, num_f - exact_f),
            (f"exact {name}", f"numerical {name}", f"signed error {name}"),
            ("viridis", "viridis", "RdBu_r"),
            strict=True,
        ):
            if title.startswith("signed"):
                peak = float(max(abs(field).max(), 1.0e-16))
                vmin, vmax = -peak, peak
            else:
                vmin, vmax = limits
            mesh = axis.contourf(x, y, field, levels=24, cmap=cmap, vmin=vmin, vmax=vmax)
            figure.colorbar(mesh, ax=axis, label=unit)
            axis.set_aspect("equal")
            axis.set_xlabel("x")
            axis.set_ylabel("y")
            axis.set_title(title)
        _save(figure, dest / f"triptych_{name}_t{t_end:g}.png", caption)
        plt.close(figure)
        written.append(str(dest / f"triptych_{name}_t{t_end:g}.png"))

    figure, axes = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    skip = max(1, finest // 16)
    axes.contourf(x, y, primitives["rho"], levels=20, cmap="viridis", vmin=rho_lim[0], vmax=rho_lim[1])
    axes.quiver(
        x[::skip, ::skip],
        y[::skip, ::skip],
        primitives["u"][::skip, ::skip],
        primitives["v"][::skip, ::skip],
        color="k",
    )
    axes.streamplot(
        x[0, :],
        y[:, 0],
        primitives["u"],
        primitives["v"],
        color="0.3",
        density=0.8,
    )
    axes.set_aspect("equal")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_title("velocity quiver / streamlines")
    _save(figure, dest / f"velocity_quiver_t{t_end:g}.png", caption)
    plt.close(figure)
    written.append(str(dest / f"velocity_quiver_t{t_end:g}.png"))

    center = exact.analytic_center(t_end)
    dx = exact.minimum_image(x - center[0])
    dy = exact.minimum_image(y - center[1])
    radius = np.sqrt(dx * dx + dy * dy)
    order = np.argsort(radius.ravel())
    r = radius.ravel()[order]
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), constrained_layout=True)
    for axis, field, oracle_f, title in zip(
        axes,
        (primitives["rho"], primitives["p"], vorticity),
        (oracle["rho"], oracle["p"], vorticity_exact),
        ("rho(r)", "p(r)", "vorticity(r)"),
        strict=True,
    ):
        axis.plot(r, oracle_f.ravel()[order], label="exact cell-average / analytic")
        axis.plot(r, field.ravel()[order], ".", ms=2, label="numerical")
        axis.set_xlabel("r")
        axis.set_ylabel(title)
        axis.legend(fontsize=7)
    _save(figure, dest / f"radial_cuts_t{t_end:g}.png", caption)
    plt.close(figure)
    written.append(str(dest / f"radial_cuts_t{t_end:g}.png"))

    if len(campaign["resolutions"]) >= 4:
        claim = analyze.evaluate_order_claim(campaign)
        if claim.get("spacings") and claim.get("linf"):
            figure, axes = plt.subplots(figsize=(5.6, 4.4), constrained_layout=True)
            axes.loglog(claim["spacings"], claim["l1"], "o-", label="L1")
            axes.loglog(claim["spacings"], claim["l2"], "s-", label="L2")
            axes.loglog(claim["spacings"], claim["linf"], "D-", label="Linf")
            href = np.asarray(claim["spacings"], dtype=np.float64)
            axes.loglog(href, claim["linf"][0] * (href / href[0]) ** 2, "k--", label="order 2")
            axes.set_xlabel("h")
            axes.set_ylabel(f"error ({claim['family']})")
            axes.legend()
            axes.set_title(f"EU-02 {claim['family']} convergence")
            _save(figure, dest / "convergence.png", caption)
            plt.close(figure)
            written.append(str(dest / "convergence.png"))

    snap = series / f"n{finest:03d}" / "snapshots.npz"
    if not snap.is_file():
        for child in series.rglob("snapshots.npz"):
            snap = child
            break
    if snap.is_file():
        data = np.load(snap)
        times = [float(v) for v in data["times"]]
        ax_c, ay_c, nx_c, ny_c = [], [], [], []
        figure, axes = plt.subplots(1, len(times), figsize=(3.2 * len(times), 3.4), constrained_layout=True)
        if len(times) == 1:
            axes = [axes]
        for index, instant in enumerate(times):
            packed_t = data[f"t_{index}"]
            prim = run.conserved_to_primitives(packed_t)
            axes[index].contourf(x, y, prim["rho"], levels=16, cmap="viridis", vmin=rho_lim[0], vmax=rho_lim[1])
            axes[index].set_aspect("equal")
            axes[index].set_title(f"t={instant:g}")
            analytic = exact.analytic_center(instant)
            numerical = analyze.vortex_center_from_density(prim["rho"], finest, expected=analytic)
            ax_c.append(analytic[0])
            ay_c.append(analytic[1])
            nx_c.append(numerical[0])
            ny_c.append(numerical[1])
        _save(figure, dest / "contact_sheet_rho.png", caption)
        plt.close(figure)
        written.append(str(dest / "contact_sheet_rho.png"))
        figure, axes = plt.subplots(figsize=(5.4, 5.0), constrained_layout=True)
        axes.plot(ax_c, ay_c, "k-", label="analytic centre")
        axes.plot(nx_c, ny_c, "o", label="numerical centre")
        axes.set_aspect("equal")
        axes.set_xlabel("x_c")
        axes.set_ylabel("y_c")
        axes.legend()
        _save(figure, dest / "vortex_center_trajectory.png", caption)
        plt.close(figure)
        written.append(str(dest / "vortex_center_trajectory.png"))
        try:
            import imageio.v2 as imageio

            frames = []
            for index, instant in enumerate(times):
                packed_t = data[f"t_{index}"]
                prim = run.conserved_to_primitives(packed_t)
                fig, ax = plt.subplots(figsize=(4.2, 4.0), constrained_layout=True)
                ax.contourf(x, y, prim["rho"], levels=16, cmap="viridis", vmin=rho_lim[0], vmax=rho_lim[1])
                ax.set_aspect("equal")
                ax.set_title(f"t={instant:g}")
                fig.text(0.01, 0.01, caption, fontsize=7)
                frame_path = dest / f"gif_frame_{index:02d}.png"
                fig.savefig(frame_path, dpi=100)
                plt.close(fig)
                frames.append(imageio.imread(frame_path))
            imageio.mimsave(dest / "eu02_advection.gif", frames, duration=0.4)
            written.append(str(dest / "eu02_advection.gif"))
        except Exception as exc:
            (dest / "gif_error.txt").write_text(f"{type(exc).__name__}: {exc}\n")

    index = {
        "case_id": "EU-02",
        "data_kind": "campaign",
        "repository_sha": sha,
        "leaf_sha256": leaf,
        "series": str(series),
        "plots": written,
        "label": "campaign result",
    }
    (dest / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    (dest / "INDEX.md").write_text(
        "# EU-02 real plots\n\n"
        + f"SHA `{sha}` leaf `{leaf}`\n\n"
        + "\n".join(f"- `{Path(path).name}`" for path in written)
        + "\n",
        encoding="utf-8",
    )
    return index


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("usage: plot_eu02.py SERIES_DIR [BUILD_DIR]")
    plot_bundle(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
