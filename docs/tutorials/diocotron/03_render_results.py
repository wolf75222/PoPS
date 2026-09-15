"""Render only numerical snapshots saved by tutorials 01, 02 and 04.

Example: python 03_render_results.py /path/to/euler-mode5 /path/to/fan-li15-mode5 \
    --output /path/to/figures

Each input directory is one trajectory; its segment subdirectories are searched
recursively. See SNAPSHOTS.md for the AMR, transform, timing and missing-data rules.
"""
import argparse
import csv
import hashlib
import json
import math
import operator
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.ticker import MaxNLocator, NullFormatter, ScalarFormatter
import numpy as np
from PIL import Image


# 1. Explicit plotting choices, fixed paper times and fixed fit windows.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("runs", nargs="+", type=Path, help="one directory per physical trajectory")
parser.add_argument("--output", required=True, type=Path, help="new output directory")
parser.add_argument("--time-tolerance", type=float, default=1e-8)
parser.add_argument("--normalization-time-max", type=float, default=1e-8,
                    help="latest permitted actual near-zero potential timestamp")
parser.add_argument("--schlieren-strength", type=float, default=1.0)
parser.add_argument("--gradient-scale", type=float,
                    help="density/length; default (initial peak density-background)/radius")
parser.add_argument("--density-min", type=float)
parser.add_argument("--density-max", type=float)
parser.add_argument("--dpi", type=int, default=180)
parser.add_argument("--gif-width", type=int, default=960)
parser.add_argument("--gif-seconds", type=float, default=20.0)
parser.add_argument("--skip-gif", action="store_true")
parser.add_argument("--growth-end", type=float, default=2.0)
parser.add_argument("--context-label", default="",
                    help="scope label displayed on every figure and GIF frame, e.g. 'Integration smoke only'")
args = parser.parse_args()
if (args.time_tolerance < 0 or args.normalization_time_max <= 0
        or args.schlieren_strength <= 0 or args.dpi < 50 or args.gif_width < 320
        or args.gif_seconds <= 0 or args.growth_end <= 0
        or (args.gradient_scale is not None and args.gradient_scale <= 0)):
    parser.error("time, image and transform settings must be finite positive values")
if not all(math.isfinite(value) for value in
           (args.time_tolerance, args.normalization_time_max, args.schlieren_strength,
            args.gif_seconds, args.growth_end)):
    parser.error("settings must be finite")
TARGETS = (.1, 1.25, 2.5, 3.75, 5., 6.25, 7.5, 8.75, 10.)
TIME_LABELS = (r"0.01t_f", r"1/8t_f", r"2/8t_f", r"3/8t_f", r"4/8t_f",
               r"5/8t_f", r"6/8t_f", r"7/8t_f", r"t_f")
FIT_WINDOWS = {3: (.4, .7), 4: (.6, .75), 5: (1.15, 1.35)}
PAPER_RATES = {3: .772, 4: .911, 5: .683}
SIGNATURE_KEYS = ("model", "mode", "radius", "ring", "alpha", "omega", "temperature",
                  "background", "mean_ring", "perturbation", "nr", "ntheta", "max_levels",
                  "cfl", "max_dt", "split", "source", "spatial", "artifact_identity")
OPTIONAL_SIGNATURE_KEYS = ("coarse_max_grid", "cluster_max_grid", "distribute_coarse",
                           "potential_history_slot", "potential_history_contract",
                           "potential_history_transfer", "field_initial_guess", "time_calendar",
                           "field_coarse_method", "field_coarse_restart", "field_coarse_iteration_cap",
                           "field_coarse_preconditioner",
                           "field_interface_coupling", "source_rotation", "transport_path",
                           "transport_conserved_components", "transport_regularized_indices",
                           "admissibility", "snapshot_raw_moments",
                           "output_interval", "growth_output_interval", "growth_output_end")
COLORS = plt.colormaps["Blues"](np.linspace(0, 1, 256))
COLORS[0] = (1., 1., 1., 1.)
COLOR_MAP = ListedColormap(COLORS)
OUTSIDE = "#53586e"
SERIES_COLORS = ("#1f77b4", "#d97706", "#556b2f", "#8559a5", "#b14f75")
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
summary = {"contract": 1, "input_directories": [str(path.resolve()) for path in args.runs],
           "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "requested_density_times": TARGETS, "paper_final_time": 10.,
           "time_tolerance": args.time_tolerance, "context_label": args.context_label,
           "runs": [], "warnings": [], "growth_outputs": [],
           "sources": ["https://arxiv.org/abs/2510.11808",
                       "https://ccap.hep.ph.ic.ac.uk/trac/raw-attachment/wiki/Research/LhARA/GaborLens/Literature/1998_Davidson.pdf"]}


# 2. Read actual snapshot headers; merge only demonstrably identical segment overlaps.
trajectories = []
for run_number, directory in enumerate(args.runs):
    directory = directory.resolve()
    paths = sorted(directory.rglob("snapshot-*.npz")) if directory.is_dir() else []
    label = "%s-%s" % (run_number + 1, re.sub(r"[^A-Za-z0-9_.-]+", "-", directory.name))
    run_summary = {"label": label, "directory": str(directory), "snapshots": [],
                   "duplicates": [], "missing_panel_times": [], "outputs": [], "warnings": []}
    summary["runs"].append(run_summary)
    if not paths:
        run_summary["status"] = "no numerical snapshots; no figures or GIF generated"
        run_summary["missing_panel_times"] = list(TARGETS)
        continue
    records, by_step = [], {}
    signature = None
    for path in paths:
        with np.load(path, allow_pickle=False) as stored:
            metadata = json.loads(str(stored["metadata"].item()))
            parameters = json.loads(str(stored["parameters"].item()))
            patches = json.loads(str(stored["patches"].item()))
            candidate = {key: parameters[key] for key in SIGNATURE_KEYS}
            candidate.update({key: parameters.get(key) for key in OPTIONAL_SIGNATURE_KEYS})
            if signature is None:
                signature = candidate
            elif candidate != signature:
                raise ValueError("one input directory must contain one unchanged case: %s" % path)
            time_value, step = float(metadata["time"]), int(metadata["macro_step"])
            potential_time = metadata["potential_time"]
            levels = int(metadata["levels"])
            if (not math.isfinite(time_value) or time_value < 0 or step < 0 or levels < 1
                    or not math.isfinite(float(metadata["mass"]))):
                raise ValueError("invalid accepted-state metadata: %s" % path)
            q_keys = {"q0_level%d" % level for level in range(levels)}
            psi_keys = {"psi_level%d" % level for level in range(levels)}
            moment_keys = {key for key in stored.files if key.startswith("moments_level")}
            if {key for key in stored.files if key.startswith("q0_level")} != q_keys:
                raise ValueError("incomplete density hierarchy: %s" % path)
            if parameters["model"] == "FanLi15":
                if (parameters.get("snapshot_raw_moments") !=
                        "all fifteen q=r*M components, q-outer ordering"
                        or moment_keys != {"moments_level%d" % level for level in range(levels)}):
                    raise ValueError("FanLi15 requires its complete declared raw-moment hierarchy: %s" % path)
                for level in range(levels):
                    moments = stored["moments_level%d" % level]
                    q0 = stored["q0_level%d" % level]
                    if (moments.dtype != np.dtype(np.float64) or moments.shape != (15,) + q0.shape
                            or not np.array_equal(moments[0], q0)):
                        raise ValueError("FanLi15 raw-moment shape, precision or density disagrees: %s" % path)
            if potential_time is None:
                if step != 0 or any(key.startswith("psi_level") for key in stored.files):
                    raise ValueError("only the true initial state can omit its potential: %s" % path)
            elif (not math.isfinite(float(potential_time)) or not 0 <= float(potential_time) <= time_value
                  or {key for key in stored.files if key.startswith("psi_level")} != psi_keys):
                raise ValueError("invalid midpoint potential hierarchy/timestamp: %s" % path)
            if step in by_step:
                previous = by_step[step]
                with np.load(previous["path"], allow_pickle=False) as earlier:
                    same = patches == previous["patches"]
                    # initial_mass and elapsed_seconds are segment-local diagnostics.
                    for key in ("time", "potential_time", "macro_step", "mass", "levels"):
                        same = same and metadata[key] == previous["metadata"][key]
                    same = same and moment_keys == {
                        key for key in earlier.files if key.startswith("moments_level")}
                    for key in q_keys | moment_keys | (psi_keys if potential_time is not None else set()):
                        same = same and key in earlier.files and np.array_equal(stored[key], earlier[key])
                if not same:
                    raise ValueError("conflicting snapshots at macro step %d: %s" % (step, path))
                run_summary["duplicates"].append({"path": str(path), "same_state_as": str(previous["path"])})
                continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        record = {"path": path, "time": time_value, "macro_step": step, "metadata": metadata,
                  "parameters": parameters, "patches": patches, "sha256": digest.hexdigest()}
        records.append(record)
        by_step[step] = record
    records.sort(key=operator.itemgetter("time", "macro_step"))
    if any(right["time"] <= left["time"] or right["macro_step"] <= left["macro_step"]
           for left, right in zip(records, records[1:], strict=False)):
        raise ValueError("accepted times and macro steps must increase: %s" % directory)
    parameters = records[0]["parameters"]
    if parameters["model"] not in ("Euler", "HYQMOM15", "FanLi15") or int(parameters["mode"]) not in FIT_WINDOWS:
        raise ValueError("this renderer accepts only the authored diocotron models and modes")
    run_summary["parameters"] = parameters
    run_summary["status"] = "actual snapshots available"
    trajectories.append((records, run_summary))


# 3. Prepare only the requested panels for which a saved density state exists.
for records, run_summary in trajectories:
    parameters = records[0]["parameters"]
    qualified_potential_times = type(parameters.get("potential_history_slot")) is int \
        and parameters["potential_history_slot"] == 1
    qualified_potential_transfer = (
        parameters.get("potential_history_contract") == "scalar-output-field-v1"
        and parameters.get("potential_history_transfer") == "authenticated-1to1-retain-overlap-v1")
    qualified_potential = qualified_potential_times and qualified_potential_transfer
    run_summary["potential_timestamp_status"] = (
        "newest accepted raw history slot 1" if qualified_potential_times else
        "unqualified legacy slot convention; Fourier samples omitted")
    run_summary["potential_transfer_status"] = (
        "authenticated equal-clock scalar output; retained overlap preserved"
        if qualified_potential_transfer else
        "unqualified legacy AMR history transfer; Fourier samples omitted")
    if not qualified_potential:
        summary["warnings"].append(run_summary["label"] +
            ": potential timestamps or AMR transfer are unqualified; density figures remain available.")
    radius, ring_radius = float(parameters["radius"]), float(parameters["ring"][0])
    nr, ntheta, mode = int(parameters["nr"]), int(parameters["ntheta"]), int(parameters["mode"])
    density_min = float(parameters["background"] if args.density_min is None else args.density_min)
    density_max = float(parameters["mean_ring"] + parameters["perturbation"]
                        if args.density_max is None else args.density_max)
    gradient_scale = float(args.gradient_scale if args.gradient_scale is not None else
                           (parameters["mean_ring"] + parameters["perturbation"] - parameters["background"]) / radius)
    if not (nr >= 2 and ntheta >= 2*mode+1 and 0 < ring_radius < radius
            and math.isfinite(density_min) and math.isfinite(density_max) and density_min < density_max
            and math.isfinite(gradient_scale) and gradient_scale > 0):
        raise ValueError("invalid polar grid or plotting normalization")
    density_norm, schlieren_norm = Normalize(density_min, density_max), Normalize(0, 1)
    run_summary["display"] = {"density_color_limits": [density_min, density_max],
        "schlieren_transform": "1 - exp(-strength * abs(physical_grad_rho) / gradient_scale)",
        "schlieren_strength": args.schlieren_strength, "gradient_scale": gradient_scale,
        "schlieren_color_limits": [0., 1.], "colormap": "Blues, first color replaced by white",
        "gradient": "centered available same-level neighbors; one-sided at patch/radial edges; periodic theta",
        "spatial_display": "native polar cells, covered coarse cells masked; no image-space interpolation"}
    available, selected_steps = [], {}
    saved_times = np.array([record["time"] for record in records])
    for target_index, target in enumerate(TARGETS):
        index = int(np.argmin(np.abs(saved_times-target)))
        if abs(saved_times[index]-target) <= args.time_tolerance:
            if records[index]["macro_step"] in selected_steps:
                raise ValueError("one saved state cannot represent multiple requested times; reduce the tolerance")
            available.append((target_index, index))
            selected_steps[records[index]["macro_step"]] = len(available)-1
        else:
            run_summary["missing_panel_times"].append(target)
    plate = None
    if available:
        columns = min(3, len(available))
        rows = math.ceil(len(available)/columns)
        plate, plate_axes = plt.subplots(rows, columns, figsize=(3.6*columns, 3.8*rows), squeeze=False)
        plate_axes = plate_axes.ravel()
        for unused in plate_axes[len(available):]:
            unused.remove()
        for axis_index, (target_index, _record_index) in enumerate(available):
            plate_axes[axis_index].set_title("(%s) $t=%s$" % (chr(97+target_index), TIME_LABELS[target_index]), y=-.11)
        plate.suptitle("%s, mode %d — %d/9 saved paper times" % (parameters["model"], mode, len(available))
                      + ("\n" + args.context_label if args.context_label else ""))
    angular_count = ntheta * 2**(max(record["metadata"]["levels"] for record in records)-1)
    circle_theta = (np.arange(angular_count)+.5) * (2*np.pi/angular_count)
    run_summary["fourier"] = {"radius": ring_radius, "angular_samples": angular_count,
        "coefficient": "sum(psi(r,theta_j) * exp(-i*mode*theta_j)) / Ntheta",
        "spatial_sampling": "periodic bilinear interpolation of valid cell-center potential; finest level with all four corners available",
        "time": "stored potential_time, the actual source midpoint", "samples": []}
    frames, frame_times = [], []
    palette = Image.new("P", (1, 1))
    palette_colors = (255 * COLOR_MAP(np.linspace(0, 1, 252))[:, :3]).astype(np.uint8).reshape(-1).tolist()
    palette.putpalette(palette_colors + [83, 88, 110, 0, 0, 0, 255, 255, 255, 128, 128, 128])


    # 4. Reconstruct ownership from the exact inclusive native patch boxes.
    for record in records:
        metadata, patches = record["metadata"], record["patches"]
        levels = int(metadata["levels"])
        if (not patches["built"] or int(patches["n_levels"]) != levels
                or tuple(patches["base_cells"]) != (nr, ntheta)
                or not np.allclose(patches["domain_bounds"], [[0, 0], [radius, 2*np.pi]], rtol=0, atol=1e-12)):
            raise ValueError("snapshot and patch-table geometry disagree: %s" % record["path"])
        level_reports = {int(entry["level"]): entry for entry in patches["per_level"]}
        if set(level_reports) != set(range(levels)):
            raise ValueError("patch table must report every stored level")
        masks = []
        for level in range(levels):
            factor = 2**level
            valid = np.ones((ntheta, nr), dtype=bool) if level == 0 else np.zeros((ntheta*factor, nr*factor), dtype=bool)
            entry = level_reports[level]
            if level == 0 and entry["boxes"]:
                raise ValueError("unexpected base-level boxes; inspect the changed native report contract")
            if level and len(entry["boxes"]) != int(entry["n_patches"]):
                raise ValueError("fine patch census disagrees with its boxes")
            for bounds in entry["boxes"]:
                if len(bounds) != 4 or any(int(value) != value for value in bounds):
                    raise ValueError("expected ranked inclusive (lo_r,lo_theta,hi_r,hi_theta)")
                lo_r, lo_theta, hi_r, hi_theta = map(int, bounds)
                if not (0 <= lo_r <= hi_r < nr*factor and 0 <= lo_theta <= hi_theta < ntheta*factor):
                    raise ValueError("patch box falls outside its level")
                valid[lo_theta:hi_theta+1, lo_r:hi_r+1] = True
            if level and int(valid.sum()) != int(entry["cells"]):
                raise ValueError("overlapping boxes or inconsistent fine-cell census")
            masks.append(valid)
        owned = [valid.copy() for valid in masks]
        for level in range(levels-1):
            coarse_shape = masks[level].shape
            children = masks[level+1].reshape(coarse_shape[0], 2, coarse_shape[1], 2)
            any_child, all_children = children.any(axis=(1, 3)), children.all(axis=(1, 3))
            if np.any(any_child != all_children) or np.any(any_child & ~masks[level]):
                raise ValueError("fine coverage must be nested and aligned with complete parent cells")
            owned[level] &= ~all_children


        # 5. Recover physical cell density and its polar gradient without reading invalid fine cells.
        native_fields, composite_mass = [], 0.0
        sample_summary = {"path": str(record["path"]), "sha256": record["sha256"],
                          "metadata": metadata, "active_cells": 0, "negative_density_cells": 0,
                          "density_below_color_min": 0, "density_above_color_max": 0,
                          "schlieren_above_0_999": 0, "maximum_gradient": 0.0}
        circle, circle_level = np.full(angular_count, np.nan), np.full(angular_count, -1, dtype=int)
        with np.load(record["path"], allow_pickle=False) as stored:
            for level in range(levels):
                factor, valid, active = 2**level, masks[level], owned[level]
                dr, dtheta = radius/(nr*factor), 2*np.pi/(ntheta*factor)
                centers_r = (np.arange(nr*factor)+.5)*dr
                q0 = np.asarray(stored["q0_level%d" % level], dtype=np.float64)
                if q0.shape != valid.shape or not np.all(np.isfinite(q0[valid])):
                    raise ValueError("nonfinite or incorrectly shaped valid density cells")
                if parameters["model"] == "FanLi15":
                    moments = stored["moments_level%d" % level]
                    if not np.all(np.isfinite(moments[:, valid])):
                        raise ValueError("nonfinite valid FanLi15 raw-moment cells")
                    sample_summary["raw15_validated_cells"] = (
                        sample_summary.get("raw15_validated_cells", 0) + int(valid.sum()))
                rho = np.where(valid, q0/centers_r[None, :], 0.)
                right, left = np.roll(valid, -1, axis=1), np.roll(valid, 1, axis=1)
                right[:, -1], left[:, 0] = False, False
                if np.any(valid & ~(right | left)):
                    raise ValueError("isolated radial cell has no supported gradient stencil")
                rho_right, rho_left = np.roll(rho, -1, axis=1), np.roll(rho, 1, axis=1)
                radial_gradient = np.where(right & left, (rho_right-rho_left)/(2*dr),
                                            np.where(right, (rho_right-rho)/dr, (rho-rho_left)/dr))
                forward, backward = np.roll(valid, -1, axis=0), np.roll(valid, 1, axis=0)
                if np.any(valid & ~(forward | backward)):
                    raise ValueError("isolated angular cell has no supported gradient stencil")
                rho_forward, rho_backward = np.roll(rho, -1, axis=0), np.roll(rho, 1, axis=0)
                angular_gradient = np.where(forward & backward, (rho_forward-rho_backward)/(2*dtheta),
                                              np.where(forward, (rho_forward-rho)/dtheta, (rho-rho_backward)/dtheta))
                gradient = np.hypot(radial_gradient, angular_gradient/centers_r[None, :])
                if not np.all(np.isfinite(gradient[valid])):
                    raise ValueError("nonfinite physical density gradient")
                schlieren = -np.expm1(-args.schlieren_strength*gradient/gradient_scale)
                edges_r = np.arange(nr*factor+1)*dr
                edges_theta = np.arange(ntheta*factor+1)*dtheta
                x_edges = np.cos(edges_theta)[:, None]*edges_r[None, :]
                y_edges = np.sin(edges_theta)[:, None]*edges_r[None, :]
                native_fields.append((x_edges, y_edges, rho, schlieren, active))
                composite_mass += float(q0[active].sum())*dr*dtheta
                sample_summary["active_cells"] += int(active.sum())
                sample_summary["negative_density_cells"] += int((rho[active] < 0).sum())
                sample_summary["density_below_color_min"] += int((rho[active] < density_min).sum())
                sample_summary["density_above_color_max"] += int((rho[active] > density_max).sum())
                sample_summary["schlieren_above_0_999"] += int((schlieren[active] > .999).sum())
                if np.any(active):
                    sample_summary["maximum_gradient"] = max(sample_summary["maximum_gradient"], float(gradient[active].max()))


                # 6. Sample only valid potential values, at their actual midpoint time.
                if metadata["potential_time"] is not None and qualified_potential:
                    psi = np.asarray(stored["psi_level%d" % level], dtype=np.float64)
                    if psi.shape != valid.shape or not np.all(np.isfinite(psi[valid])):
                        raise ValueError("nonfinite or incorrectly shaped valid potential cells")
                    radial_position = ring_radius/dr-.5
                    ir = math.floor(radial_position)
                    radial_fraction = radial_position-ir
                    if not 0 <= ir < psi.shape[1]-1:
                        raise ValueError("the Fourier circle needs two bracketing radial centers")
                    angular_position = (np.arange(angular_count)+.5)/(angular_count/psi.shape[0])-.5
                    angular_lower = np.floor(angular_position).astype(int)
                    angular_fraction = angular_position-angular_lower
                    j0, j1 = angular_lower % psi.shape[0], (angular_lower+1) % psi.shape[0]
                    supported = valid[j0, ir] & valid[j0, ir+1] & valid[j1, ir] & valid[j1, ir+1]
                    radial_values = (1-radial_fraction)*psi[:, ir] + radial_fraction*psi[:, ir+1]
                    angular_values = (1-angular_fraction)*radial_values[j0] + angular_fraction*radial_values[j1]
                    circle[supported] = angular_values[supported]
                    circle_level[supported] = level
        sample_summary["composite_mass_from_snapshot"] = composite_mass
        sample_summary["mass_difference_from_native_metadata"] = composite_mass-float(metadata["mass"])
        run_summary["snapshots"].append(sample_summary)
        if sample_summary["negative_density_cells"]:
            run_summary["warnings"].append("%s contains %d negative physical density cells" %
                (record["path"].name, sample_summary["negative_density_cells"]))
        if metadata["potential_time"] is not None and qualified_potential:
            if np.any(circle_level < 0) or not np.all(np.isfinite(circle)):
                raise ValueError("incomplete valid potential coverage of the Fourier circle")
            coefficient = np.sum(circle*np.exp(-1j*mode*circle_theta))/angular_count
            run_summary["fourier"]["samples"].append({"potential_time": float(metadata["potential_time"]),
                "density_time": record["time"], "macro_step": record["macro_step"],
                "real": float(coefficient.real), "imaginary": float(coefficient.imag),
                "modulus": float(abs(coefficient)),
                "angular_samples_per_level": [int((circle_level == level).sum()) for level in range(levels)]})


        # 7. Render native cells: actual panels and actual density/schlieren GIF frames.
        frame_figure, frame_axes = plt.subplots(1, 2, figsize=(9, 5), dpi=args.gif_width/9)
        frame_axes[0].set_title("Density; fixed range [%g, %g]" % (density_min, density_max))
        frame_axes[1].set_title("Schlieren; $1-e^{-k|\\nabla\\rho|/g_*}$")
        frame_figure.suptitle("%s, mode %d; t=%.10g; accepted step %d" %
                             (parameters["model"], mode, record["time"], record["macro_step"])
                             + ("\n" + args.context_label if args.context_label else ""), fontsize=11)
        frame_figure.text(.5, .03, "k=%g, g*=%g; actual saved density time; no temporal interpolation" %
                          (args.schlieren_strength, gradient_scale), ha="center", fontsize=8)
        destinations = [(frame_axes[0], 2, density_norm), (frame_axes[1], 3, schlieren_norm)]
        if record["macro_step"] in selected_steps:
            destinations.append((plate_axes[selected_steps[record["macro_step"]]], 3, schlieren_norm))
        for axis, field_index, norm in destinations:
            axis.set_facecolor(OUTSIDE)
            axis.set_aspect("equal")
            axis.set_xlim(-1.12*radius, 1.12*radius)
            axis.set_ylim(-1.12*radius, 1.12*radius)
            axis.set_xticks([])
            axis.set_yticks([])
            for fields in native_fields:
                axis.pcolormesh(fields[0], fields[1], np.ma.array(fields[field_index], mask=~fields[4]),
                                cmap=COLOR_MAP, norm=norm, shading="flat", antialiased=False, rasterized=True)
        frame_figure.tight_layout(rect=(0, .06, 1, .90 if args.context_label else .93))
        if record is records[-1]:
            latest_path = output/(run_summary["label"]+"-latest-state.png")
            frame_figure.savefig(latest_path, dpi=args.dpi)
            run_summary["outputs"].append(str(latest_path))
        if not args.skip_gif:
            frame_figure.canvas.draw()
            rgb = np.asarray(frame_figure.canvas.buffer_rgba())[..., :3].copy()
            frames.append(Image.fromarray(rgb).quantize(palette=palette, dither=Image.Dither.NONE))
            frame_times.append(record["time"])
        plt.close(frame_figure)
    if plate is not None:
        plate.text(.5, .01, "$t_f=10$; $S=1-e^{-k|\\nabla\\rho|/g_*}$; k=%g, g*=%g; fixed color range [0,1]" %
                   (args.schlieren_strength, gradient_scale), ha="center", fontsize=9)
        plate.tight_layout(rect=(0, .035, 1, .935 if args.context_label else .965))
        suffix = "-schlieren-panels" + ("-partial" if len(available) < 9 else "")
        for extension in ("png", "pdf"):
            figure_path = output/(run_summary["label"]+suffix+"."+extension)
            plate.savefig(figure_path, dpi=args.dpi, bbox_inches="tight")
            run_summary["outputs"].append(str(figure_path))
        plt.close(plate)
    if len(frames) >= 2:
        intervals = np.diff(frame_times)
        durations = [max(10, int(round(1000*args.gif_seconds*dt/intervals.sum()/10))*10) for dt in intervals]
        durations.append(1000)
        gif_path = output/(run_summary["label"]+"-density-schlieren.gif")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=durations,
                       loop=0, optimize=False, disposal=2)
        run_summary["outputs"].append(str(gif_path))
        run_summary["gif"] = {"frames": len(frames), "density_times": frame_times, "durations_ms": durations,
            "timing": "proportional physical intervals rounded to 10 ms, minimum 10 ms; final frame held 1 s",
            "palette": "one fixed 256-color palette; no temporal interpolation"}
    else:
        run_summary["gif"] = {"status": "disabled" if args.skip_gif else "at least two saved states are required"}
    frames.clear()


    # 8. Normalize by the first real near-zero potential sample, then fit fixed windows.
    samples = run_summary["fourier"]["samples"]
    samples.sort(key=operator.itemgetter("potential_time"))
    if any(right["potential_time"] <= left["potential_time"] for left, right in zip(samples, samples[1:], strict=False)):
        raise ValueError("potential timestamps must increase across accepted snapshots")
    fit_lower, fit_upper = FIT_WINDOWS[mode]
    fit = {"window": [fit_lower, fit_upper], "status": "no qualified potential samples"}
    fourier = run_summary["fourier"]
    fourier["fit"] = fit
    if samples:
        initial = samples[0]
        if not 0 <= initial["potential_time"] <= args.normalization_time_max or initial["modulus"] <= 0:
            fourier["normalization_status"] = "missing positive-modulus near-zero potential sample; normalized curve omitted"
        else:
            fourier["normalization_status"] = "normalized by the first actual near-zero sample"
            fourier["normalization_time"] = initial["potential_time"]
            fourier["normalization_modulus"] = initial["modulus"]
            for sample in samples:
                sample["normalized_modulus"] = sample["modulus"]/initial["modulus"]
            times = np.array([sample["potential_time"] for sample in samples])
            amplitude = np.array([sample["normalized_modulus"] for sample in samples])
            fourier["zero_amplitude_samples"] = int((amplitude == 0).sum())
            in_window = (times >= fit_lower) & (times <= fit_upper)
            fit["samples_in_window"] = int(in_window.sum())
            if times[0] > fit_lower or times[-1] < fit_upper or in_window.sum() < 3:
                fit["status"] = "fixed fit window is incomplete or contains fewer than three samples"
            elif np.any(amplitude[in_window] <= 0):
                fit["status"] = "nonpositive observed amplitude in the fixed fit window"
            else:
                design = np.column_stack((np.ones(in_window.sum()), times[in_window]))
                logarithm = np.log(amplitude[in_window])
                intercept, growth = np.linalg.lstsq(design, logarithm, rcond=None)[0]
                residual = logarithm-design@np.array([intercept, growth])
                fit.update(status="fitted", intercept=float(intercept), growth_rate=float(growth),
                           log_rmse=float(np.sqrt(np.mean(residual**2))),
                           log_max_absolute_residual=float(np.abs(residual).max()),
                           slope_standard_error=float(np.sqrt(np.sum(residual**2)/(in_window.sum()-2)
                               /np.sum((times[in_window]-times[in_window].mean())**2))))
        annulus_ratio = parameters["ring"][0]/parameters["ring"][1]
        wall_ratio = parameters["ring"][1]/parameters["radius"]
        b_mode = (mode*(1-annulus_ratio**2)+(1-annulus_ratio**(2*mode))*wall_ratio**(2*mode))/2
        c_mode = (mode*(1-annulus_ratio**2)*(1-(annulus_ratio*wall_ratio)**(2*mode))
                  -(1-annulus_ratio**(2*mode))*(1-wall_ratio**(2*mode)))
        discriminant = c_mode-b_mode**2
        nominal_density = parameters["mean_ring"]+parameters["perturbation"]
        theory = (parameters["alpha"]*nominal_density/(2*abs(parameters["omega"]))*math.sqrt(discriminant)
                  if discriminant > 0 else None)
        fourier["reference_theory"] = {"nominal_vacuum_annulus_growth_rate": theory,
            "mean_density_vacuum_annulus_growth_rate": theory*parameters["mean_ring"]/nominal_density if theory is not None else None,
            "printed_paper_growth_rate": PAPER_RATES[mode],
            "scope": "linear drift-limit vacuum-annulus theory; nominal density, finite background/perturbation not represented"}
    if samples:
        csv_path = output/(run_summary["label"]+"-fourier.csv")
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("potential_time", "density_time", "macro_step",
                                    "real", "imaginary", "modulus", "normalized_modulus"), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(samples)
        run_summary["outputs"].append(str(csv_path))


# 9. Compare only observed, normalized growth curves; mark the fixed fit windows.
growth_runs = [row for row in summary["runs"] if row.get("fourier", {}).get("normalization_modulus") is not None]
growth_modes = sorted({int(row["parameters"]["mode"]) for row in growth_runs})
if growth_modes:
    figure, axes = plt.subplots(1, len(growth_modes), figsize=(5*len(growth_modes), 4.7), squeeze=False)
    for mode_index, mode in enumerate(growth_modes):
        axis = axes[0, mode_index]
        last_observed = 0.
        for run_index, run_summary in enumerate(growth_runs):
            parameters, fourier = run_summary["parameters"], run_summary["fourier"]
            if int(parameters["mode"]) != mode:
                continue
            samples = fourier["samples"]
            times = np.array([sample["potential_time"] for sample in samples])
            values = np.array([sample["normalized_modulus"] for sample in samples])
            visible = times <= args.growth_end
            plotted_values = np.where(values > 0, values, np.nan)  # A zero becomes a gap on a logarithmic axis.
            label = "%s: %s %d×%d, L≤%d" % (run_summary["label"], parameters["model"], parameters["nr"], parameters["ntheta"], parameters["max_levels"])
            if fourier["fit"]["status"] == "fitted":
                label += "; fit γ=%.5g" % fourier["fit"]["growth_rate"]
            else:
                label += "; fit unavailable"
            line, = axis.semilogy(times[visible], plotted_values[visible], ".-", markersize=3,
                                  color=SERIES_COLORS[run_index % len(SERIES_COLORS)], label=label)
            if np.any(visible):
                last_observed = max(last_observed, float(times[visible].max()))
            theory = fourier["reference_theory"]["nominal_vacuum_annulus_growth_rate"]
            if theory is not None and np.any(visible):
                axis.semilogy(times[visible], np.exp(theory*(times[visible]-fourier["normalization_time"])),
                              "--", color=line.get_color(), alpha=.6, label="nominal theory γ=%.5g" % theory)
            if fourier["fit"]["status"] == "fitted":
                fit = fourier["fit"]
                fitted = visible & (times >= fit["window"][0]) & (times <= fit["window"][1])
                axis.semilogy(times[fitted], np.exp(fit["intercept"]+fit["growth_rate"]*times[fitted]),
                              color=line.get_color(), linewidth=2.5)
        axis.set_title("Mode %d; fit window [%g, %g]" % (mode, *FIT_WINDOWS[mode]))
        axis.set_xlabel("Actual potential midpoint time")
        axis.set_ylabel(r"$|\widehat{\psi}_{\ell}(r=6,t)|/|\widehat{\psi}_{\ell}(r=6,t_0)|$")
        if last_observed > 0:
            axis.set_xlim(0, min(args.growth_end, last_observed))
            if FIT_WINDOWS[mode][0] <= min(args.growth_end, last_observed):
                axis.axvspan(*FIT_WINDOWS[mode], color="0.8", alpha=.3, label="fixed paper fit window")
        ylower, yupper = axis.get_ylim()
        if 0 < ylower < yupper < 1.25*ylower:
            # Small actual changes need distinct numeric labels even on the log axis.
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
            axis.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            axis.yaxis.set_minor_formatter(NullFormatter())
        axis.grid(True, which="both", alpha=.2)
        axis.legend(fontsize=7)
    figure.suptitle("Observed Fourier amplitudes\nFixed-window fits; nominal drift-limit theory"
                   + ("\n" + args.context_label if args.context_label else ""), fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, .94 if args.context_label else .97))
    for extension in ("png", "pdf"):
        figure_path = output/("fourier-growth."+extension)
        figure.savefig(figure_path, dpi=args.dpi)
        summary["growth_outputs"].append(str(figure_path))
    plt.close(figure)
else:
    summary["warnings"].append("No normalized growth curve: a qualified near-zero potential sample is required.")

# Figure 5.4(d)'s comparison has rows only for successfully fitted numerical curves.
rate_rows = []
for run_summary in growth_runs:
    parameters, fourier = run_summary["parameters"], run_summary["fourier"]
    fit = fourier["fit"]
    if fit["status"] != "fitted":
        continue
    nominal_rate = fourier["reference_theory"]["nominal_vacuum_annulus_growth_rate"]
    mean_rate = fourier["reference_theory"]["mean_density_vacuum_annulus_growth_rate"]
    rate_rows.append(dict(run=run_summary["label"], model=parameters["model"],
        mode=int(parameters["mode"]), nr=parameters["nr"], ntheta=parameters["ntheta"],
        max_levels=parameters["max_levels"], max_dt=parameters["max_dt"],
        fit_lower=fit["window"][0], fit_upper=fit["window"][1],
        samples=fit["samples_in_window"], growth_rate=fit["growth_rate"],
        slope_standard_error=fit["slope_standard_error"], log_rmse=fit["log_rmse"],
        nominal_theory=nominal_rate, mean_density_theory=mean_rate,
        signed_percent_difference_from_nominal=(100*(fit["growth_rate"]/nominal_rate-1)
                                                if nominal_rate is not None else None)))
summary["growth_rate_rows"] = rate_rows
summary["growth_rate_outputs"] = []
if rate_rows:
    rate_path = output/"fourier-growth-rates.csv"
    with rate_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rate_rows[0]))
        writer.writeheader()
        writer.writerows(rate_rows)
    summary["growth_rate_outputs"].append(str(rate_path))
    columns = ["Run / base grid / levels", "Mode", "Fit interval", "Samples", r"Measured $\gamma$",
               "Fit slope SE", "Log RMSE"]
    include_nominal = all(row["nominal_theory"] is not None for row in rate_rows)
    if include_nominal:
        columns.extend([r"Nominal $\gamma$", "Difference (%)"])
    table_rows = []
    for row in rate_rows:
        cells = ["%s: %s %d×%d / L≤%d" %
                 (row["run"].split("-", 1)[0], row["model"], row["nr"], row["ntheta"], row["max_levels"]),
                 str(row["mode"]), "%g–%g" % (row["fit_lower"], row["fit_upper"]),
                 str(row["samples"]), "%.6g" % row["growth_rate"],
                 "%.2g" % row["slope_standard_error"], "%.2g" % row["log_rmse"]]
        if include_nominal:
            cells.extend(["%.6g" % row["nominal_theory"],
                          "%+.3g" % row["signed_percent_difference_from_nominal"]])
        table_rows.append(cells)
    figure, axis = plt.subplots(figsize=(14 if include_nominal else 11, 2.1+.38*len(rate_rows)))
    axis.axis("off")
    table = axis.table(cellText=table_rows, colLabels=columns, cellLoc="center", loc="center",
                       colWidths=([.26]+[.74/(len(columns)-1)]*(len(columns)-1)))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)
    for column in range(len(columns)):
        table[0, column].set_facecolor("#eaf1f8")
    figure.suptitle("Fitted Fourier growth rates at r=6; fixed paper windows"
                   + ("\n"+args.context_label if args.context_label else ""), fontsize=11)
    figure.text(.5, .03, "Fit slope SE is a regression statistic, not a discretization-error estimate.\n"
                "Nominal vacuum-annulus theory and mean-density theory are recorded separately in the CSV.",
                ha="center", fontsize=8)
    figure.tight_layout(rect=(.01, .13, .99, .82))
    for extension in ("png", "pdf"):
        rate_path = output/("fourier-growth-rates."+extension)
        figure.savefig(rate_path, dpi=args.dpi)
        summary["growth_rate_outputs"].append(str(rate_path))
    plt.close(figure)


# 10. Preserve every source, transform, fit, missing time and actual animation timestamp.
for run_summary in summary["runs"]:
    if run_summary["missing_panel_times"]:
        print("%s: missing requested panel times %s" % (run_summary["label"], run_summary["missing_panel_times"]))
    if "fourier" in run_summary:
        print("%s: %s; %s" % (run_summary["label"], run_summary["fourier"].get("normalization_status", "no potential samples"),
                               run_summary["fourier"]["fit"]["status"]))
manifest = output/"render-manifest.json"
manifest.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
print(manifest)
if not trajectories:
    raise SystemExit("No actual tutorial snapshots found; numerical rendering remains pending.")
