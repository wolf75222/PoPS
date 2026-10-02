"""NativeABI8/CP12 spatial reception @3; historical @1/@2 remain immutable.

The shared independent homogeneous reader supplies wire/identity codecs only.
Synthetic math tests are SOURCE_ONLY; only externally sealed real archives may
enter receive(). Neither owner pins nor ROOT approvals are generated here.
"""

import argparse
from copy import deepcopy
import ast
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "independent_amr_wire", Path(__file__).with_name("sol61_evolved_stage_amr_saved_reception.py")
)
b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b)
need, same, typed, exact = b.need, b.same, b.typed, b.exact

_current_spec = importlib.util.spec_from_file_location("spatial_current_wire", Path(__file__).with_name("sol61_evolved_stage_amr_saved_reception_v3.py"))
_historical_c = importlib.util.module_from_spec(_current_spec)
_current_spec.loader.exec_module(_historical_c)
# Private callback adaptation only: historical module objects remain immutable.
import types
import inspect
_accept_spec = importlib.util.spec_from_file_location("spatial_mixed_norm", Path(__file__).parents[1] / "python/support/original_field_acceptance.py")
_accept = importlib.util.module_from_spec(_accept_spec)
_accept_spec.loader.exec_module(_accept)
STOP_RULE = _accept.STOP_RULE
EXPECTED_NATIVE_ABI = 8

def diagnostic_triplet(row):
    values = [(name.decode("utf-8"), struct.unpack("<d", bits)[0]) for name, bits in row.items()]
    return _accept.selected_native_mixed_rule(values, 1e-10)

def zero_seed_selection(ir):
    solves = [node for node in ir["nodes"] if node["op"] == "solve_spatial_field"]
    need(len(solves) == 1, "zero-seed solve inventory differs")
    attrs = solves[0]["attrs"]
    request = attrs["solve_request"]
    expected = "pops.solve-initialization.v1:sha256:" + b.digest(c.wire.cbor(
        dict(protocol="pops.identity", domain="solve-initialization", schema_version=1, payload=None)))
    need("seed_index" in attrs and attrs["seed_index"] is None and request["seed"] is None
         and request["initialization_identity"] == expected, "authenticated zero-seed initialization differs")
    return expected

def norm_arithmetic_ir(ir, ranks):
    need(type(ranks) is int and ranks in (1,2), "arithmetic proof rank terms absent")
    solves = [node for node in ir["nodes"] if node["op"] == "solve_spatial_field"]
    need(len(solves) == 1, "arithmetic physical solve inventory differs")
    attrs = solves[0]["attrs"]
    need(type(attrs["ncomp"]) is int and attrs["ncomp"] == 3, "arithmetic residual width differs")
    def count(node):
        need(type(node) is list and node, "arithmetic expression schema differs")
        op = node[0]
        if op in ("literal","unknown","input"):
            return 0
        if op == "temporal_tau":
            return 1
        if op == "neg" and len(node)==2:
            return count(node[1])
        if op in ("add","sub","mul","div") and len(node)==3:
            return 1+count(node[1])+count(node[2])
        # In particular, no source-only libm pow/sqrt accuracy assumption.
        raise ValueError("arithmetic expression lacks proved primitive error bound")
    expressions = attrs["local_expressions"]
    need(len(expressions)==3, "arithmetic physical equation count differs")
    physical = sum(count(expression) for expression in expressions)
    dim, components = 2, 3
    # Source QuadraticInterpolationTransfer: 7 ops/axis to form weights;
    # 3**dim points each dim weight products, one sample product and one sum.
    interpolation = 7*dim+(3**dim)*(dim+2)
    faces = 2*dim
    # Two sides/all components interpolated; arithmetic coefficient means;
    # differences and metric division; all rows of the signed matrix product.
    face = 2*components*interpolation+2*components**2+2*components+components*(2*components-1)
    divergence = faces*(face+2*components)
    return dict(physical=physical, stencil=divergence, ranks=ranks)


def independent_original_norm(metrics, report, arithmetic=None):
    need(type(arithmetic) is dict and type(arithmetic.get("ranks")) is int and arithmetic["ranks"] in (1,2),
         "arithmetic proof rank terms absent")
    reference = metrics["original_F_zero_seed_reference_l2"]
    need(type(reference) is float and math.isfinite(reference) and 0 < reference < 1.,
         "independent zero-seed reference outside profile")
    # Absolute-input scale bounds cancellation in q+dt*load. Positive squared
    # terms, at most eight binary64 operations per term plus
    # reduction/sqrt overhead. Each of the two evaluations obeys gamma_k.
    count = metrics["original_F_reference_terms"]
    k = 8*count+8
    u = 2.**-53
    need(type(count) is int and count > 0 and k*u < 1., "reference roundoff proof domain differs")
    gamma = k*u/(1.-k*u)
    bound = 2.*gamma/(1.-gamma)*max(metrics["original_F_reference_absolute_scale_l2"], report["reference_residual_norm"])
    need(abs(reference-report["reference_residual_norm"]) <= bound,
         "native zero-seed reference differs from independent F")
    value = metrics["original_F_weighted_l2"]
    need(type(value) is float and math.isfinite(value) and value >= 0
         and value <= 1e-10,
         "independent original-F selected mixed norm fails")
    # Authenticated physical IR, source 2D stencil loop extents, global active
    # scalar terms, and explicit rank reduction count; no empirical op budget.
    final_terms = 3*count//2
    final_k = arithmetic["physical"]+arithmetic["stencil"]+3*final_terms+2+(arithmetic["ranks"]-1)
    final_gamma = final_k*u/(1.-final_k*u)
    final_bound = 2.*final_gamma/(1.-final_gamma)*metrics["original_F_absolute_scale_l2"]
    need(abs(value-report["residual_norm"]) <= final_bound,
         "native original-F norm differs from independent physical residual")


_tree = ast.parse(inspect.getsource(_historical_c.checkpoint_current))
_loops = [node for node in ast.walk(_tree) if isinstance(node, ast.For) and isinstance(node.iter, ast.Name) and node.iter.id == "parsed"]
need(len(_loops) == 1, "historical diagnostic guard adaptation differs")
_loops[0].body = ast.parse("diagnostic_triplet(row)").body
_globals = dict(_historical_c.checkpoint_current.__globals__, diagnostic_triplet=diagnostic_triplet)
exec(compile(ast.fix_missing_locations(_tree), __file__, "exec"), _globals)

def checkpoint(arrays, phase, images, n, width, ranks, identities, abi, *, transfer_subjects, history_registry):
    _historical_c.empty_component_registry(arrays)
    return _globals["checkpoint_current"](arrays, phase, images, n, width, ranks, identities, abi, transfer_subjects, history_registry)

def native_abi_receipt(receipt, native):
    exact(receipt,("schema","native","header_signature","module_abi_version","capability_abi_version","release_native_abi_version"),"ROOT NativeABI receipt")
    need(receipt["schema"] == "root.api040.native-abi@1" and typed(receipt["native"],native)
         and type(receipt["header_signature"]) is str and bool(receipt["header_signature"]),"NativeABI DSO/header origin differs")
    for name in ("module_abi_version","capability_abi_version","release_native_abi_version"):
        need(type(receipt[name]) is int and receipt[name] == EXPECTED_NATIVE_ABI,"NativeABI8 module/capability/release differs")
    return receipt

def source_native_abi(header, release):
    matches = re.findall(rb"^inline constexpr int kReleaseNativeAbiVersion = ([0-9]+);$", header, re.M)
    need(matches == [str(EXPECTED_NATIVE_ABI).encode()], "source NativeABI header differs")
    tree = ast.parse(release.decode("utf-8"))
    values = [node.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "NATIVE_ABI_VERSION" for target in node.targets)]
    need(len(values) == 1 and isinstance(values[0], ast.Constant)
         and type(values[0].value) is int and values[0].value == EXPECTED_NATIVE_ABI,
         "source NativeABI release constants differ")


c = types.SimpleNamespace(**dict(vars(_historical_c), checkpoint=checkpoint, native_abi_receipt=native_abi_receipt))
D = np.array([[0.012, 0.002], [-0.001, 0.014]])
QUALIFICATION = "nonconstant-original-composite-Q-tag-selection-periodic-strip@3"
BUILD_AST = "1cd73f8bfcdcccb3f676e7ee11f276a5fcc9a47305847825f2719232597e6aca"
EXTRA = {
    "valid",
    "cartesian_cell_volume",
    "declared_no_EB_kappa",
    "x_edges",
    "y_edges",
    "native_base_shape",
    "native_patch_boxes",
    "carrier_patch_boxes",
}


def initial_q(n):
    """Exact cell means via independently expanded Fourier moments of H(T)."""
    left, right = np.arange(n) / n, np.arange(1, n + 1) / n
    k = 2 * math.pi
    c = n * (np.sin(k * right) - np.sin(k * left)) / k
    s = n * (np.cos(k * left) - np.cos(k * right)) / k
    c2 = 0.5 + n * (np.sin(2 * k * right) - np.sin(2 * k * left)) / (4 * k)
    s2 = 1 - c2
    cs = n * (np.cos(2 * k * left) - np.cos(2 * k * right)) / (4 * k)
    avga, avgb = 0.15 + 0.02 * c, 0.25 + 0.015 * s
    aa = 0.15**2 + 2 * 0.15 * 0.02 * c + 0.02**2 * c2
    bb = 0.25**2 + 2 * 0.25 * 0.015 * s + 0.015**2 * s2
    ab = 0.15 * 0.25 + 0.25 * 0.02 * c + 0.15 * 0.015 * s + 0.02 * 0.015 * cs
    return np.stack((avga + aa + 0.1 * bb, avgb + bb + 0.2 * ab))


def geometry(rows, masks, n):
    """Full-y is an explicitly tested subspace, never a production size cap."""
    need(len(rows) == len(masks) == 2, "two-level strip inventory differs")
    for level, (row, active) in enumerate(zip(rows, masks, strict=True)):
        m = n * 2**level
        valid = np.ones((m, m), dtype=bool) if level == 0 else active
        same(row["valid"], valid, "valid geometry")
        same(row["active"], active, "finest ownership")
        need(
            np.array_equal(valid, np.broadcast_to(valid[0], valid.shape))
            and np.array_equal(active, np.broadcast_to(active[0], active.shape)),
            "not a full-y strip",
        )
        for key, value in (
            ("cartesian_cell_volume", np.full((m, m), 1 / m**2)),
            ("declared_no_EB_kappa", np.ones((m, m))),
            ("x_edges", np.arange(m + 1) / m),
            ("y_edges", np.arange(m + 1) / m),
            ("native_base_shape", np.array([n, n], dtype=np.int64)),
        ):
            same(row[key], value, "derived Cartesian metric " + key)
        same(row["native_patch_boxes"], rows[0]["native_patch_boxes"], "level native boxes")
        same(row["carrier_patch_boxes"], rows[0]["carrier_patch_boxes"], "level full carrier geometry")
    covered = ~masks[0][0]
    need(
        covered.any() and not covered.all() and np.array_equal(masks[1][0], np.repeat(covered, 2)),
        "ratio-two strip coverage differs",
    )
    return covered


def flux_action(tower, covered, *, matrix=D, reflux=True, quadratic=True, restrict=True):
    """Finite-volume leaf graph, not the author's all-faces array algorithm.

    Each edge contributes once with opposite signs to its adjacent leaf cells.
    At a fine/coarse edge the absent fine centre is evaluated by the coarse
    three-centre polynomial. Interpolation weights come from a Vandermonde
    system. Signed matrix entries act on the full oriented vector gradient.
    """
    coarse, fine = (np.asarray(x, dtype=float).copy() for x in tower)
    n = coarse.shape[1]
    need(
        coarse.shape == (2, n) and fine.shape == (2, 2 * n) and covered.shape == (n,),
        "flux vector shape differs",
    )
    if restrict:
        for i in np.flatnonzero(covered):
            coarse[:, i] = (fine[:, 2 * i] + fine[:, 2 * i + 1]) / 2
    leaves = [
        (level, i)
        for i in range(n)
        for level, i in ([(1, 2 * i), (1, 2 * i + 1)] if covered[i] else [(0, i)])
    ]
    need(len(leaves) > n and len(leaves) < 2 * n, "partial leaf graph required")
    images = [np.zeros_like(coarse), np.zeros_like(fine)]
    edge_fluxes = []

    def missing_fine(j):
        j %= 2 * n
        p = j // 2
        s = (j + 0.5) / 2 - (p + 0.5)
        if not quadratic:
            return coarse[:, p]
        nodes = np.array([-1.0, 0.0, 1.0])
        weights = np.linalg.solve(
            np.vstack((np.ones(3), nodes, nodes**2)), np.array([1.0, s, s * s])
        )
        return coarse[:, [(p - 1) % n, p, (p + 1) % n]] @ weights

    for edge, right in enumerate(leaves):
        left = leaves[edge - 1]
        if right[0] == left[0]:
            gradient = (tower[right[0]][:, right[1]] - tower[left[0]][:, left[1]]) * (
                n * 2 ** right[0]
            )
        else:
            # Edge boundary is the lower edge of the right leaf, modulo period.
            face = (right[1] * (2 if right[0] == 0 else 1)) % (2 * n)
            lo, hi = (face - 1) % (2 * n), face
            lv = fine[:, lo] if covered[lo // 2] else missing_fine(lo)
            rv = fine[:, hi] if covered[hi // 2] else missing_fine(hi)
            gradient = (rv - lv) * (2 * n)
        flux = matrix @ gradient
        edge_fluxes.append(flux)
        images[left[0]][:, left[1]] += flux * (n * 2 ** left[0])
        images[right[0]][:, right[1]] -= flux * (n * 2 ** right[0])
        if not reflux and left[0] != right[0]:
            # Deliberately wrong counter-model: coarse side retains coarse flux.
            p = right[1] // 2 if right[0] else right[1]
            coarse_flux = matrix @ (coarse[:, p] - coarse[:, (p - 1) % n]) * n
            target = left if left[0] == 0 else right
            sign = 1 if left[0] == 0 else -1
            images[0][:, target[1]] += sign * (coarse_flux - flux) * n
    return images, coarse, np.stack(edge_fluxes, axis=1)


def science(images, masks, n):
    need(set(images) == set(b.PHASES), "saved phase inventory differs")
    covered = geometry(images["initial"], masks, n)
    for phase in b.PHASES:
        geometry(images[phase], masks, n)
        for level, row in enumerate(images[phase]):
            m = n * 2**level
            keys = EXTRA | {"Q0", "Q1", "forcing", "active"}
            if phase != "initial":
                keys |= {
                    name + suffix for name in ("T0", "T1", "z") for suffix in ("", "-previous")
                }
                keys |= {"history_sample_identity_" + name for name in ("T0", "T1", "z")}
            exact(row, keys, "closed spatial NPZ")
            for key in ("Q0", "Q1", "forcing"):
                b.finite_array(row[key], ((3 * m * m,) if key == "forcing" else (m * m,)), key)
                if phase != "initial":
                    same(row["forcing"], images["initial"][level]["forcing"], "readonly load")
            if phase == "initial":
                q = np.stack([row[k].reshape(m, m) for k in ("Q0", "Q1")])
                expected = np.broadcast_to(initial_q(m)[:, None, :], q.shape)
                need(
                    np.max(np.abs((q - expected)[:, row["valid"]])) <= b.TOL,
                    "original CellBounds initial Q differs",
                )
            load = row["forcing"].reshape(3, m, m)
            need(
                np.max(np.abs(load[:2, :, row["valid"][0]] - b.declared_load(2)[3][:, None, None]))
                <= b.TOL,
                "original load differs",
            )
            marker = 1 + 0.04 * m * (
                np.sin(2 * np.pi * (np.arange(1, m + 1) / m - 0.5))
                - np.sin(2 * np.pi * (np.arange(m) / m - 0.5))
            ) / (2 * np.pi)
            need(
                np.max(np.abs((load[2] - marker[None, :])[row["valid"]])) <= b.TOL,
                "original refinement marker differs",
            )
    metrics = {}
    for phase in ("accepted", "continuous", "reloaded", "replay"):
        rows = images[phase]
        previous = images["initial" if phase in ("accepted", "reloaded") else "accepted"]
        temperatures = []
        for level, row in enumerate(rows):
            m = n * 2**level
            t = np.stack([row[k].reshape(m, m) for k in ("T0", "T1")])
            need(
                np.all(np.isfinite(t[:, row["valid"]]))
                and np.max(np.abs((t - t[:, :1, :])[:, row["valid"]])) <= b.TOL,
                "temperature leaves the full-y subspace",
            )
            need(np.min(t[:, row["valid"]]) > 0, "positive temperature branch differs")
            temperatures.append(t[:, 0, :])
            need(
                np.max(np.abs((row["z"].reshape(m, m) - 0.25 * t[0] - 0.5 * t[1])[masks[level]]))
                <= b.TOL,
                "original z constraint differs",
            )
        diff, restricted, faces = flux_action(temperatures, covered)
        need(
            np.max(np.abs(temperatures[0][:, covered] - restricted[:, covered])) <= b.TOL,
            "coarse T is not restriction of fine T",
        )
        fine_qmean = b.q_of(temperatures[1]).reshape(2, n, 2).mean(axis=2)
        coarse_q = np.stack([rows[0][k].reshape(n, n)[0] for k in ("Q0", "Q1")])
        need(
            np.max(np.abs(coarse_q[:, covered] - fine_qmean[:, covered])) <= b.TOL,
            "covered Q must restrict H(fine T), not H(restricted T)",
        )
        gap = np.max(np.abs((fine_qmean - b.q_of(restricted))[:, covered]))
        need(gap > 1e-10, "nonlinear restriction discriminator absent")
        residual, activity, original_squared, reference_squared = 0.0, 0.0, 0.0, 0.0
        reference_terms = 0
        reference_scale_squared = 0.
        final_scale_squared = 0.
        # Quadratic cell-centered offsets +/-1/4 have absolute weight sum19/16
        # per axis. The 2D tensor ghost interpolation is bounded by (19/16)^2.
        maxima = np.array([max(float(np.max(np.abs(images[phase][level]["T"+str(i)][images[phase][level]["valid"].ravel()])))
                              for level in range(2)) for i in range(2)])
        spatial_dim = int(images[phase][0]["native_base_shape"].size)
        divergence_scale = 4.*spatial_dim*(2*n)**2*(19./16.)**2*(np.abs(D)@maxima)
        amounts = np.zeros(2)
        initial_amounts = np.zeros(2)
        diffusion_amounts = np.zeros(2)
        for level, row in enumerate(rows):
            m = n * 2**level
            active = masks[level]
            q = np.stack([row[k].reshape(m, m) for k in ("Q0", "Q1")])
            old = np.stack([previous[level][k].reshape(m, m) for k in ("Q0", "Q1")])
            load = row["forcing"].reshape(3, m, m)[:2]
            action = np.broadcast_to(diff[level][:, None, :], q.shape)
            t_full = np.broadcast_to(temperatures[level][:, None, :], q.shape)
            original = b.q_of(t_full) - old - b.DT*(action+load)
            z_error = row["z"].reshape(m,m) - .25*t_full[0] - .5*t_full[1]
            original_squared += float((np.sum(original[:,active]**2)+np.sum(z_error[active]**2))/m**2)
            reference_squared += float(np.sum((old+b.DT*load)[:,active]**2)/m**2)
            reference_terms += 2*int(np.count_nonzero(active))
            reference_scale_squared += float(np.sum((np.abs(old)+b.DT*np.abs(load))[:,active]**2)/m**2)
            absolute_t = np.abs(t_full)
            h_scale = b.q_of(absolute_t)
            f_scale = h_scale+np.abs(old)+b.DT*(divergence_scale[:,None,None]+np.abs(load))
            z_scale = np.abs(row["z"].reshape(m,m))+.25*absolute_t[0]+.5*absolute_t[1]
            final_scale_squared += float((np.sum(f_scale[:,active]**2)+np.sum(z_scale[active]**2))/m**2)
            residual = max(
                residual, float(np.max(np.abs((q - old - b.DT * (action + load))[:, active])))
            )
            need(
                np.max(
                    np.abs(
                        (q - b.q_of(np.broadcast_to(temperatures[level][:, None, :], q.shape)))[
                            :, active
                        ]
                    )
                )
                <= b.TOL,
                "active original H(T) differs",
            )
            activity = max(activity, float(np.max(np.abs(action[:, active]))))
            amounts += np.sum(q[:, active], axis=1) / m**2
            qi = np.stack([images["initial"][level][k].reshape(m, m) for k in ("Q0", "Q1")])
            initial_amounts += np.sum(qi[:, active], axis=1) / m**2
            diffusion_amounts += np.sum(action[:, active], axis=1) / m**2
        need(residual <= b.TOL, "original spatial F fails")
        need(
            activity > 1e-6 and np.max(np.abs(faces)) > 1e-6,
            "nonconstant flux discriminator absent",
        )
        need(np.max(np.abs(diffusion_amounts)) <= b.TOL, "composite reflux conservation fails")
        need(
            np.max(
                np.abs(amounts - initial_amounts - b.STEPS[phase] * b.DT * b.declared_load(2)[3])
            )
            <= b.TOL
            and abs(sum(amounts - initial_amounts)) <= b.TOL,
            "closed original Q balance fails",
        )
        metrics[phase] = dict(
            original_F_linf=residual,
            flux_activity=activity,
            original_F_weighted_l2=math.sqrt(original_squared),
            original_F_zero_seed_reference_l2=math.sqrt(reference_squared),
            original_F_reference_terms=reference_terms,
            original_F_reference_absolute_scale_l2=math.sqrt(reference_scale_squared),
            original_F_absolute_scale_l2=math.sqrt(final_scale_squared),
            restriction_gap=float(gap),
            Q_amounts=amounts.tolist(),
        )
    for first, second in (("accepted", "reloaded"), ("continuous", "replay")):
        for a, c in zip(images[first], images[second], strict=True):
            exact(c, a, "replay inventory")
            for key in a:
                same(a[key], c[key], "exact saved replay " + key)
    for level, row in enumerate(images["continuous"]):
        for name in ("T0", "T1", "z"):
            same(
                row[name + "-previous"],
                images["accepted"][level][name],
                "previous accepted history",
            )
    return metrics


def declared_source(spatial, amr, equations, controls):
    c.declared_source(amr, equations, controls)
    parsed = ast.parse(spatial)
    builds = [x for x in parsed.body if isinstance(x, ast.FunctionDef) and x.name == "build"]
    need(len(builds) == 1, "duplicate original builder")
    need(
        all(
            isinstance(x, (ast.Expr, ast.Import, ast.ImportFrom, ast.FunctionDef))
            for x in parsed.body
        ),
        "module-level original builder/global rebinding",
    )
    imports = [
        ast.dump(x, include_attributes=False)
        for x in parsed.body
        if isinstance(x, (ast.Import, ast.ImportFrom))
    ]
    original_imports = ast.parse(
        "import numpy as np\nfrom tests.python.support.evolved_stage_amr import ACCEPTANCE, CONTROLS, DENSE_BYTES, DT, FD_STEP, closed_data\nfrom tests.python.support.evolved_stage_mms import DIFFUSION, accumulation"
    ).body
    need(
        imports == [ast.dump(x, include_attributes=False) for x in original_imports],
        "original physical imports differ",
    )
    build = builds[0]
    need(
        b.digest(ast.dump(build, include_attributes=False).encode()) == BUILD_AST,
        "reviewed original nonconstant build differs",
    )


def nonlinear_restriction_attacks(images, masks, n):
    """Measure actual signed Jensen gaps and demand rejection at the unchanged guard."""
    covered = ~masks[0][0]
    report = {}
    for phase in ("accepted", "continuous"):
        coarse, fine = images[phase]
        tc = np.stack([coarse[name].reshape(n,n)[0] for name in ("T0","T1")])
        tf = np.stack([fine[name].reshape(2*n,2*n)[0] for name in ("T0","T1")])
        delta = (b.q_of(tf).reshape(2,n,2).mean(axis=2) - b.q_of(tc))[:,covered]
        mutant = deepcopy(images)
        wrong = b.q_of(tc)
        for i in range(2):
            mutant[phase][0][f"Q{i}"].reshape(n,n)[:,covered] = wrong[i,covered]
        rejected = False
        try:
            science(mutant, masks, n)
        except ValueError as error:
            rejected = str(error) == "covered Q must restrict H(fine T), not H(restricted T)"
        need(rejected, "H(average Psi) countermodel not rejected by existing covered-Q scientific guard")
        report[phase] = dict(signed_component_min=delta.min(axis=1).tolist(),
                             signed_component_max=delta.max(axis=1).tolist(),
                             maximum_absolute=float(np.max(np.abs(delta))),
                             existing_guard=b.TOL, countermodel_rejected=True)
    return report


def contract():
    value = c.contract()
    value.update(
        schema="sol61.evolved-stage-amr-spatial.owner-pins@3",
        qualification=QUALIFICATION,
        cases="exact one cells8 width2: externally pinned real receipt/files/artifact/bind/semantic",
    )
    value["native_abi_version"] = EXPECTED_NATIVE_ABI
    for key in ("spatial", "acceptance", "native_abi_header", "release_constants"):
        value["source_files"][key] = "path+sha256"
    value["stop_rule"] = STOP_RULE
    value["independent_original_l2_threshold"] = 1e-10
    value["seed_selection"] = "authenticated solve-initialization identity for absent seed"
    value["approval"] = dict(
        schema="sol61.evolved-stage-amr-spatial.root-approval@3",
        approved_by="ROOT",
        qualification=QUALIFICATION,
        pins_sha256="external SHA256",
    )
    return value


def receive(pins_path, pins_sha, approval_path, approval_sha):
    raw, approved = Path(pins_path).read_bytes(), Path(approval_path).read_bytes()
    need(
        b.digest(raw) == pins_sha and b.digest(approved) == approval_sha,
        "external ROOT seals differ",
    )
    pins, approval = b.strict_json(raw), b.strict_json(approved)
    need(
        typed(
            approval,
            dict(
                schema="sol61.evolved-stage-amr-spatial.root-approval@3",
                approved_by="ROOT",
                qualification=QUALIFICATION,
                pins_sha256=pins_sha,
            ),
        ),
        "spatial ROOT approval differs",
    )
    exact(
        pins,
        (
            "schema",
            "qualification",
            "stop_rule",
            "independent_original_l2_threshold",
            "seed_selection",
            "native_evidence",
            "mode",
            "ranks",
            "source_commit",
            "native_build_source_commit",
            "abi_key",
            "native_abi_version",
            "native_abi_receipt",
            "ir_version",
            "roots",
            "native",
            "sdk",
            "package_manifest",
            "source_files",
            "junit",
            "batch_names",
            "cases",
        ),
        "spatial owner pins",
    )
    need(
        pins["schema"] == "sol61.evolved-stage-amr-spatial.owner-pins@3"
        and pins["qualification"] == QUALIFICATION
        and pins["stop_rule"] == STOP_RULE
        and typed(pins["independent_original_l2_threshold"], 1e-10)
        and pins["native_evidence"] is True,
        "spatial native scope differs",
    )
    need(
        pins["mode"] in ("serial", "mpi2")
        and type(pins["ranks"]) is int
        and pins["ranks"] == (1 if pins["mode"] == "serial" else 2),
        "rank qualification differs",
    )
    for key in ("source_commit", "native_build_source_commit"):
        need(re.fullmatch("[0-9a-f]{40}", pins[key]) is not None, "commit pin differs")
    roots = [Path(p).resolve() for p in pins["roots"]]

    def read(row):
        exact(row, ("path", "sha256"), "file pin")
        path = Path(row["path"]).resolve()
        need(
            any(path.is_relative_to(root) for root in roots)
            and path.is_file()
            and path.stat().st_size <= 1024**3,
            "offline file outside inventory/budget",
        )
        data = path.read_bytes()
        need(b.digest(data) == row["sha256"], "file digest differs: " + str(path))
        return data

    for key in ("native", "sdk", "package_manifest"):
        read(pins[key])
    need(type(pins["native_abi_version"]) is int and pins["native_abi_version"] == EXPECTED_NATIVE_ABI, "spatial profile requires NativeABI8")
    c.native_abi_receipt(b.strict_json(read(pins["native_abi_receipt"])), pins["native"])
    exact(
        pins["source_files"], ("spatial", "amr", "equations", "controls", "fixture", "acceptance", "native_abi_header", "release_constants"), "source files"
    )
    sources = {key: read(row) for key, row in pins["source_files"].items()}
    need(sources["acceptance"] == Path(_accept.__file__).read_bytes(), "selected native norm source differs")
    source_native_abi(sources["native_abi_header"], sources["release_constants"])
    declared_source(sources["spatial"], sources["amr"], sources["equations"], sources["controls"])
    need(len(pins["cases"]) == 1, "one original spatial case required")
    case = pins["cases"][0]
    exact(case, ("cells", "width", "receipt", "files", "artifact", "bind", "semantic"), "case pin")
    need(
        type(case["cells"]) is int
        and case["cells"] == 8
        and type(case["width"]) is int
        and case["width"] == 2,
        "selected native witness differs",
    )
    receipt = b.strict_json(read(case["receipt"]))
    need(
        receipt["fixture_schema"] == "pops.evolved-stage-amr-spatial-native-fixture@3"
        and receipt["stop_rule"] == STOP_RULE
        and typed(receipt["independent_original_l2_threshold"], 1e-10)
        and receipt["seed_selection"] == pins["seed_selection"]
        and typed(
            [receipt[k] for k in ("cells", "width", "dimension", "rank", "size")],
            [8, 2, 2, 0, pins["ranks"]],
        )
        and receipt["artifact"] == case["artifact"],
        "actual spatial receipt differs",
    )
    need(
        typed(
            receipt["history_protocol"],
            dict(
                wire="POPSHID1",
                raw_slots_after_publication=True,
                latest_slot=1,
                previous_slot=0,
                depth=2,
            ),
        )
        and typed(receipt["newton"], b.CONTROLS),
        "history/controls differ",
    )
    need(
        typed(
            [receipt[k] for k in ("dt", "fd_step", "acceptance", "max_dense_bytes")],
            [0.01, 1e-6, 3e-8, 256 * 1024**2],
        )
        and receipt["realization"] == "FullResidualBasisLU@1",
        "resource/method differs",
    )
    need(
        typed(
            receipt["initial_temperature"],
            dict(T0=[0.15, 0.02, "cos(2*pi*x)"], T1=[0.25, 0.015, "sin(2*pi*x)"]),
        ),
        "declared initial physics differs",
    )
    seen = set()

    def file(row):
        need(
            case["files"].get(row["path"]) == row["sha256"],
            "receipt file outside closed owner inventory",
        )
        seen.add(row["path"])
        return read(row)

    registry = b.strict_json(file(receipt["carrier_registry"]))
    exact(registry, ("schema", "dimension", "size", "phases"), "carrier registry")
    need(registry["schema"] == "sol61.amr.carrier-registry@1" and type(registry["dimension"]) is int
         and registry["dimension"] == 2 and type(registry["size"]) is int and registry["size"] == pins["ranks"]
         and set(registry["phases"]) == set(b.PHASES), "carrier registry authority differs")
    for phase in b.PHASES:
        exact(registry["phases"][phase], ("rows_by_rank",), "carrier phase")
        rows = registry["phases"][phase]["rows_by_rank"]
        need(type(rows) is list and len(rows) == pins["ranks"]
             and all(type(rank) is list and all(type(row) is list and all(type(v) is str for v in row)
                                             for row in rank) for rank in rows), "carrier rank types differ")
        need(typed(rows[0], receipt["phases"][phase]["metadata"][1]), "carrier rank-zero registry differs")
    for a, d in (("accepted", "reloaded"), ("continuous", "replay")):
        need(typed(registry["phases"][a], registry["phases"][d]), "full carrier restart hashes differ")
    file(receipt["native"])
    need(receipt["native"]["sha256"] == pins["native"]["sha256"], "native DSO differs")
    images = {
        phase: [b.wire.archive(file(row)) for row in receipt["phases"][phase]["levels"]]
        for phase in b.PHASES
    }
    arrays = {
        p: b.wire.archive(file(receipt["checkpoints"][p]))
        for p in ("accepted", "continuous", "replay")
    }
    need(len({receipt["checkpoints"][p]["path"] for p in arrays}) == 3, "checkpoint overwrite")
    # Saved author references are authenticated, never used as scientific input.
    for phase in b.PHASES:
        for row in receipt["phases"][phase]["independent_references"]:
            file(row)
    for row in receipt["fixture_sources"]:
        file({key: row[key] for key in ("path", "sha256")})
    hashes = []
    for component in receipt["compilation"]:
        for name in ("DSO", "sidecar"):
            file(component[name])
        if component["component"].startswith("program-"):
            program_ir = b.strict_json(file(component["ir.json"]))
            transfer_subjects = c.program_transfer_subjects(program_ir)
            history_registry = c.program_history_registry(program_ir)
            hashes.append(
                c.program_image(
                    program_ir,
                    file(component["cpp"]).decode(),
                    pins["ir_version"],
                    component["program_hash"],
                    2,
                )
            )
    need(len(hashes) == 1, "one actual Program required")
    need(zero_seed_selection(program_ir) == pins["seed_selection"], "ROOT-pinned zero seed differs")
    arithmetic = norm_arithmetic_ir(program_ir, pins["ranks"])
    masks = None
    manifests, diagnostics = {}, {}
    for phase, cp in arrays.items():
        need(str(cp["program_hash"].item()) == hashes[0], "CP installed Program differs")
        manifests[phase], current, diagnostics[phase] = c.checkpoint(
            cp,
            phase,
            images,
            8,
            2,
            pins["ranks"],
            (case["artifact"], case["bind"], case["semantic"]),
            pins["abi_key"],
            transfer_subjects=transfer_subjects, history_registry=history_registry,
        )
        c.receive_carriers(cp, registry["phases"][phase]["rows_by_rank"], {"Q0":1,"Q1":1,"forcing":3})
        if phase == "accepted":
            c.receive_carriers(cp, registry["phases"]["reloaded"]["rows_by_rank"], {"Q0":1,"Q1":1,"forcing":3})
        if masks is not None:
            for previous_mask, current_mask in zip(masks, current, strict=True):
                same(previous_mask, current_mask, "stationary topology")
        masks = current
    need(
        receipt["active_scalar_DOFs"] == 3 * sum(int(mask.sum()) for mask in masks),
        "active quotient inventory differs",
    )
    expected_metric = dict(
        kind="derived-from-declared-Cartesian-unit-square-and-native-shape",
        lower=[0.0, 0.0],
        upper=[1.0, 1.0],
        ratio=2,
        periodic_axes=["x", "y"],
        kappa="declared no embedded boundary, exactly one; not a native getter",
        native_shape=[8, 8],
        native_patch_boxes=arrays["accepted"]["patch_boxes"].tolist(),
    )
    need(typed(receipt["metric_authority"], expected_metric), "derived metric provenance differs")
    for phase in b.PHASES:
        meta = receipt["phases"][phase]["metadata"]
        need(
            type(meta) is list
            and len(meta) == 4
            and typed(meta[3][:2], [b.STEPS[phase] * b.DT, b.STEPS[phase]]),
            "observed lifecycle differs",
        )
        if phase == "initial":
            need(meta[0] == [], "initial published history invented")
            continue
        p = "accepted" if phase == "reloaded" else phase
        need(
            meta[3][2]
            == [
                [int(x[0]), [int(x[1]), int(x[2])], [int(x[3]), int(x[4])]]
                for x in arrays[p]["patch_boxes"]
            ],
            "observed native boxes differ",
        )
        histories = [
            [
                l,
                name,
                2,
                True,
                b.STEPS[phase],
                [b.DT.hex(), b.DT.hex()],
                images[phase][l]["history_sample_identity_" + name].tobytes().hex(),
            ]
            for l in (0, 1)
            for name in ("T0", "T1", "z")
        ]
        need(typed(meta[0], histories), "observed history point differs")
        observed = {name.encode(): struct.pack("<d", value) for name, value in meta[2]}
        need(
            len(observed) == len(meta[2]) and observed == diagnostics[p][0],
            "rank-zero diagnostic bits differ",
        )
    need(set(case["files"]) == seen, "unclosed owner file inventory")
    for phase in b.PHASES:
        for row in images[phase]:
            same(
                row["native_patch_boxes"],
                arrays["accepted"]["patch_boxes"],
                "snapshot/CP box authority",
            )
    full_boxes = c.complete_carrier_geometry(arrays["accepted"])
    for phase in b.PHASES:
        for row in images[phase]:
            same(row["carrier_patch_boxes"], full_boxes, "snapshot/full carrier geometry authority")
    for key in arrays["continuous"]:
        if key not in ("pops_checkpoint_manifest", "pops_restart_identity"):
            same(arrays["continuous"][key], arrays["replay"][key], "durable exact replay " + key)
    eq = receipt["checkpoint_equivalence"]
    need(
        eq["contract"] == "pops.evolved-stage-checkpoint-equivalence@1"
        and eq["exact_payload_and_manifest"] is True,
        "CP equivalence contract differs",
    )
    prov = eq["provenance"]
    for phase in arrays:
        p = prov[phase]
        need(
            p["clock"] == manifests[phase]["clock"]
            and p["restart"]
            == b.wire.identity_token(manifests[phase]["restart_identity"], "restart")
            and p["run"] == b.wire.identity_token(manifests[phase]["run_identity"], "run"),
            "creator authority differs",
        )
        need(
            all(p[k] == case[k] for k in ("artifact", "bind", "semantic")),
            "creator identity differs",
        )
        run = c.run_identity(p["run_manifest"])
        need(
            run["bind_identity"] == case["bind"]
            and run["run_identity"] == p["run"]
            and typed(
                [run["start_time"], run["start_macro_step"]],
                [0.0 if phase == "accepted" else 0.01, 0 if phase == "accepted" else 1],
            )
            and typed(
                [run["controls"]["t_end"], run["controls"]["max_steps"]], [b.STEPS[phase] * b.DT, 1]
            )
            and run["continuation_identity"]
            == (prov["accepted"]["run"] if phase == "replay" else None),
            "actual run request differs",
        )
    need(
        prov["accepted"]["last_restart"] is None
        and prov["continuous"]["last_restart"] is None
        and prov["replay"]["last_restart"] == prov["accepted"]["restart"],
        "restart lineage differs",
    )
    need(
        len(pins["junit"]) == pins["ranks"]
        and {r["rank"] for r in pins["junit"]} == set(range(pins["ranks"])),
        "all-rank XML absent",
    )
    for row in pins["junit"]:
        xml = ET.fromstring(read(row["file"]))
        tests = xml.findall(".//testcase")
        need(
            len(set(pins["batch_names"])) == len(pins["batch_names"])
            and len(tests) == len(pins["batch_names"])
            and sorted(t.attrib["name"] for t in tests) == sorted(pins["batch_names"]),
            "full raw XML inventory differs",
        )
        need(
            not any(
                t.find(tag) is not None for t in tests for tag in ("failure", "error", "skipped")
            ),
            "raw batch not clean",
        )
        for suite in xml.iter("testsuite"):
            need(
                all(int(suite.attrib.get(k, "0")) == 0 for k in ("failures", "errors", "skipped")),
                "raw suite not clean",
            )
        selected = [
            t
            for t in tests
            if t.attrib["name"].startswith(
                "test_public_evolved_stage_amr_nonconstant_Q_restriction_and_flux["
            )
        ]
        need(
            len(selected) == 1 and selected[0].attrib["name"].endswith("[8]"),
            "selected native spatial case differs",
        )
        props = selected[0].findall("./properties/property")
        need(len({p.attrib["name"] for p in props}) == len(props), "duplicate JUnit property")
        props = {p.attrib["name"]: p.attrib["value"] for p in props}
        need(
            all(
                props.get(k) == v
                for k, v in dict(
                    rank=str(row["rank"]),
                    size=str(pins["ranks"]),
                    dimension="2",
                    artifact_identity=case["artifact"],
                    evolved_stage_amr_spatial_receipt=case["receipt"]["path"],
                ).items()
            ),
            "JUnit native identity differs",
        )
    science_result = science(images, masks, 8)
    for phase, diagnostic_rows in diagnostics.items():
        reports = [diagnostic_triplet(row) for row in diagnostic_rows]
        for report in reports:
            independent_original_norm(science_result[phase], report, arithmetic)
        need(all(typed(report, reports[0]) for report in reports), "rank original-F diagnostics differ")
        science_result[phase]["native_original_F_triplets"] = reports
        if phase == "accepted":
            independent_original_norm(science_result["reloaded"], reports[0], arithmetic)
            science_result["reloaded"]["native_original_F_triplets"] = reports
    nonlinear_attacks = nonlinear_restriction_attacks(images, masks, 8)
    return dict(
        status="received",
        qualification=QUALIFICATION,
        stop_rule=STOP_RULE,
        independent_original_l2_threshold=1e-10,
        seed_selection=pins["seed_selection"],
        native_abi_version=EXPECTED_NATIVE_ABI,
        cases_qualified=1,
        mode=pins["mode"],
        science=science_result,
        nonlinear_restriction_attacks=nonlinear_attacks,
        cpp_to_dso_crypto_link=False,
        limits=[
            "periodic full-y strip, constant signed D and declared Cartesian/no EB; not arbitrary 2D AMR/candidate D/M06/M13/GPU",
            "ROOT-attested source/native/package provenance; no reconstructed CPP-to-DSO crypto graph",
            "opaque accepted-program/auxiliary bodies hash/replay-compared, not fully decoded; private leases not persisted",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("contract")
    recv = sub.add_parser("receive")
    for name in ("pins", "pins_sha256", "approval", "approval_sha256"):
        recv.add_argument("--" + name.replace("_", "-"), required=True)
    args = parser.parse_args()
    result = (
        contract()
        if args.command == "contract"
        else receive(args.pins, args.pins_sha256, args.approval, args.approval_sha256)
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
