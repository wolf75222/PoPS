"""Independent source/math localization of AMR original-residual Newton signs.

No PoPS imports, builds, MPI execution or manufactured native receipts. The
constant Neumann witness eliminates all spatial operators while retaining the
actual coupled reaction and cubic equation of the failed publication test.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re


REACTION = ((2., .2, -.1), (.2, 1.8, .15), (-.1, .15, 2.2))
TARGET = (.15, .25, .18)


def body(source, signature):
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 1
    for end in range(opening + 1, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[opening + 1:end]
    raise ValueError("unterminated source function")


def source_contract(root, *, negative_source_copies=None):
    paths = {
        "amr": "include/pops/runtime/program/prepared_amr_field_residual.hpp",
        "workspace": "include/pops/numerics/elliptic/interface/amr_field_newton_krylov.hpp",
        "uniform": "include/pops/runtime/program/prepared_spatial_residual.hpp",
        "fixture": "tests/cpp/unit/elliptic/amr_original_field_residual.inc",
    }
    text = {name: (root / path).read_text() for name, path in paths.items()}
    if negative_source_copies:
        if not set(negative_source_copies) <= set(paths):
            raise ValueError("unknown negative source copy")
        text.update(negative_source_copies)
    amr, workspace = text["amr"], text["workspace"]
    call = re.search(r"newton_->solve\(destinations, (\w+), derivative,", amr)
    if not call:
        raise ValueError("original AMR workspace seam changed")
    callback = body(amr, "auto " + call[1] + " =")
    if "evaluate(q, result, evaluation);" not in callback:
        raise ValueError("unrecognized defect callback")
    sign = -1 if re.search(r"scale\([^;]*Real\(-1\)\)", callback) else 1
    if sign == -1 and not ("for (auto& level : result)" in callback
                           and "scale(level, Real(-1));" in callback
                           and "local_phase_(lane" in callback):
        raise ValueError("defect negation must cover every level under its local vote")
    evaluate = body(amr, "auto evaluate =")
    if re.search(r"scale\([^;]*Real\(-1\)\)", evaluate):
        raise ValueError("physical F evaluation was negated")
    derivative = body(amr, "auto derivative =")
    if "Real(0.5) / h, plus_[level], -Real(0.5) / h, minus_[level]" not in derivative:
        raise ValueError("central original JVP changed")
    if re.search(r"scale\([^;]*Real\(-1\)\)", derivative):
        raise ValueError("physical JVP was negated")
    linear = body(workspace, "LinearResult solve_linear_")
    if "copy_(rhs, linear_residual_);" not in linear or "rotated_rhs_[0] = beta;" not in linear:
        raise ValueError("workspace RHS convention changed")
    if "lincomb_(trial_, Real(1), iterate_, step, correction_);" not in workspace:
        raise ValueError("workspace additive update changed")
    if "scale(result, Real(-1));" not in body(text["uniform"], "auto defect ="):
        raise ValueError("Uniform defect reference changed")
    recheck = body(amr, "if (report.solved_value_available())")
    if "evaluate(candidate_, recheck_, 0);" not in recheck:
        raise ValueError("original physical recheck no longer evaluates F")
    if re.search(r"scale\([^;]*Real\(-1\)\)", recheck):
        raise ValueError("original physical recheck was negated")
    return {
        "callback": call[1], "defect_sign": sign, "jvp_sign": 1, "update_sign": 1,
        "physical_recheck": "F", "source_sha256": {
            name: hashlib.sha256(text[name].encode()).hexdigest() for name in paths},
        "defect_body": callback.strip(),
    }


def solve3(matrix, rhs):
    rows = [list(row) + [value] for row, value in zip(matrix, rhs, strict=True)]
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(rows[row][column]))
        rows[column], rows[pivot] = rows[pivot], rows[column]
        diagonal = rows[column][column]
        if diagonal == 0:
            raise ValueError("singular original reaction Jacobian")
        rows[column] = [value / diagonal for value in rows[column]]
        for row in range(3):
            if row != column:
                factor = rows[row][column]
                rows[row] = [a - factor * b for a, b in zip(rows[row], rows[column], strict=True)]
    return [row[3] for row in rows]


def local(q, reaction, forcing):
    return [math.fsum([.2 * q[i] ** 3, -forcing[i], *(reaction[i][j] * q[j] for j in range(3))])
            for i in range(3)]


def norm(vector):
    return math.sqrt(math.fsum(value * value for value in vector))


def run_original_constant(sign, permutation=(0, 1, 2)):
    reaction = [[REACTION[i][j] for j in permutation] for i in permutation]
    target = [TARGET[i] for i in permutation]
    forcing = local(target, reaction, [0., 0., 0.])
    q = [0., 0., 0.]
    initial = norm(local(q, reaction, forcing))
    report = dict(initial=initial, defect_sign=sign, trials=[], method="independent exact-Jacobian Newton")
    for iteration in range(12):
        physical = local(q, reaction, forcing)
        current = norm(physical)
        if current <= 2.e-9 * max(1., initial):
            report.update(status="converged", iterations=iteration, physical_residual=current,
                          solution=q, target=target)
            return report
        matrix = [[reaction[i][j] + (.6 * q[i]**2 if i == j else 0.) for j in range(3)] for i in range(3)]
        delta = solve3(matrix, [sign * value for value in physical])
        step, accepted = 1., False
        while step >= 1.e-6:
            trial = [a + step * d for a, d in zip(q, delta, strict=True)]
            trial_norm = norm(local(trial, reaction, forcing))
            armijo = (1. - 1.e-4 * step) * current
            report["trials"].append(dict(iteration=iteration, step=step, physical_norm=trial_norm, armijo=armijo))
            if trial_norm <= armijo:
                q, accepted = trial, True
                break
            step *= .5
        if not accepted:
            report.update(status="line_search_failed", iterations=iteration + 1, physical_residual=current,
                          solution=q, target=target)
            return report
    raise AssertionError("independent constant witness exhausted Newton budget")


def exact_bad_direction():
    reaction = [[Fraction(value) for value in row] for row in REACTION]
    target = list(map(Fraction, TARGET))
    cubic = Fraction(.2)
    forcing = [sum(reaction[i][j] * target[j] for j in range(3)) + cubic * target[i]**3 for i in range(3)]
    delta = solve3(reaction, [-value for value in forcing])
    if not all(value < 0 for value in delta) or not all(value > 0 for value in forcing):
        raise AssertionError("monotone wrong-direction certificate changed")
    # R*delta=-forcing; at alpha>0 every residual component is strictly more
    # negative: -(1+alpha)*forcing + cubic*alpha^3*delta^3 < -forcing.
    return dict(direction=[float(x) for x in delta], forcing=[float(x) for x in forcing],
                all_components_strictly_uphill_for_every_positive_step=True)


def hierarchy_squared(vector, cells, ranks, *, replicated=False):
    partials = []
    for rank in range(ranks):
        value = 0.
        for level in (0, 1):
            patches = ((0, cells//2), (cells//2, cells)) if level == 0 else ((cells//2, cells), (cells, 3*cells//2))
            measure = 1. / (cells * (1 if level == 0 else 2))
            for patch, (lower, upper) in enumerate(patches):
                if not replicated and (patch if ranks > 1 else 0) != rank:
                    continue
                if replicated and rank != 0:
                    continue
                for index in range(lower, upper):
                    if level == 0 and cells//4 <= index < 3*cells//4:
                        continue
                    value += measure * math.fsum(x*x for x in vector)
        partials.append(value)
    return partials, math.fsum(partials)


def receive(root):
    contract = source_contract(root)
    bad, good = run_original_constant(1), run_original_constant(-1)
    if bad["status"] != "line_search_failed" or good["status"] != "converged":
        raise AssertionError("original-equation counterexample changed")
    return dict(schema="sol61.amr-original-newton-source-math@1", source_contract=contract,
                status="sign_defect_detected" if contract["defect_sign"] == 1 else "sign_convention_compatible",
                wrong_sign=bad, corrected_convention=good, exact_certificate=exact_bad_direction(),
                native_execution=False, mpi_execution=False, saved_states_received=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = receive(args.checkout.resolve())
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: receipt[key] for key in ("status", "native_execution", "mpi_execution")}))
