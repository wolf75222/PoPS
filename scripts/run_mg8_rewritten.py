#!/usr/bin/env python3
"""The « rewritten » lane of MG8: the vertical slice against master's tutorial 02 (OD31, S1-c).

The case file of the slice (``tests/gates/mg8_rewritten.toml``, ``case``) is master's tutorial 02
written in the surface. ``run`` executes it from a fresh copy, under the environment of the MG8
lane of the contract, after ``run_mg8_tutorials.py run --phase reference`` has run master's
tutorials on the same machine, and compares its output with master's archive:

* the file as written (``as-written``): the eleven scalars of ``metadata=`` equal; master's steps
  and final time; master's ``initial`` and ``final`` against the records of ``u`` at t = 0 and
  t = ``t_end``, transposed (2.6.5); at t = 0 the cells that differ are among the cells of
  ``[bound]``, each by at most ``ulps`` units in the last place, and every other cell is equal bit
  for bit (a recorded cell that no longer differs is named, so that the bound shrinks); at
  t = ``t_end`` the largest difference relative to the largest value of master is at most
  ``final_relative``;
* the companion (``master-initial``): the same file with its initial value replaced by master's
  ``initial`` array bound as data; both records equal master bit for bit. It shows that the bound
  covers the initial data only (driver's ruling, ``tranche/decisions_pilote.md``, « S1-c, écart
  exp »).

``[bound]`` names its cause (C44/G07: ``pops.project`` evaluates ``exp`` on b6's analytic route,
master in NumPy). It only shrinks (MG6, ``check --check-base``) and leaves when G07 delivers
initial data without the analytic VM; a cell that differs outside it fails (CF14) and goes back
to the driver, never to a wider bound without a ruling (« S1-c, suite »).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "tests" / "gates" / "mg8_rewritten.toml"
GATE = "mg8-rewritten"
KEYS = {"schema_version", "gate", "tutorial", "lane", "case", "output", "scalars", "array",
        "bound"}
BOUND_KEYS = {"cause", "issue", "cells", "ulps", "final_relative"}
# The companion: the statements of the case file that it replaces, each exactly once.
COMPANION = (
    ('u = model.quantity("u", pops.Scalar(domain))',
     'u = model.quantity("u", pops.Scalar(domain))\n'
     'master_initial = model.data("master_initial", pops.Scalar(domain))'),
    ("initial={u: pops.project(bump, quadrature=GaussLegendre(1))},",
     "initial={u: master_initial},"),
    ("simulation = pops.bind(artifact)",
     "import numpy\n"
     "simulation = pops.bind(artifact, data={master_initial: numpy.load('master_initial.npy')})"),
)


class ContractError(ValueError):
    """The contract of the rewritten lane breaks a rule."""


def _mg8() -> Any:
    spec = importlib.util.spec_from_file_location("run_mg8_tutorials",
                                                  ROOT / "scripts" / "run_mg8_tutorials.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(text: str) -> dict:
    data = tomllib.loads(text)
    if set(data) - KEYS or not KEYS - {"bound"} <= set(data):
        raise ContractError("the contract has exactly %s, and bound" % sorted(KEYS - {"bound"}))
    if data["schema_version"] != 1 or data["gate"] != GATE:
        raise ContractError("schema_version 1 and gate %r are required" % GATE)
    for row in data["array"]:
        if set(row) != {"master", "record", "index", "transpose"}:
            raise ContractError("an [[array]] names master, record, index and transpose: %r"
                                % (row,))
    bound = data.get("bound")
    if bound is not None:
        if set(bound) != BOUND_KEYS:
            raise ContractError("[bound] names exactly %s" % sorted(BOUND_KEYS))
        cells = [tuple(cell) for cell in bound["cells"]]
        if not cells or len(set(cells)) != len(cells) or any(len(cell) != 2 for cell in cells):
            raise ContractError("[bound] lists distinct cells (i, j) of master's initial array")
        if not (isinstance(bound["ulps"], int) and bound["ulps"] >= 1) \
                or not 0 < bound["final_relative"] <= 1e-12:
            raise ContractError("[bound]: ulps >= 1 and 0 < final_relative <= 1e-12 (T7)")
    return data


def read(path: Path = CONTRACT) -> dict:
    return load(path.read_text(encoding="utf-8"))


def grown(current: dict, base: dict) -> list:
    """How the bound records more than its base (MG6: it only shrinks)."""
    problems = []
    for key in sorted(KEYS - {"bound"}):
        if current.get(key) != base.get(key):
            problems.append("%s changed" % key)
    new, old = current.get("bound"), base.get("bound")
    if new is None:
        return problems
    if old is None:
        return problems + ["[bound] is not in the base"]
    if new["cause"] != old["cause"] or new["issue"] != old["issue"]:
        problems.append("[bound] changed its cause")
    extra = sorted({tuple(cell) for cell in new["cells"]} - {tuple(cell) for cell in old["cells"]})
    if extra:
        problems.append("[bound] gains the cells %s" % extra)
    if new["ulps"] > old["ulps"] or new["final_relative"] > old["final_relative"]:
        problems.append("[bound] widens")
    return problems


def companion_source(source: str) -> str:
    for old, new in COMPANION:
        if source.count(old) != 1:
            raise ContractError("the companion replaces %r, which the case file holds %d times"
                                % (old, source.count(old)))
        source = source.replace(old, new)
    return source


def compare(contract: dict, master: Any, record: Any, steps: tuple, variant: str,
            notes: list | None = None) -> list:
    """The problems of one run of the case file against master's archive (numpy arrays);
    ``notes`` receives the recorded cells that no longer differ."""
    import numpy as np

    problems = []
    for name in contract["scalars"]:
        if name not in record.files or not np.array_equal(master[name], record[name]):
            problems.append("%s: the scalar %s differs from master's" % (variant, name))
    record_steps = (int(record["step"][-1]), float(record["time"][-1]))
    if record_steps != steps:
        problems.append("%s: (steps, final time) %s, master %s" % (variant, record_steps, steps))
    bound = contract.get("bound") if variant == "as-written" else None
    for row in contract["array"]:
        expected = np.asarray(master[row["master"]])[0]
        actual = np.asarray(record[row["record"]])[row["index"]]
        if row["transpose"]:
            actual = actual.T
        if expected.shape != actual.shape:
            problems.append("%s: %s has shape %s, master %s"
                            % (variant, row["master"], actual.shape, expected.shape))
            continue
        differ = sorted(map(tuple, np.argwhere(expected != actual).tolist()))
        if bound is None:
            if differ:
                problems.append("%s: %s differs from master's in %d cells (bit for bit)"
                                % (variant, row["master"], len(differ)))
            continue
        if row["index"] == 0:
            # the differing cells are among the recorded ones (driver's ruling « S1-c, suite »:
            # NumPy and glibc dispatch exp on CPU features); a cell outside them fails and goes
            # back to the driver, and a recorded cell that no longer differs is named so that
            # the set shrinks (MG6)
            cells = sorted(tuple(cell) for cell in bound["cells"])
            outside = sorted(set(differ) - set(cells))
            if outside:
                problems.append("%s: %s differs in the cells %s, outside the bound %s"
                                % (variant, row["master"], outside, cells))
            unused = sorted(set(cells) - set(differ))
            if unused and notes is not None:
                notes.append("%s: %s equals master's in the recorded cells %s: shrink the bound"
                             % (variant, row["master"], unused))
            gaps = [abs(int(np.float64(expected[cell]).view(np.int64))
                        - int(np.float64(actual[cell]).view(np.int64))) for cell in differ]
            if gaps and max(gaps) > bound["ulps"]:
                problems.append("%s: %s differs by %d ulp, the bound is %d"
                                % (variant, row["master"], max(gaps), bound["ulps"]))
        else:
            relative = float(np.max(np.abs(expected - actual)) / np.max(np.abs(expected)))
            if relative > bound["final_relative"]:
                problems.append("%s: %s differs by %.3e relative, the bound is %.0e"
                                % (variant, row["master"], relative, bound["final_relative"]))
    return problems


def _master_steps(reference: Path, tutorial: str) -> tuple:
    rank = json.loads((reference / tutorial / "record" / "rank0.json").read_text("utf-8"))
    last = rank["observations"][-1]
    return int(last["macro_step"]), float.fromhex(last["time"])


def run(args: argparse.Namespace) -> int:
    import numpy as np

    mg8 = _mg8()
    contract = read(Path(args.contract))
    lane = mg8.load_contract(Path(args.mg8_contract))["lanes"][contract["lane"]]
    reference = Path(args.reference).resolve()
    output = Path(args.output).resolve()
    master_archive = reference / contract["tutorial"] / "tutorial" / contract["output"]
    master = np.load(master_archive)
    steps = _master_steps(reference, contract["tutorial"])
    source = (Path(args.source) / contract["case"]).read_text(encoding="utf-8")
    verdict: dict[str, Any] = {"gate": GATE, "master_steps": list(steps), "runs": []}
    problems: list = []
    notes: list = []
    for variant, text in (("as-written", source), ("master-initial", companion_source(source))):
        work = output / variant
        if work.exists():
            shutil.rmtree(work)
        case = work / "case"
        case.mkdir(parents=True)
        (case / "case.py").write_text(text, encoding="utf-8")
        np.save(case / "master_initial.npy", np.ascontiguousarray(master["initial"][0].T))
        environment = mg8._tutorial_environment(args, lane, work)
        log = work / "case.log"
        with log.open("w", encoding="utf-8") as stream:
            code, timeout, _ = mg8._launch([str(Path(args.python).absolute()), "case.py"], case,
                                           environment, stream, args.watchdog)
        found = []
        if code != 0:
            found.append("%s: the case file exits with %d%s; its log is %s"
                         % (variant, code, " (watchdog)" if timeout else "", log))
            print(log.read_text(encoding="utf-8")[-4000:], file=sys.stderr)
        else:
            found = compare(contract, master, np.load(case / contract["output"]), steps, variant,
                            notes)
        verdict["runs"].append({"variant": variant, "returncode": code, "problems": found})
        problems += found
    verdict["notes"] = notes
    for note in notes:
        print("MG8 rewritten (note): %s" % note)
    output.mkdir(parents=True, exist_ok=True)
    (output / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    for problem in problems:
        print("MG8 rewritten: %s" % problem, file=sys.stderr)
    if not problems:
        print("MG8 rewritten: the slice gives master's tutorial %s (as written, within its "
              "bound; with master's initial data, bit for bit)" % contract["tutorial"])
    return 1 if problems else 0


def check(args: argparse.Namespace) -> int:
    problems = []
    try:
        contract = read(Path(args.contract))
        if not (Path(args.source) / contract["case"]).is_file():
            problems.append("the case file %s does not exist" % contract["case"])
        else:
            companion_source((Path(args.source) / contract["case"]).read_text("utf-8"))
        if args.check_base:
            relative = Path(args.contract).resolve().relative_to(Path(args.source).resolve())
            shown = subprocess.run(["git", "-C", args.source, "show",
                                    "%s:%s" % (args.check_base, relative.as_posix())],
                                   capture_output=True, text=True)
            if shown.returncode == 0:
                problems += grown(contract, load(shown.stdout))
    except ContractError as error:
        problems.append(str(error))
    for problem in problems:
        print("MG8 rewritten: %s" % problem, file=sys.stderr)
    return 1 if problems else 0


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    running = commands.add_parser("run", help="run the slice and compare it with master's run")
    running.add_argument("--source", default=str(ROOT))
    running.add_argument("--reference", required=True,
                         help="the output of run_mg8_tutorials.py run --phase reference")
    running.add_argument("--output", required=True)
    running.add_argument("--python", required=True)
    running.add_argument("--pythonpath", action="append", default=[])
    running.add_argument("--include")
    running.add_argument("--kokkos-root")
    running.add_argument("--watchdog", type=int, default=1800)
    running.add_argument("--contract", default=str(CONTRACT))
    running.add_argument("--mg8-contract",
                         default=str(ROOT / "tests" / "gates" / "mg8_master_tutorials.toml"))
    checking = commands.add_parser("check", help="check the contract and its base")
    checking.add_argument("--source", default=str(ROOT))
    checking.add_argument("--contract", default=str(CONTRACT))
    checking.add_argument("--check-base", metavar="REVISION")
    arguments = parser.parse_args(argv)
    return run(arguments) if arguments.command == "run" else check(arguments)


if __name__ == "__main__":
    sys.exit(main())
