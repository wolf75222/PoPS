#!/usr/bin/env python3
"""Replay master's scalar-advection tutorials unchanged and compare their results (MG8).

``run`` executes the tutorials of one source tree.  Each tutorial runs from a fresh copy of the
tutorial directory, so its outputs never touch the checkout, under one installed PoPS and one
pinned environment.  A probe inside the tutorial process wraps ``pops.run``: after every call it
reads the clock, the hierarchy and every state level through the public methods that the
tutorials themselves call, and saves the arrays with their SHA-256.  The tutorial's own printed
lines, the console-monitor lines and every file that it writes are recorded as well.  The script
of the tutorial is never edited and its return code is the verdict of its own assertions.

``compare`` checks a candidate run against a reference run of ``master`` made on the same
machine: return codes, every observation bitwise, the printed and monitor lines exactly, and the
outputs of each tutorial by kind (NumPy archives, ParaView series, bytes).  The contract
``tests/gates/mg8_master_tutorials.toml`` names what is not compared (identities, the run
directory) and records the divergences of the candidate: per tutorial, the exact checks that
differ and their causes, each cause declared once with its commits, its issue and its evidence.
Only a result can be recorded, by its exact name: a tutorial that fails, a run that is missing,
an observation that could not be made, a missing or unlisted output and a different number of
``pops.run`` calls or printed lines always fail (CF14).  The recorded checks only shrink (MG6):
``compare`` fails on a recorded check that no longer differs, and ``check --check-base`` fails
on a ledger that grows against the contract of a base revision.

``archive`` packs a run deterministically (the reference of master is kept outside the code
tree, MG8-0), and ``check`` validates the contract against a source tree without running anything:
the tutorials on disk, the ledger against a base revision (``--check-base``) and the commits that
the causes cite (``--verify-commits``, which needs the history).
"""

from __future__ import annotations

import argparse
import base64
import builtins
from collections.abc import Iterable, Mapping, Sequence
import fnmatch
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import runpy
import shutil
import subprocess
import sys
import time
import tomllib
import traceback
from typing import Any
import xml.etree.ElementTree as ET
import zlib


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "tests/gates/mg8_master_tutorials.toml"
SCHEMA = "pops.mg8.master-tutorials.v1"
CONTRACT_SCHEMA_VERSION = 1
OUTPUT_KINDS = {"bytes", "npz", "paraview", "record"}
# The results that a divergence may record, by exact name (MG8): what compare finds different
# after a run that completed and was observed. Every other check (a return code, a missing run,
# an observation that failed, a different number of pops.run calls or printed lines, a missing,
# different or unlisted set of output files) is a failure and is never recorded (CF14).
OBSERVED_FIELDS = ("controls", "time", "macro_step", "n_levels", "configured_n_levels", "blocks",
                   "patch_boxes", "report", "states")
RECORDABLE = (
    re.compile(r"run\d+:rank\d+:(%s)" % "|".join(OBSERVED_FIELDS)),
    re.compile(r"run\d+:rank\d+:run\d+-[A-Za-z0-9_.-]+:(dtype|shape|sha256)"),
    re.compile(r"printed:rank\d+:line\d+"),
    re.compile(r"monitor:rank\d+"),
)
PARAVIEW_CHECKS = ("series", "recipes", "kinds")
# A divergence says whether a build of this tree with master's semantics for its causes was
# run and gave master's results ("reproduced", with that run) or not ("inferred").
ATTRIBUTIONS = ("reproduced", "inferred")
# String arrays of ParaView outputs are compared, except the output manifest: JSON that carries
# versioned identities, which differ between master and this tree like the identity lines.
IDENTITY_ARRAYS = frozenset({"pops_output_manifest"})
COMPARED_REPORT_FIELDS = (
    "accepted_steps", "rejected_steps", "final_time", "final_macro_step", "stop_reason",
)
# The environment of a tutorial is rebuilt from these names only: anything else (PYTHONPATH,
# POPS_THREADS, POPS_CATALYST, POPS_PARAVIEW_ROOT, POPS_DSL_OPTFLAGS, ...) would change what the
# tutorial runs without changing its file.
INHERITED_ENVIRONMENT = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR", "CC", "CXX")
INHERITED_PREFIXES = ("OMPI_", "OPAL_", "PRTE_", "PMIX_", "HYDRA_", "MPICH_", "I_MPI_")
RANK_VARIABLES = ("OMPI_COMM_WORLD_RANK", "PMIX_RANK", "PMI_RANK", "SLURM_PROCID")
MATPLOTLIBRC = "backend: Agg\nsvg.hashsalt: pops-mg8-master-tutorials\n"


class ContractError(RuntimeError):
    """The MG8 contract does not describe the tutorials of the tree exactly."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recordable(check: str, row: Mapping[str, Any]) -> bool:
    """Whether ``check`` names a result that compare can find different for tutorial ``row``."""
    if any(pattern.fullmatch(check) for pattern in RECORDABLE):
        return True
    for output in row.get("outputs", ()):
        prefix = "output:%s:" % output["pattern"]
        if check.startswith(prefix):
            rest = check[len(prefix):]
            if output["kind"] == "paraview":
                return rest in PARAVIEW_CHECKS
            if output["kind"] in ("bytes", "npz"):
                return fnmatch.fnmatchcase(rest, output["pattern"])
    return False


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------
# Contract


def load_contract(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return parse_contract(tomllib.load(stream))


def parse_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ContractError("unsupported MG8 contract schema_version")
    if contract.get("gate") != "mg8-master-tutorials":
        raise ContractError("the MG8 contract must name gate = 'mg8-master-tutorials'")
    master = contract.get("master")
    if not isinstance(master, str) or not re.fullmatch(r"[0-9a-f]{40}", master):
        raise ContractError("master must be one full commit id")
    lanes = contract.get("lanes")
    if not isinstance(lanes, Mapping) or not lanes:
        raise ContractError("the MG8 contract declares no lane")
    for name, lane in lanes.items():
        if set(lane) != {"dimension", "ranks", "mpi"}:
            raise ContractError("lane %s must declare exactly dimension, ranks and mpi" % name)
        if lane["dimension"] not in (1, 2, 3) or not isinstance(lane["ranks"], int) \
                or lane["ranks"] < 1 or not isinstance(lane["mpi"], bool) \
                or (lane["ranks"] > 1 and not lane["mpi"]):
            raise ContractError("lane %s is not a valid native dimension and launch" % name)
    re.compile(contract.get("monitor", ""))
    tutorials = contract.get("tutorial")
    if not isinstance(tutorials, list) or not tutorials:
        raise ContractError("the MG8 contract lists no tutorial")
    seen: set[str] = set()
    for row in tutorials:
        unknown = set(row) - {"id", "script", "lane", "outputs", "identity_lines"}
        if unknown:
            raise ContractError("tutorial %s has unknown keys %s" % (row.get("id"), sorted(unknown)))
        if row.get("id") in seen or not re.fullmatch(r"\d\d", str(row.get("id", ""))):
            raise ContractError("tutorial ids must be unique two-digit numbers")
        seen.add(row["id"])
        if not str(row.get("script", "")).startswith(row["id"] + "_"):
            raise ContractError("tutorial %s names a script of another number" % row["id"])
        if row.get("lane") not in lanes:
            raise ContractError("tutorial %s names an undeclared lane" % row["id"])
        for output in row.get("outputs", ()):
            if output.get("kind") not in OUTPUT_KINDS or not output.get("pattern"):
                raise ContractError("tutorial %s has an output without pattern or kind" % row["id"])
            if output["kind"] == "record" and not output.get("reason"):
                raise ContractError("an output of tutorial %s is recorded without a reason"
                                    % row["id"])
    causes: dict[str, Mapping[str, Any]] = {}
    for row in contract.get("cause", ()):
        if set(row) != {"id", "commits", "issue", "text", "evidence"}:
            raise ContractError("each cause names id, commits, issue, text and evidence")
        if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", str(row["id"])) or row["id"] in causes:
            raise ContractError("cause ids must be unique kebab-case names")
        if not isinstance(row["commits"], list) or not row["commits"] or not all(
                isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{7,40}", commit)
                for commit in row["commits"]):
            raise ContractError("cause %s must name the commits that introduced it" % row["id"])
        if not all(isinstance(row[key], str) and row[key].strip()
                   for key in ("issue", "text", "evidence")):
            raise ContractError("cause %s must state its issue, text and evidence" % row["id"])
        causes[row["id"]] = row
    cited: set[str] = set()
    recorded: set[str] = set()
    rows = {row["id"]: row for row in tutorials}
    for row in contract.get("divergence", ()):
        keys = {"tutorial", "causes", "checks", "attribution"}
        if row.get("attribution") == "reproduced":
            keys.add("reproduction")
        if set(row) != keys or row["tutorial"] not in seen or row["tutorial"] in recorded:
            raise ContractError("each tutorial has at most one divergence, which names its "
                                "causes, checks and attribution (and its reproduction when "
                                "reproduced)")
        recorded.add(row["tutorial"])
        if row["attribution"] not in ATTRIBUTIONS or (
                row["attribution"] == "reproduced"
                and not (isinstance(row["reproduction"], str) and row["reproduction"].strip())):
            raise ContractError("the divergence of tutorial %s must say whether its attribution "
                                "is reproduced, by which run, or inferred" % row["tutorial"])
        if not isinstance(row["causes"], list) or not row["causes"] \
                or len(set(row["causes"])) != len(row["causes"]) \
                or not set(row["causes"]) <= set(causes):
            raise ContractError("the divergence of tutorial %s cites undeclared causes"
                                % row["tutorial"])
        if not isinstance(row["checks"], list) or not row["checks"] \
                or len(set(row["checks"])) != len(row["checks"]) \
                or not all(isinstance(check, str) and check for check in row["checks"]):
            raise ContractError("the divergence of tutorial %s needs distinct result checks"
                                % row["tutorial"])
        failures = [check for check in row["checks"]
                    if not recordable(check, rows[row["tutorial"]])]
        if failures:
            raise ContractError("the divergence of tutorial %s records %s: not results that "
                                "compare can find different, failures are never recorded (CF14)"
                                % (row["tutorial"], failures))
        cited.update(row["causes"])
    if set(causes) - cited:
        raise ContractError("causes cited by no divergence: %s" % sorted(set(causes) - cited))
    return contract


def check_contract(contract: Mapping[str, Any], source: Path) -> list[str]:
    """Return the problems between the contract and the tutorials of ``source``."""
    directory = source / contract["directory"]
    on_disk = sorted(path.name for path in directory.glob("[0-9][0-9]_*.py"))
    listed = sorted(row["script"] for row in contract["tutorial"])
    problems = []
    if on_disk != listed:
        problems.append("tutorials on disk %s differ from the contract %s" % (on_disk, listed))
    return problems


def ledger_growth(base: Mapping[str, Any], head: Mapping[str, Any]) -> list[str]:
    """Return how the contract ``head`` records more than ``base`` (MG6: the ledger only shrinks).

    Everything but the ledger stays as it is, except an output that becomes compared; a cause
    keeps its issue; a divergence may lose checks and causes and turn from inferred to
    reproduced. Texts, evidence and the commits of a cause may be corrected: the commits admit
    nothing, and ``verify_commits`` checks each of them.
    """
    problems = []
    for key in ("schema_version", "gate", "master", "directory", "monitor", "lanes"):
        if base.get(key) != head.get(key):
            problems.append("%s changed" % key)
    old_rows = {row.get("id"): row for row in base.get("tutorial", ())}
    new_rows = {row.get("id"): row for row in head.get("tutorial", ())}
    if sorted(old_rows) != sorted(new_rows):
        problems.append("the tutorials changed: %s, then %s" % (sorted(old_rows), sorted(new_rows)))
    for tutorial in sorted(set(old_rows) & set(new_rows)):
        old, new = old_rows[tutorial], new_rows[tutorial]
        for key in ("script", "lane"):
            if old.get(key) != new.get(key):
                problems.append("tutorial %s: %s changed" % (tutorial, key))
        if not set(new.get("identity_lines", ())) <= set(old.get("identity_lines", ())):
            problems.append("tutorial %s: identity_lines grew" % tutorial)
        old_outputs = {output.get("pattern"): output for output in old.get("outputs", ())}
        new_outputs = {output.get("pattern"): output for output in new.get("outputs", ())}
        if sorted(old_outputs) != sorted(new_outputs):
            problems.append("tutorial %s: its outputs changed" % tutorial)
        for pattern in sorted(set(old_outputs) & set(new_outputs)):
            kinds = old_outputs[pattern].get("kind"), new_outputs[pattern].get("kind")
            if kinds[0] != kinds[1] and not (kinds[0] == "record" and kinds[1] != "record"):
                problems.append("tutorial %s: output %s went from %s to %s"
                                % (tutorial, pattern, kinds[0], kinds[1]))
    old_causes = {row.get("id"): row for row in base.get("cause", ())}
    for cause in head.get("cause", ()):
        old = old_causes.get(cause.get("id"))
        if old is None:
            problems.append("new cause %s" % cause.get("id"))
            continue
        if cause.get("issue") != old.get("issue"):
            problems.append("cause %s changed its issue" % cause["id"])
    old_divergences = {row.get("tutorial"): row for row in base.get("divergence", ())}
    for row in head.get("divergence", ()):
        old = old_divergences.get(row.get("tutorial"))
        if old is None:
            problems.append("new divergence of tutorial %s" % row.get("tutorial"))
            continue
        for key in ("checks", "causes"):
            added = sorted(set(row.get(key, ())) - set(old.get(key, ())))
            if added:
                problems.append("tutorial %s records new %s %s" % (row["tutorial"], key, added))
        if old.get("attribution") == "reproduced" and row.get("attribution") != "reproduced":
            problems.append("tutorial %s: a reproduced attribution became %s"
                            % (row["tutorial"], row.get("attribution")))
    return problems


def _git(source: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", "-C", str(source), *arguments], capture_output=True, text=True,
                          check=False)


def contract_at(source: Path, revision: str, contract: Path) -> dict[str, Any] | None:
    """Return the contract of ``revision`` as TOML data, or None when it has none yet."""
    if _git(source, "cat-file", "-e", revision + "^{commit}").returncode != 0:
        raise ContractError("the base %s is not in this clone: fetch it" % revision)
    top = Path(_git(source, "rev-parse", "--show-toplevel").stdout.strip())
    shown = _git(source, "show", "%s:%s" % (
        revision, contract.resolve().relative_to(top.resolve()).as_posix()))
    return tomllib.loads(shown.stdout) if shown.returncode == 0 else None


# The paths whose code the tutorials execute: the package, its native module and their build.
EXECUTED_PATHS = ("include/", "src/", "python/", "cmake/")


def executed(path: str) -> bool:
    return path == "CMakeLists.txt" or path.startswith(EXECUTED_PATHS)


def verify_commits(contract: Mapping[str, Any], source: Path) -> list[str]:
    """Each cited commit is in the history of HEAD and changes code that the tutorials execute."""
    shallow_file = Path(_git(source, "rev-parse", "--git-common-dir").stdout.strip())
    if not shallow_file.is_absolute():
        shallow_file = source / shallow_file
    shallow_file = shallow_file / "shallow"
    shallow = set(shallow_file.read_text().split()) if shallow_file.is_file() else set()
    problems = []
    for cause in contract.get("cause", ()):
        for commit in cause["commits"]:
            full = _git(source, "rev-parse", "--verify", "--quiet", commit + "^{commit}")
            if full.returncode != 0:
                problems.append("cause %s cites %s, which is not in this clone: fetch the "
                                "history" % (cause["id"], commit))
                continue
            full_id = full.stdout.strip()
            if full_id in shallow:
                problems.append("cause %s cites %s, the boundary of a shallow clone, whose "
                                "changes are unknown: fetch the history" % (cause["id"], commit))
                continue
            if _git(source, "merge-base", "--is-ancestor", full_id, "HEAD").returncode != 0:
                problems.append("cause %s cites %s, which is not in the history of HEAD"
                                % (cause["id"], commit))
                continue
            paths = _git(source, "diff-tree", "--root", "--no-commit-id", "--name-only", "-r",
                         "-m", "--first-parent", full_id).stdout.split()
            if not any(executed(path) for path in paths):
                problems.append("cause %s cites %s, which changes no path that the tutorials "
                                "execute (%s, CMakeLists.txt)"
                                % (cause["id"], commit, ", ".join(EXECUTED_PATHS)))
    return problems


def tutorial_rows(contract: Mapping[str, Any], selected: Sequence[str],
                  excluded: Sequence[str] = ()) -> list[dict[str, Any]]:
    rows = [dict(row) for row in contract["tutorial"]]
    unknown = (set(selected) | set(excluded)) - {row["id"] for row in rows}
    if unknown:
        raise ContractError("unknown tutorials %s" % sorted(unknown))
    return [row for row in rows if (not selected or row["id"] in selected)
            and row["id"] not in excluded]


# ---------------------------------------------------------------------------------------------
# Run


def _inventory(root: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            rows[path.relative_to(root).as_posix()] = {
                "sha256": sha256_file(path), "size": path.stat().st_size}
    return rows


def _tutorial_environment(args: argparse.Namespace, lane: Mapping[str, Any],
                          work: Path) -> dict[str, str]:
    environment = {name: os.environ[name] for name in INHERITED_ENVIRONMENT if name in os.environ}
    environment.update({name: value for name, value in os.environ.items()
                        if name.startswith(INHERITED_PREFIXES)})
    mplconfig = work / "matplotlib"
    mplconfig.mkdir(parents=True, exist_ok=True)
    (mplconfig / "matplotlibrc").write_text(MATPLOTLIBRC, encoding="utf-8")
    environment.update(
        POPS_NATIVE_DIM=str(lane["dimension"]),
        POPS_CACHE_DIR=str(work / "cache" / "dsl"),
        XDG_CACHE_HOME=str(work / "cache" / "xdg"),
        MPLCONFIGDIR=str(mplconfig),
        MPLBACKEND="Agg",
        SOURCE_DATE_EPOCH="0",
        PYTHONHASHSEED="0",
        PYTHONNOUSERSITE="1",
        PYTHONUNBUFFERED="1",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OMP_PROC_BIND="false",
    )
    if args.pythonpath:
        environment["PYTHONPATH"] = os.pathsep.join(str(Path(p).resolve()) for p in args.pythonpath)
    if args.include:
        environment["POPS_INCLUDE"] = str(Path(args.include).resolve())
    if args.kokkos_root:
        environment["Kokkos_ROOT"] = environment["POPS_KOKKOS_ROOT"] = str(
            Path(args.kokkos_root).resolve())
    return environment


def _native_identity(python: str, environment: Mapping[str, str], cwd: Path) -> dict[str, Any]:
    # NumPy's runtime (its SIMD extensions) and the versions of the tutorials' libraries name
    # what the elementary functions of the reference dispatch on, next to the CPU (O1).
    code = (
        "import hashlib, json, os, matplotlib, numpy, pops; "
        "from pops._native_selector import select_native_dimension; "
        "native = select_native_dimension(int(os.environ['POPS_NATIVE_DIM'])); "
        "numpy.show_runtime(); "
        "print(json.dumps({'package': pops.__file__, 'native': native.__file__, "
        "'version': pops.__version__, 'native_sha256': hashlib.sha256("
        "open(native.__file__, 'rb').read()).hexdigest(), 'numpy': numpy.__version__, "
        "'matplotlib': matplotlib.__version__}))"
    )
    completed = subprocess.run([python, "-c", code], env=dict(environment), cwd=cwd,
                               capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return {"error": completed.stderr.strip()[-4000:]}
    lines = completed.stdout.strip().splitlines()
    return {**json.loads(lines[-1]), "numpy_runtime": "\n".join(lines[:-1])}


def run_tutorials(args: argparse.Namespace) -> int:
    contract = load_contract(Path(args.contract))
    source = Path(args.source).resolve()
    problems = check_contract(contract, source)
    if problems:
        raise ContractError("; ".join(problems))
    lane = contract["lanes"][args.lane]
    rows = [row for row in tutorial_rows(contract, args.tutorial, args.exclude)
            if row["lane"] == args.lane]
    if not rows:
        raise ContractError("no selected tutorial runs on lane %s" % args.lane)
    # A run of a selection writes its own report: two steps of one lane may share an output.
    selection = "" if not (args.tutorial or args.exclude) else "-" + "-".join(
        row["id"] for row in rows)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    python = str(Path(args.python).absolute())
    revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=False).stdout.strip()
    status = subprocess.run(["git", "-C", str(source), "status", "--porcelain", "--",
                             contract["directory"]], capture_output=True, text=True,
                            check=False).stdout
    tree = subprocess.run(["git", "-C", str(source), "status", "--porcelain",
                           "--untracked-files=no"], capture_output=True, text=True, check=False)
    tutorial_root = source / contract["directory"]
    report_path = output / ("report-%s%s.json" % (args.lane, selection))
    report: dict[str, Any] = {
        "schema": SCHEMA, "phase": args.phase, "lane": args.lane, "lane_contract": lane,
        "runner_sha256": sha256_file(Path(__file__)),
        "contract_sha256": sha256_file(Path(args.contract)),
        "source": str(source), "revision": revision, "tutorials_dirty": bool(status.strip()),
        "tree_dirty": tree.returncode != 0 or bool(tree.stdout.strip()),
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": _processor(), "python": python, "tutorials": [],
    }
    for row in rows:
        work = output / row["id"]
        if work.exists():
            shutil.rmtree(work)
        copy = work / "tutorial"
        shutil.copytree(tutorial_root, copy, ignore=shutil.ignore_patterns(
            "results", "__pycache__", "*.pyc"))
        pristine = _inventory(copy)
        record = work / "record"
        record.mkdir(parents=True)
        environment = _tutorial_environment(args, lane, work)
        if "native" not in report:
            report["native"] = _native_identity(python, environment, copy)
            report["environment"] = {key: value for key, value in sorted(environment.items())
                                     if key not in ("HOME", "USER", "LOGNAME", "PATH")}
        command = [python, str(Path(__file__).resolve()), "probe",
                   "--tutorial", str(copy / row["script"]), "--record", str(record)]
        if lane["mpi"]:
            if not args.mpiexec:
                raise ContractError("lane %s needs --mpiexec" % args.lane)
            command = [args.mpiexec, "-n", str(lane["ranks"]), *args.mpiexec_arg, *command]
        launcher_log = work / "launcher.log"
        started = time.monotonic()
        timeout = False
        with launcher_log.open("w", encoding="utf-8") as stream:
            try:
                completed = subprocess.run(command, cwd=copy, env=environment, stdout=stream,
                                           stderr=subprocess.STDOUT, timeout=args.watchdog,
                                           check=False)
                code = completed.returncode
            except subprocess.TimeoutExpired:
                code, timeout = 124, True
        seconds = time.monotonic() - started
        after = _inventory(copy)
        written = {path: meta for path, meta in after.items() if pristine.get(path) != meta}
        removed = sorted(set(pristine) - set(after))
        ranks = []
        for rank in range(lane["ranks"]):
            rank_record = record / ("rank%d.json" % rank)
            rank_log = record / ("rank%d.log" % rank)
            ranks.append({
                "rank": rank,
                "record": rank_record.relative_to(output).as_posix()
                if rank_record.is_file() else None,
                "log": rank_log.relative_to(output).as_posix() if rank_log.is_file() else None,
                "log_sha256": sha256_file(rank_log) if rank_log.is_file() else None,
            })
        entry = {
            "id": row["id"], "script": row["script"],
            "script_sha256": sha256_file(tutorial_root / row["script"]),
            "command": command, "ranks": ranks, "returncode": code, "timeout": timeout,
            "seconds": round(seconds, 3), "work": work.relative_to(output).as_posix(),
            "launcher_log": launcher_log.relative_to(output).as_posix(),
            "launcher_log_sha256": sha256_file(launcher_log),
            "written": written, "removed": removed,
        }
        report["tutorials"].append(entry)
        write_json(report_path, report)
        print(json.dumps({key: entry[key] for key in ("id", "returncode", "timeout", "seconds")}),
              flush=True)
        if not args.keep_caches:
            shutil.rmtree(work / "cache", ignore_errors=True)
    return int(any(entry["returncode"] for entry in report["tutorials"]))


def _processor() -> dict[str, Any]:
    """Name the CPU and the features that select different floating-point library paths."""
    info: dict[str, Any] = {"name": platform.processor()}
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return info
    for line in text.splitlines():
        if line.startswith("model name") and "model" not in info:
            info["model"] = line.split(":", 1)[1].strip()
        if line.startswith("flags") and "features" not in info:
            flags = set(line.split(":", 1)[1].split())
            info["features"] = sorted(flags & {"sse4_2", "avx", "avx2", "fma", "avx512f",
                                               "avx512dq", "avx512vl", "avx512bw"})
    return info


# ---------------------------------------------------------------------------------------------
# Probe (runs inside the tutorial process)


def _launcher_rank() -> int:
    for name in RANK_VARIABLES:
        value = os.environ.get(name)
        if value is not None and value.isdigit():
            return int(value)
    return 0


def _json_value(value: Any, root: Path | None = None) -> Any:
    """Return an exact JSON form: floats in hexadecimal, paths relative to the tutorial."""
    if isinstance(value, bool | int | str) or value is None:
        return value
    if isinstance(value, float):
        return {"float": value.hex()}
    if isinstance(value, os.PathLike):
        path = Path(value).resolve()
        if root is not None and path.is_relative_to(root):
            return {"path": "<tutorial>/" + path.relative_to(root).as_posix()}
        return {"path": os.fspath(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item, root) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item, root) for item in value]
    return {"type": type(value).__name__}


def _save_array(array: Any, directory: Path, name: str) -> dict[str, Any]:
    import numpy as np

    data = np.ascontiguousarray(np.asarray(array))
    np.save(directory / (name + ".npy"), data, allow_pickle=False)
    return {"name": name, "dtype": data.dtype.str, "shape": list(data.shape),
            "sha256": sha256_bytes(data.tobytes()), "file": name + ".npy"}


def _observe(instance: Any, report: Any, controls: Mapping[str, Any], index: int,
             directory: Path, root: Path) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "index": index, "controls": _json_value(dict(controls), root),
        "time": float(instance.time()).hex(), "macro_step": int(instance.macro_step()),
        "n_levels": int(instance.n_levels()),
        "configured_n_levels": int(instance.configured_n_levels()),
        "blocks": [str(block) for block in instance.block_names()],
    }
    adaptive = observation["configured_n_levels"] > 1
    states = []
    for block in observation["blocks"]:
        if adaptive:
            for level in range(observation["n_levels"]):
                states.append(_save_array(instance.block_level_state_global(block, level),
                                          directory, "run%d-%s-level%d" % (index, block, level)))
        else:
            states.append(_save_array(instance.state_global(block), directory,
                                      "run%d-%s" % (index, block)))
    observation["states"] = states
    if adaptive:
        observation["patch_boxes"] = _json_value(list(instance.patch_boxes()))
    fields = {}
    for name in COMPARED_REPORT_FIELDS:
        value = getattr(report, name, None)
        fields[name] = _json_value(value if isinstance(value, int | float | str) or value is None
                                   else str(getattr(value, "value", value)))
    observation["report"] = fields
    return observation


def probe(args: argparse.Namespace) -> int:
    rank = _launcher_rank()
    record = Path(args.record)
    log = os.open(record / ("rank%d.log" % rank), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(log, 1)
    os.dup2(log, 2)
    os.close(log)
    arrays = record / ("rank%d" % rank)
    arrays.mkdir(exist_ok=True)
    observations: list[dict[str, Any]] = []
    printed: list[str] = []

    def recording_print(*values: Any, sep: str | None = " ", end: str | None = "\n",
                        file: Any = None, flush: bool = False) -> None:
        if file is None or file is sys.stdout:
            printed.append((" " if sep is None else sep).join(str(value) for value in values)
                           + ("\n" if end is None else end))
        builtins.print(*values, sep=sep, end=end, file=file, flush=flush)

    import pops

    original_run = pops.run

    def observed_run(instance: Any, *positional: Any, **controls: Any) -> Any:
        report = original_run(instance, *positional, **controls)
        try:
            observations.append(_observe(instance, report, controls, len(observations), arrays,
                                         tutorial.parent))
        except Exception as error:  # an observation failure is a result, never a crash
            observations.append({"index": len(observations), "error": "%s: %s" % (
                type(error).__name__, error), "traceback": traceback.format_exc()})
        return report

    pops.run = observed_run
    tutorial = Path(args.tutorial).resolve()
    sys.argv = [str(tutorial)]
    sys.path.insert(0, str(tutorial.parent))
    status: dict[str, Any] = {"ok": True}
    try:
        runpy.run_path(str(tutorial), init_globals={"print": recording_print},
                       run_name="__main__")
    except SystemExit as stop:
        if stop.code not in (None, 0):
            status = {"ok": False, "exception": "SystemExit", "message": str(stop.code)}
    except BaseException as error:  # the tutorial's own verdict, recorded as it is
        status = {"ok": False, "exception": type(error).__name__, "message": str(error),
                  "traceback": traceback.format_exc()}
        traceback.print_exc()
    write_json(record / ("rank%d.json" % rank), {
        "rank": rank, "status": status, "observations": observations, "printed": printed})
    sys.stdout.flush()
    sys.stderr.flush()
    return 0 if status["ok"] else 1


# ---------------------------------------------------------------------------------------------
# Readers of tutorial outputs


VTK_TYPES = {
    "Int8": "i1", "UInt8": "u1", "Int16": "i2", "UInt16": "u2", "Int32": "i4", "UInt32": "u4",
    "Int64": "i8", "UInt64": "u8", "Float32": "f4", "Float64": "f8",
}


def _vtk_payload(text: str, header: str, compressed: bool) -> bytes:
    """Decode one inline binary VTK XML DataArray (raw or vtkZLibDataCompressor)."""
    text = "".join(text.split())
    size = 8 if header == "UInt64" else 4
    width = "<u8" if size == 8 else "<u4"
    import numpy as np

    if not compressed:
        raw = base64.b64decode(text)
        count = int(np.frombuffer(raw[:size], dtype=width)[0])
        return raw[size:size + count]
    # The header (block count, block size, last block size, compressed sizes) is encoded apart
    # from the blocks: read the block count first, then the whole header.
    blocks = int(np.frombuffer(base64.b64decode(text[:12])[:size], dtype=width)[0])
    header_bytes = (3 + blocks) * size
    header_chars = 4 * ((header_bytes + 2) // 3)
    numbers = np.frombuffer(base64.b64decode(text[:header_chars])[:header_bytes], dtype=width)
    sizes = [int(value) for value in numbers[3:]]
    data = base64.b64decode(text[header_chars:])
    out, offset = [], 0
    for compressed_size in sizes:
        out.append(zlib.decompress(data[offset:offset + compressed_size]))
        offset += compressed_size
    return b"".join(out)


def vtk_arrays(path: Path) -> dict[str, dict[str, Any]]:
    """Return every DataArray of a VTK XML file by its element path and name.

    Numeric arrays are decoded; other arrays are compared by their decoded bytes, except the
    identity-bearing ones (IDENTITY_ARRAYS) and the declarations of a parallel file.
    """
    import numpy as np

    root = ET.parse(path).getroot()
    header = root.get("header_type", "UInt32")
    compressed = root.get("compressor") == "vtkZLibDataCompressor"
    rows: dict[str, dict[str, Any]] = {}

    def visit(element: ET.Element, trail: tuple[str, ...]) -> None:
        for index, child in enumerate(element):
            name = child.get("Name", "")
            here = trail + ("%s[%d]%s" % (child.tag, index, ":" + name if name else ""),)
            if child.tag in ("DataArray", "PDataArray"):
                kind = child.get("type", "")
                key = "/".join(here)
                if child.tag == "PDataArray" or name in IDENTITY_ARRAYS:
                    rows[key] = {"type": kind, "declared": True}
                    continue
                if kind not in VTK_TYPES:
                    payload = (child.text or "").encode() if child.get("format") == "ascii" \
                        else _vtk_payload(child.text or "", header, compressed)
                    rows[key] = {"type": kind, "sha256": sha256_bytes(payload)}
                    continue
                components = int(child.get("NumberOfComponents", "1"))
                if child.get("format") == "ascii":
                    values = np.array((child.text or "").split(), dtype=VTK_TYPES[kind])
                else:
                    values = np.frombuffer(_vtk_payload(child.text or "", header, compressed),
                                           dtype="<" + VTK_TYPES[kind])
                rows[key] = {"type": kind, "components": components,
                             "count": int(values.size), "sha256": sha256_bytes(values.tobytes())}
            else:
                visit(child, here)
    visit(root, (root.get("type", root.tag),))
    return rows


def paraview_series(root: Path, files: Iterable[str]) -> dict[str, Any]:
    """Summarize ParaView outputs: every PVD in step order and every dataset it names."""
    step = re.compile(r"__s(\d{9})")
    # The PVD of a growing series names every earlier dataset again: read each file once.
    arrays: dict[Path, dict[str, dict[str, Any]]] = {}
    pieces_of: dict[Path, list[dict[str, Any]]] = {}

    def read(path: Path) -> dict[str, dict[str, Any]]:
        if path not in arrays:
            arrays[path] = vtk_arrays(path)
        return arrays[path]

    series = []
    for relative in sorted(files, key=lambda name: (step.findall(name)[-1:] or [""], name)):
        if not relative.endswith(".pvd"):
            continue
        tree = ET.parse(root / relative).getroot()
        datasets = []
        for dataset in tree.iter("DataSet"):
            target = (root / relative).parent / dataset.get("file", "")
            entry: dict[str, Any] = {"timestep": dataset.get("timestep"),
                                     "part": dataset.get("part"), "group": dataset.get("group")}
            found = step.findall(target.name)
            entry["step"] = found[-1] if found else None
            entry["arrays"] = read(target)
            if target.suffix == ".pvtu":
                if target not in pieces_of:
                    pieces_of[target] = [
                        {"step": (step.findall(leaf.name) or [None])[-1], "arrays": read(leaf)}
                        for leaf in (target.parent / piece.get("Source", "")
                                     for piece in ET.parse(target).getroot().iter("Piece"))]
                entry["pieces"] = pieces_of[target]
            datasets.append(entry)
        series.append({"step": (step.findall(relative) or [None])[-1], "datasets": datasets})
    recipes = []
    for relative in sorted(files):
        if relative.endswith(".view.json"):
            payload = read_json(root / relative).get("payload", {})
            recipes.append({"step": (step.findall(relative) or [None])[-1],
                            "cell_arrays": payload.get("cell_arrays"),
                            "point_arrays": payload.get("point_arrays"),
                            "presentation": payload.get("presentation")})
    return {"series": series, "recipes": recipes,
            "kinds": sorted({_suffix(name) for name in files})}


def _suffix(name: str) -> str:
    for suffix in (".view.json", ".view.py", ".pvd", ".pvtu", ".vtu", ".pvsm"):
        if name.endswith(suffix):
            return suffix
    return Path(name).suffix


def npz_arrays(path: Path) -> dict[str, dict[str, Any]]:
    import numpy as np

    rows = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in sorted(archive.files):
            value = np.ascontiguousarray(archive[key])
            rows[key] = {"dtype": value.dtype.str, "shape": list(value.shape),
                         "sha256": sha256_bytes(value.tobytes())}
    return rows


# ---------------------------------------------------------------------------------------------
# Compare


class Comparison:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []

    def differ(self, tutorial: str, check: str, detail: str) -> None:
        self.rows.append({"tutorial": tutorial, "check": check, "detail": detail})


def _normalize_line(line: str, work: str) -> str:
    return line.replace(work, "<tutorial>")


def _printed_lines(record: Mapping[str, Any], work: str) -> list[str]:
    text = "".join(record.get("printed", ()))
    return [_normalize_line(line, work) for line in text.splitlines()]


DIGEST = re.compile(r"[0-9a-f]{12,}")


def _compare_printed(result: Comparison, tutorial: str, rank: int, reference: list[str],
                     candidate: list[str], identity_lines: Sequence[str]) -> None:
    """Compare printed lines exactly; in an identity line only the digests are not compared."""
    if len(reference) != len(candidate):
        result.differ(tutorial, "printed:rank%d" % rank, "%d printed lines on master, %d here"
                      % (len(reference), len(candidate)))
        return
    for number, (expected, actual) in enumerate(zip(reference, candidate, strict=True), 1):
        label = expected.split(":", 1)[0].strip()
        if label in identity_lines:
            expected, actual = DIGEST.sub("<digest>", expected), DIGEST.sub("<digest>", actual)
        if expected != actual:
            result.differ(tutorial, "printed:rank%d:line%d" % (rank, number),
                          "master %r, here %r" % (expected, actual))


def _monitor_lines(log: Path, pattern: re.Pattern[str]) -> list[str]:
    if not log.is_file():
        return []
    return [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
            if pattern.fullmatch(line)]


def _compare_observations(result: Comparison, tutorial: str, rank: int,
                          reference: Sequence[Mapping[str, Any]],
                          candidate: Sequence[Mapping[str, Any]]) -> None:
    if len(reference) != len(candidate):
        result.differ(tutorial, "runs:rank%d" % rank, "%d pops.run calls on master, %d here"
                      % (len(reference), len(candidate)))
    for expected, actual in zip(reference, candidate, strict=False):
        where = "run%d:rank%d" % (expected["index"], rank)
        if "error" in expected or "error" in actual:
            result.differ(tutorial, where + ":observation", "master %s, here %s" % (
                expected.get("error", "observed"), actual.get("error", "observed")))
            continue
        for key in ("controls", "time", "macro_step", "n_levels", "configured_n_levels",
                    "blocks", "patch_boxes", "report"):
            if expected.get(key) != actual.get(key):
                result.differ(tutorial, where + ":" + key, "master %s, here %s" % (
                    json.dumps(expected.get(key))[:400], json.dumps(actual.get(key))[:400]))
        states = {row["name"]: row for row in actual.get("states", ())}
        names = [row["name"] for row in expected.get("states", ())]
        if sorted(names) != sorted(states):
            result.differ(tutorial, where + ":states", "master %s, here %s"
                          % (names, sorted(states)))
        for row in expected.get("states", ()):
            other = states.get(row["name"])
            if other is None:
                continue
            for key in ("dtype", "shape", "sha256"):
                if row[key] != other[key]:
                    result.differ(tutorial, where + ":" + row["name"] + ":" + key,
                                  "master %s, here %s" % (row[key], other[key]))


def _outputs(entry: Mapping[str, Any], pattern: str) -> list[str]:
    return sorted(path for path in entry["written"] if fnmatch.fnmatchcase(path, pattern))


def _compare_outputs(result: Comparison, row: Mapping[str, Any], reference_root: Path,
                     candidate_root: Path, reference: Mapping[str, Any],
                     candidate: Mapping[str, Any]) -> None:
    tutorial = row["id"]
    claimed_reference: set[str] = set()
    claimed_candidate: set[str] = set()
    for output in row.get("outputs", ()):
        pattern, kind = output["pattern"], output["kind"]
        expected = _outputs(reference, pattern)
        actual = _outputs(candidate, pattern)
        claimed_reference.update(expected)
        claimed_candidate.update(actual)
        where = "output:" + pattern
        if not expected or not actual:
            result.differ(tutorial, "output-missing:" + pattern,
                          "master wrote no file matching it" if not expected
                          else "no file matching it was written")
            continue
        ref_dir = reference_root / reference["work"] / "tutorial"
        cand_dir = candidate_root / candidate["work"] / "tutorial"
        if kind == "record":
            continue
        if kind == "paraview":
            left = paraview_series(ref_dir, expected)
            right = paraview_series(cand_dir, actual)
            for key in ("series", "recipes", "kinds"):
                if left[key] != right[key]:
                    result.differ(tutorial, where + ":" + key, _first_difference(
                        left[key], right[key]))
            continue
        if expected != actual:
            result.differ(tutorial, "output-files:" + pattern, "master wrote %s, here %s"
                          % (expected, actual))
            continue
        for relative in expected:
            if kind == "bytes":
                if reference["written"][relative]["sha256"] != candidate["written"][relative][
                        "sha256"]:
                    result.differ(tutorial, where + ":" + relative, "bytes differ")
            elif kind == "npz":
                left = npz_arrays(ref_dir / relative)
                right = npz_arrays(cand_dir / relative)
                if left != right:
                    result.differ(tutorial, where + ":" + relative, _first_difference(left, right))
    for side, written, claimed in (("master", reference["written"], claimed_reference),
                                   ("here", candidate["written"], claimed_candidate)):
        unclaimed = sorted(set(written) - claimed)
        if unclaimed:
            result.differ(tutorial, "output-unlisted",
                          "%s wrote files that the contract does not list: %s"
                          % (side, unclaimed[:5]))


def _first_difference(left: Any, right: Any, trail: str = "") -> str:
    if type(left) is not type(right):
        return "%s: master %s, here %s" % (trail or "/", type(left).__name__,
                                           type(right).__name__)
    if isinstance(left, Mapping):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                return "%s/%s: present only %s" % (trail, key, "here" if key in right
                                                   else "on master")
            if left[key] != right[key]:
                return _first_difference(left[key], right[key], "%s/%s" % (trail, key))
    if isinstance(left, list):
        if len(left) != len(right):
            return "%s: %d items on master, %d here" % (trail or "/", len(left), len(right))
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            if a != b:
                return _first_difference(a, b, "%s[%d]" % (trail, index))
    return "%s: master %s, here %s" % (trail or "/", str(left)[:200], str(right)[:200])


def compare_runs(args: argparse.Namespace) -> int:
    contract = load_contract(Path(args.contract))
    monitor = re.compile(contract["monitor"])
    reference_root = Path(args.reference).resolve()
    candidate_root = Path(args.candidate).resolve()

    def entries(root: Path) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
        rows: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for path in sorted(root.glob("report-*.json")):
            report = read_json(path)
            if report.get("schema") != SCHEMA:
                raise ContractError("%s is not an MG8 run report" % path)
            for entry in report["tutorials"]:
                if entry["id"] in rows:
                    raise ContractError("tutorial %s ran twice in %s" % (entry["id"], root))
                rows[entry["id"]] = (report, entry)
        return rows

    references = entries(reference_root)
    candidates = entries(candidate_root)
    result = Comparison()
    selected = [row for row in tutorial_rows(contract, args.tutorial)
                if not args.lane or row["lane"] == args.lane]
    if args.lane and args.lane not in contract["lanes"]:
        raise ContractError("unknown lane %s" % args.lane)
    for row in selected:
        tutorial = row["id"]
        if tutorial not in references:
            result.differ(tutorial, "reference", "master was not run")
            continue
        if tutorial not in candidates:
            result.differ(tutorial, "candidate", "the tutorial was not run")
            continue
        ref_report, reference = references[tutorial]
        cand_report, candidate = candidates[tutorial]
        if ref_report.get("revision") != contract["master"]:
            result.differ(tutorial, "reference", "the reference ran %s, not master %s"
                          % (ref_report.get("revision"), contract["master"]))
        elif ref_report.get("tree_dirty", True):
            result.differ(tutorial, "reference", "master's checkout had local changes, or its "
                          "report does not say")
        for side, entry in (("master", reference), ("here", candidate)):
            if entry["returncode"] != 0:
                result.differ(tutorial, "returncode:" + side, "exit %d%s; %s" % (
                    entry["returncode"], " (watchdog)" if entry["timeout"] else "",
                    _failure_summary(reference_root if side == "master" else candidate_root,
                                     entry)))
        if reference["returncode"] != 0 or candidate["returncode"] != 0:
            continue
        for rank_ref, rank_cand in zip(reference["ranks"], candidate["ranks"], strict=True):
            rank = rank_ref["rank"]
            left = read_json(reference_root / rank_ref["record"])
            right = read_json(candidate_root / rank_cand["record"])
            _compare_observations(result, tutorial, rank, left["observations"],
                                  right["observations"])
            ref_work = str(reference_root / reference["work"] / "tutorial")
            cand_work = str(candidate_root / candidate["work"] / "tutorial")
            _compare_printed(result, tutorial, rank, _printed_lines(left, ref_work),
                             _printed_lines(right, cand_work), row.get("identity_lines", ()))
            expected = _monitor_lines(reference_root / rank_ref["log"], monitor)
            actual = _monitor_lines(candidate_root / rank_cand["log"], monitor)
            if expected != actual:
                result.differ(tutorial, "monitor:rank%d" % rank, _first_difference(
                    expected, actual))
        _compare_outputs(result, row, reference_root, candidate_root, reference, candidate)

    ids = {row["id"] for row in selected}
    ledger = {row["tutorial"]: row for row in contract.get("divergence", ())
              if row["tutorial"] in ids}
    rows_by_id = {row["id"]: row for row in selected}
    explained, unexplained = [], []
    for difference in result.rows:
        entry = ledger.get(difference["tutorial"])
        # Exact names only: a recorded check explains that check, never a family of checks.
        matched = entry is not None and difference["check"] in entry["checks"] \
            and recordable(difference["check"], rows_by_id[difference["tutorial"]])
        (explained if matched else unexplained).append(
            {**difference, "causes": list(entry["causes"])} if matched else difference)
    # Shrink only (MG6): a recorded check that no longer differs must be removed.
    found = {(row["tutorial"], row["check"]) for row in result.rows}
    stale = [{"tutorial": tutorial, "check": check, "causes": list(entry["causes"])}
             for tutorial, entry in sorted(ledger.items()) for check in entry["checks"]
             if (tutorial, check) not in found]
    verdict = {
        "schema": SCHEMA + ".verdict", "master": contract["master"],
        "reference": str(reference_root), "candidate": str(candidate_root),
        "tutorials": [row["id"] for row in selected],
        "unexplained": unexplained, "explained": explained, "stale_divergences": stale,
        "passed": not unexplained and not stale,
    }
    if args.verdict:
        write_json(Path(args.verdict), verdict)
    causes = {row["id"]: row for row in contract.get("cause", ())}
    summary = _summary(verdict, references, candidates, selected, causes)
    if args.summary:
        Path(args.summary).write_text(summary, encoding="utf-8")
    print(summary)
    return 0 if verdict["passed"] else 1


def _failure_summary(root: Path, entry: Mapping[str, Any]) -> str:
    for rank in entry["ranks"]:
        if rank.get("record"):
            status = read_json(root / rank["record"])["status"]
            if not status["ok"]:
                return "%s: %s" % (status.get("exception", "failure"),
                                   str(status.get("message", ""))[:300])
    lines = (root / entry["launcher_log"]).read_text(encoding="utf-8", errors="replace")
    return lines.strip().splitlines()[-1][:300] if lines.strip() else "no record and no log"


def _summary(verdict: Mapping[str, Any], references: Mapping[str, Any],
             candidates: Mapping[str, Any], selected: Sequence[Mapping[str, Any]],
             causes: Mapping[str, Mapping[str, Any]]) -> str:
    out = io.StringIO()
    out.write("## MG8: master's tutorials (master %s)\n\n" % verdict["master"][:12])
    out.write("| Tutorial | master | here | unexplained | recorded | causes |\n"
              "|---|---|---|---:|---:|---|\n")
    for row in selected:
        tutorial = row["id"]
        ref = references.get(tutorial, (None, {}))[1]
        cand = candidates.get(tutorial, (None, {}))[1]
        count = sum(1 for item in verdict["unexplained"] if item["tutorial"] == tutorial)
        recorded = [item for item in verdict["explained"] if item["tutorial"] == tutorial]
        cited = sorted({cause for item in recorded for cause in item["causes"]})
        out.write("| %s `%s` | %s | %s | %d | %d | %s |\n" % (
            tutorial, row["script"], _cell(ref), _cell(cand), count, len(recorded),
            ", ".join(cited) or "-"))
    if verdict["unexplained"]:
        out.write("\n### Unexplained differences\n\n")
        for item in verdict["unexplained"]:
            out.write("- %s `%s`: %s\n" % (item["tutorial"], item["check"], item["detail"]))
    if verdict["explained"]:
        out.write("\n### Recorded divergences\n\n")
        for item in verdict["explained"]:
            out.write("- %s `%s`: %s\n" % (item["tutorial"], item["check"], item["detail"]))
        cited = sorted({cause for item in verdict["explained"] for cause in item["causes"]})
        out.write("\n### Their causes\n\n")
        for name in cited:
            cause = causes[name]
            out.write("- `%s` (%s, %s): %s\n" % (name, cause["issue"], ", ".join(cause["commits"]),
                                                 " ".join(cause["text"].split())))
    if verdict["stale_divergences"]:
        out.write("\n### Recorded divergences that no longer occur (remove them)\n\n")
        for item in verdict["stale_divergences"]:
            out.write("- %s `%s` (%s)\n" % (item["tutorial"], item["check"],
                                            ", ".join(item["causes"])))
    out.write("\nVerdict: %s\n" % ("passed" if verdict["passed"] else "failed"))
    return out.getvalue()


def _cell(entry: Mapping[str, Any]) -> str:
    if not entry:
        return "not run"
    if entry["returncode"] != 0:
        return "exit %d" % entry["returncode"]
    return "ok, %.0f s" % entry["seconds"]


# ---------------------------------------------------------------------------------------------
# Archive (MG8-0: the reference of master, kept outside the code tree)


def archive_run(args: argparse.Namespace) -> int:
    """Pack a run deterministically: reports, records, logs and every written file."""
    import tarfile

    run = Path(args.run).resolve()
    members: list[Path] = []
    manifest = ["tutorial\tlane\treturncode\tseconds\tkind\tname\tdtype\tshape\tsha256"]
    for report_path in sorted(run.glob("report-*.json")):
        report = read_json(report_path)
        members.append(report_path)
        for entry in report["tutorials"]:
            prefix = [entry["id"], report["lane"], str(entry["returncode"]), str(entry["seconds"])]
            work = run / entry["work"]
            members.append(work / "launcher.log")
            members.extend(sorted(path for path in (work / "record").rglob("*") if path.is_file()))
            members.extend(work / "tutorial" / relative for relative in sorted(entry["written"]))
            for rank in entry["ranks"]:
                if not rank.get("record"):
                    continue
                record = read_json(run / rank["record"])
                for observation in record["observations"]:
                    for state in observation.get("states", ()):
                        manifest.append("\t".join(prefix + [
                            "state:rank%d" % rank["rank"], state["name"], state["dtype"],
                            "x".join(str(n) for n in state["shape"]), state["sha256"]]))
            for relative, meta in sorted(entry["written"].items()):
                manifest.append("\t".join(prefix + ["output", relative, "", str(meta["size"]),
                                                    meta["sha256"]]))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_name(output.name.removesuffix(".tar.xz") + ".manifest.tsv")
    manifest_path.write_text("\n".join(manifest) + "\n", encoding="utf-8")

    def normalized(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        info.mode = 0o644
        return info

    with tarfile.open(output, "w:xz", format=tarfile.PAX_FORMAT) as tar:
        for path in members:
            tar.add(path, arcname=path.relative_to(run).as_posix(), recursive=False,
                    filter=normalized)
    sums = output.with_name(output.name.removesuffix(".tar.xz") + ".files.sha256")
    sums.write_text("".join("%s  %s\n" % (sha256_file(path), path.relative_to(run).as_posix())
                            for path in members), encoding="utf-8")
    print(json.dumps({"archive": str(output), "sha256": sha256_file(output),
                      "members": len(members), "manifest": str(manifest_path)}))
    return 0


# ---------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the tutorials of one lane of a source tree")
    run.add_argument("--source", required=True, help="checkout whose tutorials run")
    run.add_argument("--lane", required=True, help="lane of the contract")
    run.add_argument("--python", required=True, help="interpreter of the installed PoPS")
    run.add_argument("--output", required=True)
    run.add_argument("--phase", choices=("reference", "candidate"), required=True)
    run.add_argument("--tutorial", action="append", default=[], help="two-digit id")
    run.add_argument("--exclude", action="append", default=[],
                     help="two-digit id that another run of the lane runs (compare fails on a "
                          "tutorial that no run ran)")
    run.add_argument("--mpiexec")
    run.add_argument("--mpiexec-arg", action="append", default=[])
    run.add_argument("--pythonpath", action="append", default=[],
                     help="build-tree package directory, for an uninstalled PoPS")
    run.add_argument("--include", help="POPS_INCLUDE for an uninstalled PoPS")
    run.add_argument("--kokkos-root")
    run.add_argument("--watchdog", type=int, default=3600, help="seconds per tutorial")
    run.add_argument("--keep-caches", action="store_true")
    run.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    probe_parser = commands.add_parser("probe", help=argparse.SUPPRESS)
    probe_parser.add_argument("--tutorial", required=True)
    probe_parser.add_argument("--record", required=True)
    compare = commands.add_parser("compare", help="compare a candidate run with master's")
    compare.add_argument("--reference", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--tutorial", action="append", default=[])
    compare.add_argument("--lane", help="compare only the tutorials of this lane")
    compare.add_argument("--verdict")
    compare.add_argument("--summary")
    compare.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    archive = commands.add_parser("archive", help="pack a run outside the code tree")
    archive.add_argument("--run", required=True)
    archive.add_argument("--output", required=True, help="path of the .tar.xz to write")
    check = commands.add_parser("check", help="validate the contract against a tree")
    check.add_argument("--source", default=str(ROOT))
    check.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    check.add_argument("--check-base", metavar="REVISION",
                       help="fail if the ledger records more than the contract of REVISION")
    check.add_argument("--verify-commits", action="store_true",
                       help="fail unless every commit of a cause is in the history of HEAD "
                            "and changes code that the tutorials execute")
    args = parser.parse_args(argv)
    if args.command == "probe":
        return probe(args)
    if args.command == "run":
        return run_tutorials(args)
    if args.command == "compare":
        return compare_runs(args)
    if args.command == "archive":
        return archive_run(args)
    source = Path(args.source)
    contract = load_contract(Path(args.contract))
    problems = check_contract(contract, source)
    if args.check_base:
        base = contract_at(source, args.check_base, Path(args.contract))
        if base is None:
            print("the base %s has no MG8 contract: first introduction" % args.check_base)
        else:
            problems += ["against %s: %s" % (args.check_base, problem)
                         for problem in ledger_growth(base, contract)]
    if args.verify_commits:
        problems += verify_commits(contract, source)
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
