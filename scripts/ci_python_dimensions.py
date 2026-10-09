"""Run selected Python files in processes with their declared native specialization.

The default CI contract is Dim2. Exceptions are explicit data, never inferred from
source text or whichever native artifact happens to be present. No file is excluded.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "tests/python/native_dimensions.json"


def partition(paths: list[str], contract: dict) -> dict[int, list[str]]:
    default, overrides = contract["default"], contract["files"]
    if any(type(dim) is not int or dim not in (1, 2, 3)
           for dim in (default, *overrides.values())):
        raise ValueError("native dimensions must be explicit integers 1, 2 or 3")
    if len(paths) != len(set(paths)):
        raise ValueError("selected Python files must not be duplicated")
    groups: dict[int, list[str]] = {}
    for path in paths:
        groups.setdefault(overrides.get(path, default), []).append(path)
    return dict(sorted(groups.items()))


def run_groups(groups: dict[int, list[str]], packages: Path, timings: Path,
               *, install_root: Path | None = None, backend: str = "serial") -> int:
    """Install every declared native group from its retained wheel; never inject a build tree."""
    status = 0
    if install_root is None:
        runner_temp = os.environ.get("RUNNER_TEMP")
        if not runner_temp:
            raise ValueError("installed CI execution requires explicit RUNNER_TEMP")
        install_root = Path(runner_temp) / "pops-installed-dimensions"
    install_root = install_root.resolve()
    if install_root.is_relative_to(ROOT.resolve()):
        raise ValueError("CI installation prefix must be outside checkout")
    if backend not in {"serial", "mpi"}:
        raise ValueError("explicit native backend must be serial or mpi")
    for dimension, paths in groups.items():
        package = (packages / f"dim{dimension}").resolve()
        wheels = sorted(package.glob("pops-*.whl"))
        if len(wheels) != 1:
            raise FileNotFoundError(f"Dim{dimension} requires exactly one retained wheel: {package}")
        receipts = (timings / f"dim{dimension}").resolve()
        receipts.mkdir(parents=True, exist_ok=True)
        (receipts / "selected.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
        prefix = install_root / f"dim{dimension}"
        if prefix.exists():
            raise FileExistsError(f"refuse reused CI environment: {prefix}")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        prefix.parent.chmod(0o700)
        wheel_dir = install_root / f"wheel-dim{dimension}"
        wheel_dir.mkdir(mode=0o700)
        retained = wheel_dir / wheels[0].name
        import shutil
        shutil.copyfile(wheels[0], retained)
        environment = os.environ.copy()
        for key in ("PYTHONPATH", "PYTHONOPTIMIZE", "POPS_NATIVE_VARIANTS_ROOT", "POPS_INCLUDE", "POPS_CI_NATIVE_PACKAGE", "PYTEST_ADDOPTS", "PIP_PREFIX", "PIP_TARGET", "PIP_USER"):
            environment.pop(key, None)
        environment.update(POPS_NATIVE_DIM=str(dimension), PYTHONNOUSERSITE="1",
                           POPS_CI_PYTEST_TIMINGS_DIR=str(receipts),
                           POPS_CI_TEST_EXECUTION="installed")
        python = prefix / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        stages = [
            [sys.executable, "-m", "venv", "--system-site-packages", str(prefix)],
            [str(python), "-m", "pip", "--isolated", "install", "--ignore-installed", "--no-deps", "--prefix", str(prefix), str(retained)],
            [str(python), str(ROOT / "scripts/prove_installed_wheel.py"),
             "--wheel", str(retained), "--expect-dim", str(dimension)],
            [str(python), str(ROOT / "scripts/verify_installed_native.py"),
             "--expect-dim", str(dimension), "--expect-" + backend],
        ]
        stage_receipts = []
        verified = True
        for command in stages:
            if any("prove_installed_wheel.py" in str(arg) for arg in command):
                with (receipts / "installed-wheel-proof.json").open("w") as proof:
                    result = subprocess.run(command, cwd=ROOT, env=environment, check=False, stdout=proof)
            else:
                result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
            stage_receipts.append({"argv": command, "returncode": result.returncode})
            if result.returncode:
                status = status or result.returncode
                verified = False
                break
        (receipts / "installation-stages.json").write_text(json.dumps(stage_receipts, indent=2) + "\n")
        if not verified:
            continue
        bootstrap = (
            "import sys; from pathlib import Path; "
            f"sys.path[:0]=[{str(ROOT / 'scripts')!r},{str(ROOT)!r}]; "
            "import pops; "
            "assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()); "
            f"assert not Path(pops.__file__).resolve().is_relative_to(Path({str(ROOT)!r})); "
            "import pytest; raise SystemExit(pytest.main(sys.argv[1:]))"
        )
        result = subprocess.run(
            [str(python), "-u", "-c", bootstrap, "-v", "-ra", "--durations=20",
             "-o", "pythonpath=", "-o", f"cache_dir={receipts / 'pytest-cache'}",
             f"--basetemp={receipts / 'pytest-tmp'}", "-p", "ci_pytest_timings",
             f"--junitxml={receipts / 'junit.xml'}", *paths],
            cwd=ROOT, env=environment, check=False)
        status = status or result.returncode
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-file", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--packages-root", type=Path)
    parser.add_argument("--timings-dir", type=Path)
    parser.add_argument("--install-root", type=Path)
    parser.add_argument("--native-backend", choices=("serial", "mpi"), default="serial")
    args = parser.parse_args()
    paths = [line.strip() for line in args.selected_file.read_text().splitlines() if line.strip()]
    groups = partition(paths, json.loads(CONTRACT.read_text()))
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as output:
            for dimension in (1, 2, 3):
                output.write(f"dim{dimension}_count={len(groups.get(dimension, []))}\n")
        return 0
    if args.packages_root is None or args.timings_dir is None:
        parser.error("execution requires --packages-root and --timings-dir")
    return run_groups(groups, args.packages_root, args.timings_dir, install_root=args.install_root, backend=args.native_backend)


if __name__ == "__main__":
    raise SystemExit(main())
