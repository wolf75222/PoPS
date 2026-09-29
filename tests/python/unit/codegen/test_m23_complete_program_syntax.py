"""Compile the entire public SSPRK2 Program against its own source SDK.

POPS_M23_SOURCE_ROOT selects an isolated source checkout, never an installed
qualification. The emission worker uses -I and authenticates its pops import.
No JIT, linking, simulation, or package installation occurs here.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


EMIT_WORKER = r'''
import pathlib
import sys
root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "python"))
import pops
assert pathlib.Path(pops.__file__).resolve() == root / "python/pops/__init__.py"
from pops import math
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D, Cartesian2D, Cartesian3D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.lib.time import SSPRK2
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import CoupledGradient, DiscretizationPlan
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt

n, reverse, repeated, dimension = (int(arg) for arg in sys.argv[3:7])
order = tuple(reversed(range(n))) if reverse else tuple(range(n))
frame = CartesianDomain("periodic", lower=(0.,)*dimension,
    upper=(6.283185307179586,)*dimension).frame(
    (Cartesian1D, Cartesian2D, Cartesian3D)[dimension-1]())
model = pops.Model("independent_coupled_gradient", frame=frame)
labels = ("north", "east", "third")[:n]
state = model.state("field", components=tuple(labels[i] for i in order))
if n == 2:
    d = ((0., 0.), (0., 0.))
    r = ((0., -.3), (.3, 0.))
else:
    # Exact rank-one PSD, including its nullspace; no SPD substitution.
    v = (1., 2., -1.)
    d = tuple(tuple(a*b for b in v) for a in v)
    r = ((0., -.3, .2), (.3, 0., -.1), (-.2, .1, 0.))
permute = lambda a: tuple(tuple(a[i][j] for j in order) for i in order)
flux = model.coupled_gradient_flux("constitutive", state=state,
    dissipative=permute(d), reversible=permute(r))
rhs = math.div(flux)
if repeated:
    rhs += 2 * math.div(flux)
rate = model.rate("evolution", equation=math.ddt(state) == rhs)
case = pops.Case("independent_m23_syntax")
block = case.block("physics", model, states=(state,))
numerics = DiscretizationPlan()
numerics.rates.add(rate, CoupledGradient(flux=flux))
case.numerics(numerics, block=block)
program = SSPRK2(block[state], rate=rate)
program.step_strategy(FixedDt(.00001))
case.program(program)
case.initials.add(InitialCondition(state=block[state], value=BindArray(),
    projection=ConservativeCellAverage()))
layout = Uniform(CartesianGrid(frame=frame, cells=(32,)*dimension,
    periodic=PeriodicAxes(frame.axes)))
resolved = pops.resolve(pops.validate(case), layout=layout)
graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
source = emit_cpp_program(resolved.time, model_graph=graph)
assert "PreparedCoupledGradient<pops::kNativeDimension, %d>" % n in source
assert "DiffusiveLawResult<pops::kNativeDimension, %d, true>" % n in source
# Both SSPRK2 evaluations must be fully assembled, including accepted ledgers.
assert source.count(".stage_accepted_exchanges(") == (4 if repeated else 2)
if repeated:
    assert source.count("/occurrence:1") == 2
pathlib.Path(sys.argv[2]).write_text(source)
print("source_import=" + pops.__file__)
'''


@pytest.mark.parametrize("dimension,components,reverse,repeated", [
    (1, 2, False, False), (1, 2, True, False), (1, 3, False, False),
    (1, 3, True, False), (1, 3, True, True),
    (2, 2, False, True), (2, 3, True, False), (3, 2, False, False),
    (3, 3, True, True),
])
def test_m23_complete_ssprk2_program_syntax(tmp_path, dimension, components, reverse, repeated):
    root = Path(os.environ.get("POPS_M23_SOURCE_ROOT", Path(__file__).resolve().parents[4])).resolve()
    compiler = shutil.which("clang++")
    kokkos = Path(sys.prefix) / "include"
    if compiler is None or not (kokkos / "Kokkos_Core.hpp").is_file():
        pytest.skip("real compiler and Kokkos SDK required")
    if not (kokkos / "mpi.h").is_file():
        pytest.skip("real MPI SDK required for this MPI-branch syntax test")
    source = tmp_path / "complete_m23_program.cpp"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["POPS_NATIVE_DIM"] = str(dimension)
    emitted = subprocess.run(
        [sys.executable, "-I", "-c", EMIT_WORKER, str(root), str(source),
         str(components), str(int(reverse)), str(int(repeated)), str(dimension)],
        env=env, capture_output=True, text=True, timeout=90,
    )
    assert emitted.returncode == 0, emitted.stdout + emitted.stderr
    flags = ["-std=c++20", "-fsyntax-only", "-fno-fast-math",
             "-DPOPS_NATIVE_DIM=" + str(dimension),
             "-DPOPS_RUNTIME_SHARED_EXCEPTION_ABI", "-DPOPS_HAS_KOKKOS",
             "-DKOKKOS_DEPENDENCE", "-DPOPS_HAS_MPI",
             "-I" + str(root / "include"), "-I" + str(kokkos)]
    config = (kokkos / "KokkosCore_config.h").read_text()
    if re.search(r"^#define KOKKOS_ENABLE_OPENMP\b", config, re.MULTILINE):
        if sys.platform == "darwin":
            omp = Path("/opt/homebrew/opt/libomp/include")
            if not (omp / "omp.h").is_file():
                pytest.skip("installed Kokkos requires real OpenMP headers")
            flags += ["-Xpreprocessor", "-fopenmp", "-I" + str(omp)]
        else:
            flags += ["-fopenmp"]
    command = [compiler, *flags, str(source)]
    (tmp_path / "compiler-command.txt").write_text("\n".join(command))
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    (tmp_path / "compiler.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
