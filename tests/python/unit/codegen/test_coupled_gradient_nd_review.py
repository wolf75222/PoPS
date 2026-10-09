"""Independent full-Program syntax and unsupported-route probes; source only."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest


WORKER = r'''
import pathlib, sys
root, output = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2])
sys.path.insert(0, str(root / "python"))
import pops
assert pathlib.Path(pops.__file__).resolve() == root / "python/pops/__init__.py"
from pops import math
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.domain import CartesianDomain
from pops.frames import Cartesian2D, Cartesian3D
from pops.layouts import Uniform
from pops.lib.time import SSPRK2
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import CoupledGradient, DiscretizationPlan
from pops.physics.diffusion import DiffusiveBoundary
from pops.time import FixedDt
dim, permuted = int(sys.argv[3]), int(sys.argv[4])
order = (2, 0, 1) if permuted else (0, 1, 2)
frame = CartesianDomain("anisotropic_geometry", lower=(0.,)*dim,
    upper=(1.1, 2.3, 3.7)[:dim]).frame((Cartesian2D, Cartesian3D)[dim-2]())
model = pops.Model("joint_components", frame=frame)
state = model.state("field", components=tuple(("first", "second", "third")[i] for i in order))
v = (1., 2., -1.)
d = tuple(tuple(a*b for b in v) for a in v)
r = ((0., -.3, .2), (.3, 0., -.1), (-.2, .1, 0.))
permute = lambda a: tuple(tuple(a[i][j] for j in order) for i in order)
flux = model.coupled_gradient_flux("law", state=state,
    dissipative=permute(d), reversible=permute(r))
method = CoupledGradient(flux=flux)
bounded_flux = model.coupled_gradient_flux("bounded_law", state=state,
    dissipative=permute(d), reversible=permute(r), boundaries={component: tuple(
        DiffusiveBoundary(axis, side, "conormal" if axis == dim-1 else "periodic")
        for axis in range(dim) for side in ("lower", "upper")) for component in state.components})
try:
    CoupledGradient(flux=bounded_flux)
except ValueError as error:
    assert "periodic" in str(error), str(error)
else:
    raise AssertionError("unsupported physical boundary silently admitted")
rate = model.rate("balance", equation=math.ddt(state) == .5*math.div(flux)+1.5*math.div(flux))
case = pops.Case("nd_review")
block = case.block("fields", model, states=(state,))
numerics = DiscretizationPlan()
numerics.rates.add(rate, method)
case.numerics(numerics, block=block)
program = SSPRK2(block[state], rate=rate)
program.step_strategy(FixedDt(1.e-5))
case.program(program)
layout = Uniform(CartesianGrid(frame=frame, cells=(6, 8, 10)[:dim], periodic=PeriodicAxes(frame.axes)))
resolved = pops.resolve(pops.validate(case), layout=layout)
graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
source = emit_cpp_program(resolved.time, model_graph=graph)
assert source.count(".stage_accepted_exchanges(") == 4
assert "PreparedCoupledGradient<pops::kNativeDimension, 3>" in source
output.write_text(source)
# Asking the real emitter for an AMR route must not fall back to Uniform.
try:
    emit_cpp_program(resolved.time, model_graph=graph, target="amr_system")
except NotImplementedError as error:
    assert "AMR" in str(error) or "Uniform" in str(error), str(error)
else:
    raise AssertionError("unsupported AMR coupled gradient silently admitted")
print("source_import=" + pops.__file__)
'''


@pytest.mark.parametrize("dimension,permuted", [(2, False), (3, True)])
def test_nd_complete_ssprk2_program_and_amr_refusal(tmp_path, dimension, permuted):
    root = Path(os.environ.get("POPS_ND_REVIEW_SOURCE_ROOT", Path(__file__).resolve().parents[4])).resolve()
    compiler = shutil.which("clang++")
    include = Path(sys.prefix) / "include"
    if compiler is None or not all((include / name).is_file() for name in ("Kokkos_Core.hpp", "mpi.h")):
        pytest.skip("real compiler, Kokkos and MPI headers required")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["POPS_NATIVE_DIM"] = str(dimension)
    source = tmp_path / "complete_program.cpp"
    emission = subprocess.run([sys.executable, "-I", "-c", WORKER, str(root), str(source),
        str(dimension), str(int(permuted))], capture_output=True, text=True, env=env, timeout=90)
    assert emission.returncode == 0, emission.stdout + emission.stderr
    flags = ["-std=c++20", "-fsyntax-only", "-fno-fast-math", f"-DPOPS_NATIVE_DIM={dimension}",
        "-DPOPS_RUNTIME_SHARED_EXCEPTION_ABI", "-DPOPS_HAS_KOKKOS", "-DKOKKOS_DEPENDENCE",
        "-DPOPS_HAS_MPI", "-I"+str(root / "include"), "-I"+str(include)]
    if re.search(r"^#define KOKKOS_ENABLE_OPENMP\b", (include / "KokkosCore_config.h").read_text(), re.MULTILINE):
        if sys.platform == "darwin":
            flags += ["-Xpreprocessor", "-fopenmp", "-I/opt/homebrew/opt/libomp/include"]
        else:
            flags += ["-fopenmp"]
    command = [compiler, *flags, str(source)]
    (tmp_path / "compiler-command.txt").write_text("\n".join(command))
    compiled = subprocess.run(command, capture_output=True, text=True, timeout=90)
    (tmp_path / "compiler.stderr").write_text(compiled.stderr)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr


def test_sdk_signature_changes_artifact_spec_despite_unchanged_method_schema(tmp_path):
    root = Path(os.environ.get("POPS_ND_REVIEW_SOURCE_ROOT", Path(__file__).resolve().parents[4])).resolve()
    worker = r'''
import pathlib, sys, shutil, subprocess
root, tmp = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2])
sys.path.insert(0, str(root / "python"))
import pops
assert pathlib.Path(pops.__file__).resolve() == root / "python/pops/__init__.py"
from pops.codegen.toolchain import pops_header_signature
from pops.identity import artifact_spec_identity
from pops.identity.semantic import semantic_identity
from pops.frames import Cartesian1D
from pops.numerics import CoupledGradient
relative = "pops/numerics/diffusion/prepared_diffusion.hpp"
old_header = subprocess.check_output(["git", "show", "9ea0342:include/"+relative], cwd=root)
assert old_header != (root / "include" / relative).read_bytes()
shutil.copytree(root / "include", tmp / "old-sdk")
(tmp / "old-sdk" / relative).write_bytes(old_header)
old_sig = pops_header_signature(tmp / "old-sdk")
new_sig = pops_header_signature(root / "include")
assert old_sig != new_sig  # Only this manifested header changed in the copy.
model = pops.Model("cache_review", frame=Cartesian1D())
state = model.state("u", components=("a", "b"))
flux = model.coupled_gradient_flux("law", state=state,
    dissipative=((0., 0.), (0., 0.)), reversible=((0., -.3), (.3, 0.)))
method = CoupledGradient(flux=flux)
_ = model.module
# Hold physical semantics and method metadata fixed; only the authentic SDK
# digest varies. This is an artifact-identity seam test, not compilation.
assert method.validate()
data = {"law": flux.law.to_data(), "method": "coupled_gradient",
    "schema_version": 1, "spatial_realization": "periodic_two_point_component_matrix_v1"}
semantic = semantic_identity(data)
def spec(signature):
    return artifact_spec_identity(semantic, target="system", backend="production",
        precision="double", abi=signature+"|clang++|c++20|dim=1", toolchain="clang++|c++20",
        routes={}, components={}, flags=[], libraries=[])
assert spec(old_sig) != spec(new_sig)
print("old_header_signature="+old_sig)
print("new_header_signature="+new_sig)
'''
    result = subprocess.run([sys.executable, "-I", "-c", worker, str(root), str(tmp_path)],
                            capture_output=True, text=True, timeout=90)
    (tmp_path / "signature-proof.txt").write_text(result.stdout+result.stderr)
    assert result.returncode == 0, result.stdout+result.stderr
