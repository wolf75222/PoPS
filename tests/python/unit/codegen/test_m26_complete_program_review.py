"""Compile full M26 Program translation units, without linking or JIT."""
from pathlib import Path
import re

import numpy as np
import pytest

from tests.python.unit.codegen import test_m23_complete_program_syntax as compiler


WORKER = r'''
import pathlib, sys
root, output = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2])
sys.path.insert(0, str(root / "python"))
sys.path.insert(0, str(root / "examples/migration/scientific"))
import pops
assert pathlib.Path(pops.__file__).resolve() == root / "python/pops/__init__.py"
import api040_m26_finite_interaction as witness
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
if bool(int(sys.argv[5])):
    from pops.linalg import FiniteMeasure
    # Probe the same complete Program lowering with a nonuniform measure.
    # Binary-exact weights remain tied to physical DOF labels under permutation.
    witness.FiniteMeasure = lambda support, unused: FiniteMeasure(
        support, tuple((int(label[1:])+1)/16 for label in support.dofs))
case, layout, subjects, order = witness.build_case(permuted=bool(int(sys.argv[4])))
resolved = pops.resolve(pops.validate(case), layout=layout)
graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
source = emit_cpp_program(resolved.time, model_graph=graph)
assert source.count("pops::detail::finite_linear_apply<12, 12>") == 2
assert "ctx.commit_many(" in source
output.write_text(source)
print("source_import="+pops.__file__)
'''


@pytest.mark.parametrize("permuted", (False, True))
@pytest.mark.parametrize("nonuniform", (False, True))
def test_m26_full_program_syntax(tmp_path, monkeypatch, permuted, nonuniform):
    monkeypatch.setattr(compiler, "EMIT_WORKER", WORKER)
    monkeypatch.setenv("POPS_M23_SOURCE_ROOT", str(Path(__file__).resolve().parents[4]))
    compiler.test_m23_complete_ssprk2_program_syntax(tmp_path, 12, permuted, nonuniform)
    source = (tmp_path / "complete_m23_program.cpp").read_text()
    # Inspect the actual emitted arrays, rather than rebuilding the interaction
    # through the API under review. Each Fourier basis column independently
    # gives the action of this rank-two periodic kernel on a unit vector.
    matrices = re.findall(
        r"finite_linear_apply<12, 12>\(Kokkos::Array<pops::Real, 144>\{([^}]+)\}, "
        r"Kokkos::Array<pops::Real, 12>\{([^}]+)\}",
        source,
    )
    assert len(matrices) == 2
    order = np.array((6, 0, 9, 3, 11, 2, 8, 5, 1, 10, 4, 7)
                     if permuted else tuple(range(12)))
    theta = 2 * np.pi * (np.arange(12) + .5) / 12
    weights = (np.arange(1, 13)/16 if nonuniform else np.full(12, 1/12))
    basis_action = (np.outer(np.cos(theta), np.cos(theta))
                    + np.outer(np.sin(theta), np.sin(theta))) * weights[None, :]
    for text, inputs in matrices:
        assert inputs.split(", ") == ["cse%d_" % i for i in range(12)]
        coefficients = np.array([float(value) for value in re.findall(
            r"pops::Real\(([^)]+)\)", text)])
        assert coefficients.shape == (144,)
        matrix = coefficients.reshape(12, 12)
        np.testing.assert_allclose(matrix, basis_action[np.ix_(order, order)],
                                   rtol=0, atol=1.3e-15)
        inverse = np.argsort(order)
        np.testing.assert_allclose(matrix[np.ix_(inverse, inverse)], basis_action,
                                   rtol=0, atol=1.3e-15)
        measured_matrix = weights[order, None] * matrix
        # W is certified exactly symmetric. Multiplying its rounded native
        # coefficients by the second measure weight adds floating roundoff.
        np.testing.assert_allclose(measured_matrix, measured_matrix.T,
                                   rtol=0, atol=1e-16)
        if nonuniform:
            assert np.max(np.abs(matrix - matrix.T)) > .1
    # Twelve component loads are made at the same spatial index in each
    # kernel. This is a finite batch at one native point, never a mesh gather.
    loads = re.findall(
        r"((?:        const pops::Real cse\d+_ = u\d+A\(index, \d+\);\n)+)"
        r"        const auto cse\d+_ = pops::detail::finite_linear_apply", source)
    assert len(loads) == 2
    for group in loads:
        reads = re.findall(r"= (u\d+A)\(index, (\d+)\);", group)
        assert [component for field, component in reads] == [str(i) for i in range(12)]
        assert len({field for field, component in reads}) == 1
    # Check the generated acceptance boundary only. Scratch outputs may be
    # written, but every finite diagnostic guard precedes accepted-state commit.
    assert source.count("non-finite scientific expression input or intermediate") == 3
    assert source.rfind("throw pops::runtime::program::StepAttemptRejected") < source.index(
        "ctx.commit_many(")
