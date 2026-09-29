"""Prepared native invalid injection: a rebound negative rate must roll back."""
from pathlib import Path
import sys

import numpy as np
import pytest

import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes


EXAMPLES = Path(__file__).resolve().parents[4] / "examples/migration/scientific"
sys.path.insert(0, str(EXAMPLES))
from api040_m13_reaction_chain import build_case, cell_mean_initial


@pytest.mark.compiler
@pytest.mark.native_loader
def test_negative_rebound_rate_rejected_before_native_publication(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    frame = Rectangle("M13_negative_injection", lower=(0., 0.),
                      upper=(1., 2.)).frame(Cartesian2D())
    case, declarations = build_case(frame, name="M13_negative_rebind")
    validated = pops.validate(case)
    parameters = tuple(validated.resolve(item) for item in declarations)
    artifact = pops.compile(pops.resolve(validated, layout=Uniform(
        CartesianGrid(frame=frame, cells=(4, 4),
                      periodic=PeriodicAxes(frame.axes)))))
    initial = cell_mean_initial((4, 4), ("A", "B", "C"))
    context = pops.ExecutionContext.mpi_world(artifact)
    simulation = pops.bind(artifact, initial_state={"reactor": initial.copy()},
                           params={parameters[0]: -10., parameters[1]: .4},
                           resources={"execution_context": context})
    before = np.asarray(simulation.state_global("reactor")).copy()
    with pytest.raises(Exception):
        pops.run(simulation, t_end=.002, max_steps=1)
    after = np.asarray(simulation.state_global("reactor"))
    np.testing.assert_array_equal(after, before)
    assert simulation.time() == 0.
