"""Real native execution proof for the resolved ``MagnitudeAbove`` AMR leaf.

The negative seed distinguishes magnitude tagging from ordinary ``Above``:
``n > threshold`` matches nothing, while ``abs(n) > threshold`` must build a
fine level through the same prepared Kokkos tagging program used by bound AMR
simulations.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pops
import pytest
from pops.mesh._amr import (
    Above,
    ConflictPolicy,
    EqualityPolicy,
    Hysteresis,
    MagnitudeAbove,
    TaggingGraph,
)
from pops.params import RuntimeParam
from pops.runtime import _engine_descriptors as engine
from pops.runtime._engine_descriptors import Periodic
from pops.runtime._runtime_mesh_lowering import flow_bootstrap_tagging
from pops.runtime._system import AmrSystem


N = 16
THRESHOLD = 0.5

pytestmark = [
    pytest.mark.kokkos,
    pytest.mark.regression,
]


def _resolved_leaf(node_type):
    model = pops.Model("native-magnitude-tagging-model")
    state = model.state("U", components=("n",))
    case = pops.Case("native-magnitude-tagging-case")
    block = case.block("tracer", model, states=(state,))
    threshold = case.param(RuntimeParam("threshold", default=THRESHOLD))
    validated = pops.validate(case)
    indicator = validated.resolve(block[state])
    bound_threshold = validated.resolve(threshold)
    graph = TaggingGraph(
        refine=node_type(indicator, bound_threshold),
        coarsen=None,
        hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
        conflict_policy=ConflictPolicy.REFINE_WINS,
    ).resolve()
    return graph, bound_threshold, indicator.qualified_id


def _install_state_transfer_routes(simulation, subject):
    routes = (
        ("prolongation", "conservative_linear", 2, [1]),
        ("restriction", "volume_average", 1, [0]),
        ("coarse_fine_fill", "conservative_coarse_fine", 2, [1]),
    )
    for operation, kernel, order, ghost_depth in routes:
        simulation._s._register_bootstrap_transfer_route(
            "test::native-magnitude::%s" % operation,
            [subject],
            "test::native-magnitude-transfer",
            "cell",
            "cell",
            "conservative",
            "dense",
            operation,
            kernel,
            order,
            ghost_depth * 2,
            [2, 2],
        )
    simulation._s._bind_bootstrap_block_subject(subject, "tracer")


def _native_hierarchy(node_type):
    graph, threshold, subject = _resolved_leaf(node_type)
    simulation = AmrSystem(
        n=N,
        L=1.0,
        periodicity=(True, True),
        regrid_every=0,
        explicit_bootstrap=True,
    )
    # This direct-runtime fixture still consumes the resolved Case Handle. Install that exact
    # owner-qualified identity before declaring the native block, just as pops.bind does.
    simulation._s._install_block_state_route("tracer", subject)
    simulation.set_temporal_relations([2], [1], ["integral_only"])
    simulation.add_equation(
        "tracer",
        engine.Model(
            engine.Scalar(),
            engine.ExB(),
            engine.NoSource(),
            engine.BackgroundDensity(alpha=0.0, n0=0.0),
        ),
        spatial=engine.Spatial(),
        time=engine.Explicit(),
    )
    simulation.set_poisson(bc=Periodic())
    flow_bootstrap_tagging(
        simulation,
        SimpleNamespace(tagging=graph),
        {threshold: THRESHOLD},
        clock_identity="case::native-magnitude-tagging-clock",
    )
    values = np.zeros((N, N), dtype=np.float64)
    values[N // 2 - 2:N // 2 + 2, N // 2 - 2:N // 2 + 2] = -1.0
    simulation.set_density("tracer", values)
    _install_state_transfer_routes(simulation, subject)
    simulation._s._begin_bootstrap_plan()
    created = bool(simulation._s._bootstrap_next_level())
    simulation._s._commit_bootstrap_level()
    return simulation, created


def test_magnitude_above_lowers_and_executes_on_the_native_amr_tagger():
    magnitude, magnitude_created = _native_hierarchy(MagnitudeAbove)
    ordinary, ordinary_created = _native_hierarchy(Above)

    assert magnitude_created is True
    assert magnitude.n_levels() == 2
    assert magnitude.n_patches() > 0
    assert magnitude.patch_boxes()
    assert ordinary_created is False
    assert ordinary.n_levels() == 1
    assert ordinary.n_patches() == 0
    assert ordinary.patch_boxes() == []
