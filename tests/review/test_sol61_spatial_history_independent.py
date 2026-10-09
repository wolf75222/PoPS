"""Independent SOURCE_ONLY history contract and real-header host checks."""

from pathlib import Path
import os
import subprocess
import sys

import pytest
import pops
from pops.fields import (
    SpatialInteractionKernel,
    CellVolumeMeasure,
    CellMidpoint,
    DirectSpatialInteraction,
)
from pops.model.spaces import FieldSpace
from pops.time._program.spatial_interaction import interaction_contract
from pops.codegen.program_emit_spatial_interaction import emit_spatial_interaction

ROOT = Path(__file__).resolve().parents[2]


def setup():
    model = pops.Model("independent_thermal")
    T = model.state("T", components=("temperature",))
    case = pops.Case("independent_history")
    block = case.block("thermal", model, states=(T,))
    p = pops.Program("independent_history_program")
    u = p.state(block[T])
    p.keep_history(u, depth=2)
    field = p.spatial_interaction(
        u.prev(2),
        SpatialInteractionKernel(1, lambda x, y: x[0] - 2 * y[0]),
        output_space=FieldSpace("I", components=("interaction",), sampling="cell_center"),
        measure=CellVolumeMeasure(),
        quadrature=CellMidpoint(),
        realization=DirectSpatialInteraction(65536),
    )
    return p, u, field


@pytest.mark.parametrize("target", ["system", "amr_system"])
def test_exact_two_lag_source_and_conditional_IR18(target):
    p, u, f = setup()
    assert p._serialize()["version"] == 18
    seed = f.attrs["history_source"]
    assert seed["lag"] == 2 and seed["seed_id"] == u.n.id and seed["seed_point"] == u.n.point
    assert seed["state"] is u.state and seed["cold_start"] == "copy_current"
    lines = []
    emit_spatial_interaction(
        f, {f.inputs[0].id: "retained_T"}, lines, block_indices={u.block: 0}, target=target
    )
    assert "ctx.spatial_interaction_history" in "\n".join(lines)
    assert "retained_T" in "\n".join(lines)


@pytest.mark.parametrize(
    "mutation",
    [
        "derived_T",
        "Q_closure",
        "relabel",
        "wrapped_relabel",
        "foreign_state",
        "seed_bool",
        "lag_bool",
    ],
)
def test_original_seed_cannot_be_requalified_as_storage_Q(mutation):
    p, u, f = setup()
    store = p._time_history_stores[u]
    if mutation in ("derived_T", "Q_closure", "relabel", "wrapped_relabel"):
        if mutation == "derived_T":
            seed = p.value("temperature_observation", 2 * u.n, at=u.n.point)
        elif mutation == "Q_closure":
            seed = p.value("conserved_Q", u.n + u.n * u.n, at=u.n.point)
        else:
            seed = p._replace_value(u.n, point=u.next.point)
            if mutation == "wrapped_relabel":
                seed = p.value("wrapped", 2 * seed, at=u.n.point)
        changed = p.store_history(store.attrs["history"], seed, depth=2)
        p._time_history_stores[u] = changed
    else:
        attrs = dict(f.attrs)
        source = dict(attrs["history_source"])
        source[{"foreign_state": "state", "seed_bool": "seed_id", "lag_bool": "lag"}[mutation]] = (
            None if mutation == "foreign_state" else True
        )
        attrs["history_source"] = source
        f = p._replace_value(f, attrs=attrs)
    with pytest.raises(ValueError, match="seed|State.n|closure"):
        interaction_contract(f)
    with pytest.raises(ValueError):
        p._serialize()


@pytest.mark.parametrize("real_type", ["float", "double"])
def test_actual_history_header_host_selected_authority(tmp_path, real_type):
    binary = tmp_path / "history-host"
    prefix = Path(sys.prefix)
    flags = [
        "/usr/bin/clang++",
        "-std=c++20",
        "-O0",
        "-fno-fast-math",
        "-DPOPS_HAS_KOKKOS",
        "-DPOPS_REAL_TYPE=" + real_type,
        "-DPOPS_NATIVE_DIM=2",
        "-I" + str(ROOT / "include"),
        "-I" + str(prefix / "include"),
        "-Xpreprocessor",
        "-fopenmp",
        "-I/opt/homebrew/opt/libomp/include",
        str(ROOT / "tests/review/sol61_spatial_history_independent.cpp"),
        "-L" + str(prefix / "lib"),
        "-lkokkoscore",
        "-lomp",
        "-Wl,-rpath," + str(prefix / "lib"),
        "-o",
        str(binary),
    ]
    result = subprocess.run(flags, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run(
        [str(binary)],
        capture_output=True,
        text=True,
        timeout=30,
        env=dict(os.environ, OMP_NUM_THREADS="2"),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        f"independent-history checks=10 Realbits={32 if real_type == 'float' else 64}"
        in result.stdout
    )
