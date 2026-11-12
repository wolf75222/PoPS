"""N10 lot 3b gate (addendum §6.2), temporary: the bridge and the port on the same steps.

The bridge is the code of the three regrid cases before the port, kept in
``examples/api_v02/_bridge_continuation_regrid_*.py``. Every cell of the remapped Fields, valid
and halo, must be bitwise equal; the masses agree to 3e-13 relative. The three cases are periodic
on every axis, so the port's halo reads the same wrapped parent as the bridge. This file and the
bridge copies are deleted with the bridge.
"""
import ctypes
from pathlib import Path

import numpy as np
import pops
import pytest

from examples.api_v02 import _bridge_continuation_regrid_amr as bridge_amr
from examples.api_v02 import _bridge_continuation_regrid_auxiliary as bridge_auxiliary
from examples.api_v02 import continuation_regrid_amr as port_amr
from examples.api_v02 import continuation_regrid_auxiliary as port_auxiliary
from tests.python.support.amr_continuation_receipts import forest
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _amr(module, independent):
    def build(world):
        spec = collective_call(world, lambda: module.build_continuation_regrid_amr(
            independent_previous=independent))
        return spec, {}, None
    return build


def _auxiliary(module):
    def build(world):
        spec, providers = collective_call(world, module.build_continuation_regrid_auxiliary)
        return spec, {"layout_providers": providers}, module.initial_auxiliary_values
    return build


CASES = {
    "amr": (_amr(bridge_amr, False), _amr(port_amr, False)),
    "independent": (_amr(bridge_amr, True), _amr(port_amr, True)),
    "auxiliary": (_auxiliary(bridge_auxiliary), _auxiliary(port_auxiliary)),
}


def _run(build, world, route, *, counted=False):
    spec, extra, auxiliary = build(world)
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case), layout=spec.layout,
                                                       **extra))
    artifact = compile_profile_plan_once(world, plan, route=route, compile_artifact=pops.compile)
    kwargs = {} if auxiliary is None else {"aux": auxiliary(spec, artifact.plan)}
    runtime = collective_call(world, lambda: pops.bind(
        artifact, initial_values={subject: value.copy() for subject, value in spec.initial_values.items()},
        resources={"execution_context": artifact_execution_context(artifact)}, **kwargs))
    images, moved = [], []
    for _ in range(3):
        before = _relayout_bytes(artifact) if counted else 0
        collective_call(world, lambda: pops.step(runtime, t_bound=spec.t_end, console=False))
        moved.append(_relayout_bytes(artifact)-before if counted else 0)
        raw = collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))
        rank = 0 if world is None else int(world.rank)
        images.append(collective_call(world, lambda raw=raw: forest(spec, raw, rank=rank)))
    return images, moved


def _fields(decoded):
    rows = {}
    for record in decoded["records"][1:]:
        for name in ("state", "previous"):
            field = record["root"][name]
            rows[(record["level"], name)] = field
    return rows


def _relayout_bytes(artifact):
    dso = ctypes.CDLL(str(Path(artifact.program.so_path).resolve()))
    count = dso.pops_relayout_byte_count
    count.argtypes, count.restype = [], ctypes.c_uint64
    return int(count())


def _scratch_bytes(field, level):
    # Coarsening of every grown target box by r = 2**level, all components, 8-byte Reals.
    ghosts = [axis[0] for axis in field["axes"]]
    ratio = 2**level
    total = 0
    for (lo_x, hi_x, _), (lo_y, hi_y, _) in field["layout"]:
        cells = 1
        for lo, hi, g in ((lo_x, hi_x, ghosts[0]), (lo_y, hi_y, ghosts[1])):
            cells *= (hi+g)//ratio-(lo-g)//ratio+1
        total += cells*field["components"]*8
    return total


@pytest.mark.parametrize("case", sorted(CASES))
def test_port_equals_bridge_on_every_cell(case, isolated_native_cache, native_cxx, kokkos_root,
                                          record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="N10 lot 3b regrid gate "+case)
    bridge_build, port_build = CASES[case]
    bridge, _ = _run(bridge_build, world, "N10 3b gate bridge "+case)
    port, moved = _run(port_build, world, "N10 3b gate port "+case, counted=True)
    halo_cells = 0
    with collective_check(world):
        assert len(bridge[-1]["records"]) > 2, "the gate needs a refined child level"
        for step, (left, right) in enumerate(zip(bridge, port, strict=True), start=1):
            a, b = left["records"][0]["root"], right["records"][0]["root"]
            assert a["step"] == b["step"] == step
            assert abs(a["mass"]-b["mass"]) <= 3.e-13*abs(a["mass"]), (step, a["mass"], b["mass"])
            fa, fb = _fields(left), _fields(right)
            assert fa.keys() == fb.keys()
            for key, x in fa.items():
                y = fb[key]
                assert x["layout"] == y["layout"] and x["axes"] == y["axes"], key
                assert x["components"] == y["components"], key
                assert x["patches"].keys() == y["patches"].keys(), key
                for ordinal, payload in x["patches"].items():
                    other = y["patches"][ordinal]
                    assert payload.dtype == other.dtype and payload.shape == other.shape
                    assert payload.tobytes() == other.tobytes(), (step, key, ordinal)
                    (lo_x, hi_x, _), (lo_y, hi_y, _) = x["layout"][ordinal]
                    if lo_x == 0 and step == 2:
                        # Step 2 publishes the regrid: the low halo of a patch at the origin,
                        # index -1, holds what regrid_read copied from the periodic parent.
                        gx, gy = (axis[0] for axis in x["axes"])
                        nx, ny = hi_x-lo_x+1, hi_y-lo_y+1
                        values = other.reshape(x["components"], ny+2*gy, nx+2*gx)
                        column = values[:, gy:gy+ny, gx-1]
                        assert np.all(column != 0.), (step, key, ordinal)
                        halo_cells += column.size
        assert halo_cells > 0, "no patch of the gate touches index -1"
        # Step 2 is the one regrid that changes the forest; the read copies once per Field.
        assert [row["records"][0]["root"]["regrids"] for row in port] == [0, 1, 1]
        expected = sum(_scratch_bytes(field, level) for (level, _), field in _fields(port[1]).items())
        assert moved == [0, expected, 0], (moved, expected)
    record_property("halo_cells_at_minus_one", halo_cells)
    record_property("relayout_bytes", moved)
    record_property("scratch_bytes_after_regrid", expected)
