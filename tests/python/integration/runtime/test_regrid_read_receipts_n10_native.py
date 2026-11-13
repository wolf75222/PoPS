"""N10 lot 3b follow-up (review of #802, R1 and R4): the native refusals and the DM6 counter.

``regrid_read`` refuses at run time a source level finer than its target (G11) and a valid
target cell with no valid parent on the source level; the generated program counts the bytes
each read moves (``pops_relayout_byte_count``). The case is ``continuation_regrid_amr``; the two
refusals change only the source level its remap reads, through ``regrid_source``:

- finer: the finest source level, so at the second regrid the level 0 child reads level 1;
- no parent: the child's own level once the source has it, so a level 1 patch that grew reads
  level 1 cells that did not exist.
"""
import contextlib
import ctypes
from pathlib import Path

import pops
import pytest

from examples.api_v02.continuation_regrid_amr import build_continuation_regrid_amr
from pops.time._program.continuation_regrid import _ProgramRegrid
from pops.time._program.structured_control import new_typed
from tests.python.support.amr_continuation_receipts import forest
from tests.python.support.api_v02_native import _bind
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _proposal_int(b, op):
    return new_typed(b, {"vtype": "int64"}, op, (), {})


def _finest(b):
    # The finest level of the source forest: source_count - 1.
    return _proposal_int(b, "continuation_regrid_source_count")-b.int64(1)


def _own_once_present(b):
    # The child's level when the source has two levels, else 0 (two levels at most here).
    level = _proposal_int(b, "continuation_forest_level")
    return level*(_proposal_int(b, "continuation_regrid_source_count")-b.int64(1))


@contextlib.contextmanager
def _source_level(choose):
    original = _ProgramRegrid.regrid_source

    def regrid_source(self, level):
        del level
        return original(self, choose(self))
    _ProgramRegrid.regrid_source = regrid_source
    try:
        yield
    finally:
        _ProgramRegrid.regrid_source = original


def _compile(route, choose=None, steps=3):
    _, world = native_profile(dimension=2, scope="N10 lot 3b regrid_read " + route)

    def build():
        if choose is None:
            return build_continuation_regrid_amr(steps=steps)
        with _source_level(choose):
            return build_continuation_regrid_amr(steps=steps)
    spec = collective_call(world, build)
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case), layout=spec.layout))
    artifact = compile_profile_plan_once(world, plan, route="N10 3b regrid_read " + route,
                                         compile_artifact=pops.compile)
    return spec, artifact, world


def _relayout_bytes(artifact):
    dso = ctypes.CDLL(str(Path(artifact.program.so_path).resolve()))
    count = dso.pops_relayout_byte_count
    count.argtypes, count.restype = [], ctypes.c_uint64
    return int(count())


def _scratch_bytes(field, level):
    # One scratch per target patch: the coarsening of its box grown by the ghosts, by
    # r = 2**level, all components, 8-byte Reals (regrid_read_support.hpp).
    ghosts = [axis[0] for axis in field["axes"]]
    ratio = 2**level
    total = 0
    for (lo_x, hi_x, _), (lo_y, hi_y, _) in field["layout"]:
        cells = 1
        for lo, hi, g in ((lo_x, hi_x, ghosts[0]), (lo_y, hi_y, ghosts[1])):
            cells *= (hi+g)//ratio-(lo-g)//ratio+1
        total += cells*field["components"]*8
    return total


def test_the_relayout_counter_moves_the_scratch_bytes_on_the_regrid_step(
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("counter")
    runtime = _bind(spec, artifact, world)
    moved, images = [], []
    rank = 0 if world is None else int(world.rank)
    for _ in range(3):
        before = _relayout_bytes(artifact)
        collective_call(world, lambda: pops.step(runtime, t_bound=spec.t_end, console=False))
        moved.append(_relayout_bytes(artifact)-before)
        raw = collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))
        images.append(collective_call(world, lambda raw=raw: forest(spec, raw, rank=rank)))
    with collective_check(world):
        # Step 2 is the one regrid: two Fields (state, previous) read once per target level.
        assert [image["records"][0]["root"]["regrids"] for image in images] == [0, 1, 1]
        expected = sum(_scratch_bytes(record["root"][name], record["level"])
                       for record in images[1]["records"][1:] for name in ("state", "previous"))
        assert expected > 0
        assert moved == [0, expected, 0], (moved, expected)
    record_property("relayout_bytes", moved)


@pytest.mark.parametrize("refusal,choose,message", [
    ("finer", _finest, "regrid_read reads a source level finer than its target: restriction is left to G11"),
    ("no_parent", _own_once_present, "has no valid parent on source level 1"),
], ids=["finer", "no_parent"])
def test_regrid_read_refuses_at_run_time_with_its_named_message(
        refusal, choose, message, isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile(refusal, choose, steps=8)
    runtime = _bind(spec, artifact, world)
    with collective_check(world):
        assert world is None or int(world.size) == 1, "the named message is a one-rank receipt (R2)"
    attempted = []
    with pytest.raises(Exception) as caught:
        for step in range(1, 9):
            attempted.append(step)
            pops.step(runtime, t_bound=spec.t_end, console=False)
    assert message in str(caught.value), str(caught.value)
    # The first regrid (step 2) reads level 0 only; both refusals need a source with level 1.
    assert attempted[-1] >= 4, attempted
    record_property("refused_at_step", attempted[-1])
