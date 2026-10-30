"""One genuine RuntimeInstance phase of the 1→2→3→1 restart receiver."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np  # noqa: E402
import pops  # noqa: E402
from examples.api_v02.continuation_migration_amr import build_continuation_migration_amr  # noqa: E402
from tests.python.support.api_v02_native import _bind, _compile  # noqa: E402
from tests.python.support.amr_continuation_receipts import forest  # noqa: E402
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check  # noqa: E402
from tests.python.support.public_native_profile import profile_gather_value  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("phase", type=int)
    args = parser.parse_args()
    phase, directory = args.phase, args.directory
    spec, artifact, world = _compile("forest_mpi_migration", builder=build_continuation_migration_amr,
                                     cells=(8, 8), steps=4, ratio=2)
    rank, ranks = (0, 1) if world is None else (int(world.rank), int(world.size))
    runtime = _bind(spec, artifact, world)

    def accepted():
        return collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))

    def fields():
        return [collective_call(world, lambda level=level:
                    runtime.block_level_state_global("method", level))
                for level in range(runtime.n_levels())]

    source = None
    if phase:
        source = directory / ("checkpoint-%d.npz" % (phase-1))
        initial = accepted()
        _, failures = collective_attempt(world, lambda: runtime.restart(source, workspace_byte_capacity=1))
        after_rejection = accepted()
        with collective_check(world):
            assert failures, "insufficient workspace must reject before publication"
            assert after_rejection == initial
            assert runtime.time() == 0 and runtime.macro_step() == 0
        collective_call(world, lambda: runtime.restart(source, workspace_byte_capacity=64 << 20))
        restored = fields()
        with collective_check(world):
            assert runtime.macro_step() == phase and runtime.time() == phase*spec.dt
            if rank == 0:
                with np.load(directory / ("fields-%d.npz" % (phase-1)), allow_pickle=False) as saved:
                    assert len(saved.files) == len(restored)
                    for level, current in enumerate(restored):
                        np.testing.assert_array_equal(np.asarray(current).view(np.uint64),
                                                      saved[str(level)].view(np.uint64))
            receipt = runtime._executor._last_restart_migration_receipt
            assert receipt["complete_rank_table"] and receipt["target_ranks"] == ranks
    report = collective_call(world, lambda: pops.step(runtime, t_bound=spec.t_end, console=False))
    image = accepted()
    decoded = collective_call(world, lambda: forest(spec, image, rank=rank))
    all_rows = profile_gather_value(world, collective_call(world, lambda: spec.invariants(runtime)))
    with collective_check(world):
        assert report.accepted_steps == 1 and report.rejected_steps == 0
        assert all(row["passed"] for row in all_rows), all_rows
        assert decoded["records"][0]["root"]["step"] == phase+1
        for record in decoded["records"][1:]:
            history = record["root"]["history"]
            assert history is record["root"]["history_alias"]
            domain = history["domain"]
            start, end = domain["offsets"][rank:rank+2]
            assert history["values"] == [(1 << 60)+17+gid+phase+1 for gid in range(start, end)]
            q, rem = divmod(7, ranks)
            assert domain["offsets"] == [i*q+min(i, rem) for i in range(ranks+1)]
    state = fields()
    checkpoint = collective_call(world, lambda: runtime.checkpoint(str(directory / ("checkpoint-%d" % phase))))
    issuers = [list(record["root"]["history"]["domain"]["issuer"]) for record in decoded["records"][1:]]
    identities = profile_gather_value(world, dict(rank=rank, issuers=issuers,
        image=hashlib.sha256(image).hexdigest()))
    binaries = {key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for key, path in artifact.layout_program_paths.items()}
    with collective_check(world):
        if rank == 0:
            if phase:
                previous = json.loads((directory / ("receipt-%d.json" % (phase-1))).read_text())
                assert previous["artifact"] == artifact.artifact_identity.hexdigest
                assert previous["program"] == spec.case._time._ir_hash()
                assert previous["binaries"] == binaries
                old = {tuple(value) for row in previous["ranks"] for value in row["issuers"]}
                assert not old.intersection(tuple(value) for row in identities for value in row["issuers"])
            np.savez(directory / ("fields-%d.npz" % phase), **{str(i): a for i, a in enumerate(state)})
            (directory / ("receipt-%d.json" % phase)).write_text(json.dumps(dict(
                artifact=artifact.artifact_identity.hexdigest, program=spec.case._time._ir_hash(),
                binaries=binaries, ranks=identities, step=phase+1, checkpoint=str(checkpoint)), indent=2))
    print("PASS native forest phase=%d ranks=%d rank=%d" % (phase, ranks, rank), flush=True)


if __name__ == "__main__":
    main()
