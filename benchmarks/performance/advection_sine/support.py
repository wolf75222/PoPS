"""Non-scientific support for the public sine-advection performance case.

The case itself stays deliberately linear.  This module owns command-line
parsing, provenance and JSON publication so the science is not hidden behind a
large factory function.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


MEASUREMENT_SCHEMA = "pops.performance.advection-sine.measurement.v3"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Public PoPS sine-advection performance case")
    parser.add_argument("--resolution", required=True)
    parser.add_argument("--mode", choices=("x", "y", "z", "diagonal"), required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--point", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--expected-ranks", type=int, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--cfl", type=float, required=True)
    parser.add_argument("--warmups", type=int, required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.resolution = tuple(int(token) for token in args.resolution.split(","))
    if len(args.resolution) not in (1, 2, 3) or any(value < 4 for value in args.resolution):
        parser.error("--resolution must contain 1, 2, or 3 integers >= 4")
    if args.steps < 1 or args.warmups < 0 or args.repetitions < 3:
        parser.error("steps >= 1, warmups >= 0 and repetitions >= 3 are required")
    if args.block_size < 1 or args.threads < 1 or args.expected_ranks < 1 or args.nodes < 1:
        parser.error("resource and block counts must be positive")
    if not 0.0 < args.cfl < 1.0:
        parser.error("--cfl must lie strictly between zero and one")
    return args


def write_rank_measurement(output_dir: Path, rank: int, payload: dict[str, Any]) -> None:
    """Publish one rank-owned JSON atomically and fail closed on any collision."""
    output_dir.mkdir(parents=True, exist_ok=True)
    final = output_dir / ("rank-%05d.json" % rank)
    temporary = output_dir / (".%s.%d.tmp" % (final.name, os.getpid()))
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # link(2) is atomic and refuses an existing destination: no rank can
        # overwrite previous evidence, even after a failed earlier campaign.
        os.link(temporary, final)
    except FileExistsError as error:
        raise FileExistsError("refusing to overwrite performance evidence %s" % final) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
