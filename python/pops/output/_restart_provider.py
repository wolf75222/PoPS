"""Output-owned restart operation provider used by checkpoint ConsumerGraph nodes."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ReopenedRestart:
    target: Path
    cursors: Any


@dataclass(frozen=True, slots=True)
class RestartV3:
    """Immutable adapter over the strict Uniform/AMR accepted-state v3 codecs."""

    __pops_ir_immutable__ = True
    bit_identical: bool = False

    def __post_init__(self) -> None:
        if type(self.bit_identical) is not bool:
            raise TypeError("RestartV3.bit_identical must be an exact bool")

    def consumer_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider_id": "pops.restart.accepted-state-v3",
            "extension": ".npz",
            "bit_identical": self.bit_identical,
            # A serial runtime is the one-member case of this collective operation.  Providers
            # without this explicit capability (notably parallel HDF5) remain fail-closed when no
            # distributed communicator is present.
            "supports_singleton_collective": True,
        }

    def snapshot(self, runtime: Any, directory: Any) -> Path:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".pops-restart-snapshot.", suffix=".npz", dir=root)
        os.close(fd)
        os.unlink(name)
        produced = Path(runtime._checkpoint_payload(name))
        if produced != Path(name) or not produced.is_file():
            produced.unlink(missing_ok=True)
            raise RuntimeError("restart provider did not capture the exact staged snapshot")
        return produced

    def write(self, snapshot: Any, target: Any) -> Path:
        source, destination = Path(snapshot), Path(target)
        if not source.is_file():
            raise ValueError("restart snapshot is not a staged file")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError("checkpoint target collision: %s" % destination)
        os.replace(source, destination)
        return destination

    def reopen(self, runtime: Any, path: Any) -> ReopenedRestart:
        target, cursors = runtime._reopen_checkpoint(path)
        return ReopenedRestart(Path(target), cursors)

    def restore(self, runtime: Any, reopened: Any) -> Any:
        if type(reopened) is not ReopenedRestart:
            raise TypeError("RestartV3.restore requires an exact ReopenedRestart")
        return runtime._restore_checkpoint(reopened.target, reopened.cursors)


__all__ = ["ReopenedRestart", "RestartV3"]
