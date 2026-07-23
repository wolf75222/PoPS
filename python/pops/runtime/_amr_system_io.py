"""Strict accepted-state checkpoint/restart mixin for the AMR engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pops.runtime._amr_system_contract import _AmrSystem
else:
    _AmrSystem = object


@dataclass(frozen=True, slots=True)
class _PreparedAMRSystemRestart:
    restart_identity: Any
    codec: Any


class _AmrSystemIO(_AmrSystem):
    """Private accepted-state codec and transactional restore adapter for ``AmrSystem``."""

    def set_history_persistence(self, mapping: Any) -> Any:
        self._history_persistence = dict(mapping or {})
        return self

    def last_restart_report(self) -> Any:
        return getattr(self, "_last_restart_report", None)

    def checkpoint(self, path: Any) -> Any:
        """Encode the complete accepted AMR state for the RuntimeInstance checkpoint provider.

        The provider owns collective publication; this adapter serializes the installed hierarchy,
        temporal state, histories, and regrid state for frozen or active regridding.
        """
        from pops.runtime._amr_checkpoint_v3 import write_v3

        return write_v3(
            self, self._s, path, (self._L, self._Ly), (self._xlo, self._ylo),
            self._regrid_every,
            getattr(self, "_history_persistence", None) or {})

    def _prepare_checkpoint_restart(self, payload: bytes) -> _PreparedAMRSystemRestart:
        """Authenticate and preflight the complete AMR payload without native mutation."""
        from pops.output._checkpoint_collective import decode_checkpoint_bytes
        from pops.runtime._checkpoint_manifest import authenticate_checkpoint_payload
        from pops.runtime._amr_checkpoint_v3 import prepare_v3

        data = decode_checkpoint_bytes(payload)
        identity = authenticate_checkpoint_payload(self, data, runtime_kind="amr")
        version = int(data["pops_amr_checkpoint_version"])
        if version != 3:
            raise ValueError(
                "restart: AMR checkpoint version %r unsupported; expected exactly 3" % version)
        return _PreparedAMRSystemRestart(
            identity, prepare_v3(
                self, self._s, data, (self._L, self._Ly), (self._xlo, self._ylo)))

    def _begin_checkpoint_restart(self) -> None:
        if "_checkpoint_restart_python_snapshot" in self.__dict__:
            raise RuntimeError("AMR checkpoint restart transaction is already active")
        self._checkpoint_restart_python_snapshot = (
            getattr(self, "_last_restart_identity", None),
            getattr(self, "_last_restart_report", None),
            getattr(self, "_temporal_restart_state", None),
            getattr(self, "_step_controller", None),
        )
        try:
            self._s.begin_restart_transaction()
        except BaseException:
            del self._checkpoint_restart_python_snapshot
            raise

    def _apply_checkpoint_restart(self, prepared: _PreparedAMRSystemRestart) -> Any:
        if type(prepared) is not _PreparedAMRSystemRestart:
            raise TypeError("AMR restart requires its exact prepared payload")
        from pops.runtime._amr_checkpoint_v3 import apply_v3

        self._last_restart_report = apply_v3(self, self._s, prepared.codec)
        self._last_restart_identity = prepared.restart_identity
        return prepared.restart_identity

    def _commit_checkpoint_restart(self) -> None:
        # Keep the native AcceptedSnapshot live through the all-rank commit consensus.
        self._checkpoint_restart_committed = True

    def _finalize_checkpoint_restart(self) -> None:
        if not self.__dict__.get("_checkpoint_restart_committed", False):
            raise RuntimeError("AMR checkpoint restart transaction was not committed")
        self._s.commit_restart_transaction()
        del self._checkpoint_restart_committed
        del self._checkpoint_restart_python_snapshot

    def _rollback_checkpoint_restart(self) -> None:
        snapshot = self._checkpoint_restart_python_snapshot
        try:
            self._s.rollback_restart_transaction()
        finally:
            (self._last_restart_identity, self._last_restart_report,
             self._temporal_restart_state, self._step_controller) = snapshot
            self.__dict__.pop("_checkpoint_restart_committed", None)
            del self._checkpoint_restart_python_snapshot

    def restart(self, path: Any) -> Any:
        """Restore the direct AMR engine through the native collective transaction protocol."""
        from pops.output._checkpoint_collective import restore_checkpoint_path

        return restore_checkpoint_path(
            self, self, path, phase_prefix="AMR direct-engine restart")


__all__ = ["_AmrSystemIO"]
