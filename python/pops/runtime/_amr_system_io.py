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


@dataclass(frozen=True, slots=True)
class _AMRRegridRestartEvidence:
    """All-rank evidence agreed before the native restart transaction commits."""

    restart_identity: Any
    regrid_receipt: Any

    def to_data(self) -> dict[str, Any]:
        return {
            "restart_identity": self.restart_identity.token,
            "regrid_receipt": dict(self.regrid_receipt),
        }


class _AmrSystemIO(_AmrSystem):
    """Private accepted-state codec and transactional restore adapter for ``AmrSystem``."""

    def set_history_persistence(self, mapping: Any) -> Any:
        self._history_persistence = dict(mapping or {})
        return self

    def last_restart_report(self) -> Any:
        return getattr(self, "_last_restart_report", None)

    def last_restart_regrid_receipt(self) -> Any:
        return getattr(self, "_last_restart_regrid_receipt", None)

    def checkpoint(self, path: Any) -> Any:
        """Encode the complete accepted AMR state for the RuntimeInstance checkpoint provider.

        The provider owns collective publication; this adapter serializes the installed hierarchy,
        temporal state, histories, and regrid state for frozen or active regridding.
        """
        from pops.runtime._amr_checkpoint_v3 import write_v3

        return write_v3(
            self,
            self._s,
            path,
            self._regrid_every,
            getattr(self, "_history_persistence", None) or {},
        )

    def _prepare_checkpoint_restart(
        self,
        payload: bytes,
        *,
        bit_identical: bool,
        hierarchy_mode: str = "restore_recorded_hierarchy",
        hierarchy_identity: str | None = None,
    ) -> _PreparedAMRSystemRestart:
        """Authenticate and preflight the complete AMR payload without native mutation."""
        from pops.output._checkpoint_collective import (
            decode_checkpoint_bytes,
            require_restart_bit_identical,
            require_restart_hierarchy_mode,
        )
        from pops._generated_release_contract import AMR_CHECKPOINT_PAYLOAD_VERSION
        from pops.runtime._checkpoint_manifest import (
            authenticate_checkpoint_payload,
            require_exact_payload_version,
        )
        from pops.runtime._amr_checkpoint_v3 import prepare_v3

        require_restart_bit_identical(bit_identical, where="AMR restart")
        selected_hierarchy_mode = require_restart_hierarchy_mode(
            hierarchy_mode, where="AMR restart"
        )
        data = decode_checkpoint_bytes(payload)
        identity = authenticate_checkpoint_payload(self, data, runtime_kind="amr")
        require_exact_payload_version(
            data,
            key="pops_amr_checkpoint_version",
            expected=AMR_CHECKPOINT_PAYLOAD_VERSION,
            runtime_kind="AMR",
        )
        return _PreparedAMRSystemRestart(
            identity,
            prepare_v3(
                self,
                self._s,
                data,
                bit_identical=bit_identical,
                hierarchy_mode=selected_hierarchy_mode,
                hierarchy_identity=hierarchy_identity,
            ),
        )

    def _begin_checkpoint_restart(self) -> None:
        if "_checkpoint_restart_python_snapshot" in self.__dict__:
            raise RuntimeError("AMR checkpoint restart transaction is already active")
        self._checkpoint_restart_python_snapshot = (
            getattr(self, "_last_restart_identity", None),
            getattr(self, "_last_restart_report", None),
            getattr(self, "_last_restart_regrid_receipt", None),
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
        if prepared.codec.hierarchy_mode == "regrid_on_restart":
            return _AMRRegridRestartEvidence(
                prepared.restart_identity,
                self._last_restart_regrid_receipt,
            )
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
            (
                self._last_restart_identity,
                self._last_restart_report,
                self._last_restart_regrid_receipt,
                self._temporal_restart_state,
                self._step_controller,
            ) = snapshot
            self.__dict__.pop("_checkpoint_restart_committed", None)
            del self._checkpoint_restart_python_snapshot

    def restart(self, path: Any, *, bit_identical: bool = False) -> Any:
        """Restore the direct AMR engine through the native collective transaction protocol."""
        from pops.output._checkpoint_collective import restore_checkpoint_path

        return restore_checkpoint_path(
            self,
            self,
            path,
            bit_identical=bit_identical,
            phase_prefix="AMR direct-engine restart",
        )


__all__ = ["_AmrSystemIO"]
