"""Runtime-owned ConsumerGraph publication against accepted native state."""
from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pops.identity import Identity, make_identity
from pops.output.data import (
    ArrayPiece,
    DiagnosticKey,
    DiagnosticPayload,
    FieldKey,
    FieldPayload,
    LevelGeometry,
    OutputClock,
    OutputProvenance,
    OutputRequest,
    OutputSnapshot,
)
from pops.output.writers import deterministic_target

from .consumer import (
    AcceptedSideEffect,
    ConsumerKind,
    ConsumerPublisher,
    ParallelMode,
    PreparedPublication,
    PublicationReceipt,
)
from .output_publisher import ConsumerOutputPublisher, OutputPreparation


def _block_name(reference: Any, names: tuple[str, ...]) -> str:
    block = getattr(reference, "block_ref", None)
    local_id = getattr(block, "local_id", None)
    if local_id in names:
        return local_id
    if len(names) == 1:
        return names[0]
    raise ValueError("consumer reference has no exact installed block owner")


def _layout_identity(layout: Any) -> Identity:
    return make_identity("layout", layout.to_data())


def _scalar(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, Mapping):
        return None
    row = value.get("$scalar")
    if not isinstance(row, Mapping):
        return None
    kind, encoded = row.get("kind"), row.get("value")
    if kind == "binary64":
        return float.fromhex(encoded)
    if kind in ("integer", "decimal"):
        return float(encoded)
    if kind == "rational":
        numerator, denominator = encoded
        return float(numerator) / float(denominator)
    return None


def _layout_length(layout: Any, engine: Any) -> float:
    explicit = getattr(engine, "_L", None)
    if explicit is not None:
        return float(explicit)
    found = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key == "L" and (number := _scalar(item)) is not None:
                    found.append(number)
                visit(item)
        elif isinstance(value, tuple):
            for item in value:
                visit(item)

    visit(layout.descriptor_snapshot)
    values = set(found)
    if len(values) != 1 or next(iter(values)) <= 0.0:
        raise ValueError("layout geometry does not prove one positive physical extent L")
    return next(iter(values))


def _target(uri: str, format_data: Mapping[str, Any], snapshot: OutputSnapshot,
            request: OutputRequest,
            consumer_name: str, output_root: Any) -> Path:
    path = Path(uri)
    if output_root is not None:
        path = Path(output_root) / path.name
    if path.suffix:
        if path.suffix != format_data["extension"]:
            raise ValueError("consumer target suffix does not match its exact format")
        return path
    return deterministic_target(
        path, consumer_name, request, snapshot, format_data["extension"])


class _PreparedDiagnostic(PreparedPublication):
    def __init__(self, effect: AcceptedSideEffect, values: tuple[DiagnosticPayload, ...],
                 publish: Any, discard: Any, rollback: Any) -> None:
        self._effect, self._values = effect, values
        self._publish, self._discard, self._rollback = publish, discard, rollback
        self._published = self._discarded = False

    @property
    def effect_identity(self) -> Identity:
        return self._effect.identity

    @property
    def payload_identity(self) -> Identity:
        return self._effect.payload.identity

    def publish(self) -> PublicationReceipt:
        if self._discarded:
            raise RuntimeError("discarded diagnostic cannot be published")
        if not self._published:
            self._publish(self._effect, self._values)
            self._published = True
        artifact = make_identity("runtime-diagnostic-publication", {
            "effect": self.effect_identity.token,
            "values": [value.to_data() for value in self._values],
        })
        return PublicationReceipt(
            self.effect_identity, self.payload_identity,
            "pops.runtime-diagnostic.v1", artifact.token)

    def discard(self) -> None:
        if not self._published and not self._discarded:
            self._discard(self._effect)
            self._discarded = True

    def rollback(self) -> None:
        if self._discarded:
            return
        if self._published:
            self._rollback(self._effect, self._values)
        else:
            self._discard(self._effect)
        self._published = False
        self._discarded = True


class _PreparedCheckpoint(PreparedPublication):
    def __init__(self, effect: AcceptedSideEffect, engine: Any, operation: Any,
                 target: Any) -> None:
        self._effect, self._target = effect, Path(target)
        self._target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=".%s." % self._target.name, suffix=".npz",
            dir=str(self._target.parent))
        os.close(fd)
        os.unlink(temporary)
        snapshot = operation.snapshot(engine, self._target.parent)
        produced = Path(operation.write(snapshot, temporary))
        if produced != Path(temporary) or not produced.is_file():
            produced.unlink(missing_ok=True)
            raise RuntimeError("checkpoint codec did not produce the exact staged target")
        self._temporary, self._published, self._discarded = produced, False, False

    @property
    def effect_identity(self) -> Identity:
        return self._effect.identity

    @property
    def payload_identity(self) -> Identity:
        return self._effect.payload.identity

    def publish(self) -> PublicationReceipt:
        if self._discarded:
            raise RuntimeError("discarded checkpoint cannot be published")
        if not self._published:
            if self._target.exists():
                raise FileExistsError("checkpoint target collision: %s" % self._target)
            os.replace(self._temporary, self._target)
            self._published = True
        artifact = make_identity("restart-checkpoint-artifact", {
            "effect": self.effect_identity.token, "target": str(self._target)})
        return PublicationReceipt(
            self.effect_identity, self.payload_identity,
            "pops.restart-checkpoint.v3", artifact.token)

    def discard(self) -> None:
        if not self._published and not self._discarded:
            self._temporary.unlink(missing_ok=True)
            self._discarded = True

    def rollback(self) -> None:
        if self._discarded:
            return
        if self._published:
            self._target.unlink(missing_ok=True)
        self._temporary.unlink(missing_ok=True)
        self._published = False
        self._discarded = True


class RuntimeConsumerPublisher(ConsumerPublisher):
    """One publisher for diagnostics, exact outputs, monitors and restart checkpoints."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner
        self._by_id = {row.qualified_id: row for row in owner.consumer_graph.nodes}
        self._pending: dict[str, tuple[DiagnosticPayload, ...]] = {}
        self._diagnostics: dict[str, DiagnosticPayload] = {}
        self._output = ConsumerOutputPublisher(self._resolve_output)

    @property
    def diagnostics(self) -> tuple[DiagnosticPayload, ...]:
        staged = [value for rows in self._pending.values() for value in rows]
        return tuple(sorted((*self._diagnostics.values(), *staged),
                            key=lambda value: value.key.identity.token))

    def _manifest(self, effect: AcceptedSideEffect) -> Any:
        try:
            manifest = self._by_id[effect.consumer_id]
        except KeyError:
            raise ValueError("accepted effect names no installed ConsumerGraph node") from None
        if manifest.identity != effect.manifest_identity:
            raise ValueError("accepted effect manifest identity is stale")
        return manifest

    def _diagnostic_values(self, manifest: Any) -> tuple[DiagnosticPayload, ...]:
        engine = self._owner.native_executor
        names = tuple(engine.block_names())
        values = []
        for quantity in manifest.quantities:
            block = _block_name(quantity.reference, names)
            variables = tuple(engine.variable_names(block, "conservative"))
            if quantity.reference.local_id in variables:
                component = variables.index(quantity.reference.local_id)
            elif len(variables) == 1:
                component = 0
            else:
                raise ValueError(
                    "diagnostic quantity must select an exact conservative component")
            reduction = next((part for part in quantity.runtime_resource.split(":")
                              if part in {"sum", "abs_sum", "sum_sq", "min", "max", "abs_max"}),
                             "sum")
            call = getattr(engine, "composite_reduce", None)
            value = call(block, reduction, component) if callable(call) else \
                engine.reduce_component(block, reduction, component)
            key = DiagnosticKey(
                quantity.reference,
                self._owner.component_manifests[block].manifest_digest,
                self._owner.layout_identity(quantity.layout_id),
                "accepted",
                reduction,
            )
            values.append(DiagnosticPayload(key, value, "unspecified", {}))
        return tuple(values)

    def _publish_diagnostics(self, effect: AcceptedSideEffect,
                             values: tuple[DiagnosticPayload, ...]) -> None:
        for value in values:
            self._diagnostics[value.key.identity.token] = value
            recorder = getattr(self._owner.native_executor, "record_program_diagnostic", None)
            if callable(recorder):
                recorder(effect.consumer_id, value.value)
        self._pending.pop(effect.identity.token, None)

    def _discard_diagnostics(self, effect: AcceptedSideEffect) -> None:
        self._pending.pop(effect.identity.token, None)

    def _prepare_diagnostic(self, effect: AcceptedSideEffect, manifest: Any) -> Any:
        values = self._diagnostic_values(manifest)
        previous = {
            value.key.identity.token: self._diagnostics.get(value.key.identity.token)
            for value in values
        }
        existed = {
            value.key.identity.token: value.key.identity.token in self._diagnostics
            for value in values
        }

        def rollback(_effect: AcceptedSideEffect,
                     published: tuple[DiagnosticPayload, ...]) -> None:
            for value in published:
                token = value.key.identity.token
                if existed[token]:
                    self._diagnostics[token] = previous[token]
                else:
                    self._diagnostics.pop(token, None)
            self._pending.pop(_effect.identity.token, None)

        self._pending[effect.identity.token] = values
        return _PreparedDiagnostic(
            effect, values, self._publish_diagnostics, self._discard_diagnostics, rollback)

    def _resolve_output(self, effect: AcceptedSideEffect) -> OutputPreparation:
        manifest = self._manifest(effect)
        snapshot, request = self._owner.output_snapshot(manifest, self.diagnostics)
        fmt = manifest.output_format
        target = _target(
            effect.target.uri, manifest.output_format_data, snapshot, request,
            manifest.handle.local_id, self._owner.output_root)
        communicator = self._owner.execution_context.communicator.handle \
            if effect.target.parallel_mode is not ParallelMode.SERIAL else None
        return OutputPreparation(fmt, snapshot, request, target, communicator)

    def prepare(self, effect: AcceptedSideEffect) -> PreparedPublication:
        if type(effect) is not AcceptedSideEffect:
            raise TypeError("RuntimeConsumerPublisher requires an exact AcceptedSideEffect")
        manifest = self._manifest(effect)
        if manifest.kind in (ConsumerKind.DIAGNOSTIC, ConsumerKind.MONITOR):
            return self._prepare_diagnostic(effect, manifest)
        if manifest.kind is ConsumerKind.SCIENTIFIC_OUTPUT:
            return self._output.prepare(effect)
        if manifest.kind is ConsumerKind.CHECKPOINT:
            target = Path(effect.target.uri)
            if self._owner.output_root is not None:
                target = Path(self._owner.output_root) / target.name
            extension = manifest.operation_data["extension"]
            if target.suffix != extension:
                target = target.with_suffix(extension)
            return _PreparedCheckpoint(
                effect, self._owner, manifest.operation, target)
        raise TypeError("unsupported ConsumerKind %r" % manifest.kind)


class RuntimeOutputSnapshot:
    """Materialize exact output values from one accepted RuntimeInstance snapshot."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner

    def _layout(self, layout_id: str) -> Any:
        rows = [row for row in self._owner.layout_plan.layouts
                if row.handle.qualified_id == layout_id]
        if len(rows) != 1:
            raise KeyError("consumer selected unknown layout %s" % layout_id)
        return rows[0]

    def _geometry(self, layout: Any, level: int) -> LevelGeometry:
        import numpy as np

        engine = self._owner.native_executor
        nx, ny = int(engine.nx()), int(engine.ny())
        scale = layout.ratio ** level
        nx, ny = nx * scale, ny * scale
        length = _layout_length(layout, engine)
        boxes = ((0, 0, ny, nx),)
        if layout.adaptive and level > 0:
            native = [row for row in engine.patch_boxes() if int(row[0]) == level]
            boxes = tuple(sorted((int(jlo), int(ilo), int(jhi) + 1, int(ihi) + 1)
                                 for _, ilo, jlo, ihi, jhi in native))
            if not boxes:
                raise ValueError("selected AMR level has no materialized patches")
        coverage = np.zeros((ny, nx), dtype=np.bool_)
        if layout.adaptive:
            for level_index, ilo, jlo, ihi, jhi in engine.patch_boxes():
                if int(level_index) != level + 1:
                    continue
                ratio = layout.ratio
                coverage[int(jlo) // ratio:(int(jhi) + 1 + ratio - 1) // ratio,
                         int(ilo) // ratio:(int(ihi) + 1 + ratio - 1) // ratio] = True
        spacing = (length / nx, length / ny)
        volumes = np.full((ny, nx), spacing[0] * spacing[1], dtype=np.float64)
        return LevelGeometry(
            _layout_identity(layout), "amr" if layout.adaptive else "uniform", level,
            (0.0, 0.0), spacing, (ny, nx), boxes, coverage, volumes)

    def _state(
        self, block: str, layout: Any, level: int, *, collective: bool
    ) -> tuple[tuple[ArrayPiece, ...], tuple[str, ...]]:
        import numpy as np

        engine = self._owner.native_executor
        names = tuple(engine.variable_names(block, "conservative"))
        if layout.adaptive:
            communicator = self._owner.execution_context.communicator.handle
            size = 1 if communicator is None else int(communicator.Get_size())
            if collective and size > 1:
                raise NotImplementedError(
                    "collective adaptive output requires native rank-owned patch state; "
                    "the current AMR facade exposes only full-grid local/global buffers"
                )
            multi = int(engine.n_blocks()) != 1
            getter = engine.block_level_state if multi else engine.level_state
            raw = getter(block, level) if multi else getter(level)
            n = int(engine.nx()) * (layout.ratio ** level)
            values = np.asarray(raw, dtype=np.float64).reshape(len(names), n, n)
            return (ArrayPiece((0, 0), (n, n), values),), names
        if collective:
            pieces = []
            for index, (ilo, jlo, ihi, jhi) in enumerate(engine._s.local_boxes(block)):
                values = np.asarray(
                    engine._s.local_state(block, index), dtype=np.float64)
                pieces.append(ArrayPiece(
                    (int(jlo), int(ilo)), (int(jhi) + 1, int(ihi) + 1), values))
            return tuple(pieces), names
        values = np.asarray(engine._s.state_global(block), dtype=np.float64).reshape(
            len(names), int(engine.ny()), int(engine.nx()))
        return (ArrayPiece((0, 0), (int(engine.ny()), int(engine.nx())), values),), names

    def build(self, manifest: Any, diagnostics: tuple[DiagnosticPayload, ...]) \
            -> tuple[OutputSnapshot, OutputRequest]:
        geometries, fields, keys = {}, [], []
        names = tuple(self._owner.native_executor.block_names())
        for quantity in manifest.quantities:
            layout = self._layout(quantity.layout_id)
            levels = quantity.levels or tuple(row.index for row in layout.levels)
            block = _block_name(quantity.reference, names)
            for level in levels:
                geometry = self._geometry(layout, level)
                geometries[geometry.key] = geometry
                pieces, components = self._state(
                    block,
                    layout,
                    level,
                    collective=manifest.parallel_mode is ParallelMode.COLLECTIVE,
                )
                key = FieldKey(
                    quantity.reference,
                    self._owner.component_manifests[block].manifest_digest,
                    geometry.layout_identity,
                    level,
                    "accepted",
                )
                fields.append(FieldPayload(
                    key, "cell", "unspecified", components,
                    geometry.cell_shape,
                    pieces,
                ))
                keys.append(key)
        if not geometries:
            for quantity in manifest.quantities:
                layout = self._layout(quantity.layout_id)
                geometry = self._geometry(layout, 0)
                geometries[geometry.key] = geometry
        selected_diagnostics = tuple(
            value for value in diagnostics
            if value.key.layout_identity.token in {key[0] for key in geometries}
        )
        request = OutputRequest(
            manifest.qualified_id,
            tuple(keys),
            manifest.parallel_mode is ParallelMode.COLLECTIVE,
            tuple(value.key for value in selected_diagnostics),
        )
        engine = self._owner.native_executor
        run_identity = getattr(engine, "_last_run_identity", None)
        if type(run_identity) is not Identity:
            run_identity = make_identity("run", {
                "runtime": self._owner.runtime_plan.identity.token,
                "time": float(engine.time()).hex(),
                "macro_step": int(engine.macro_step()),
            })
        snapshot = OutputSnapshot(
            OutputClock.at("solution", engine.time(), engine.macro_step(), stage="accepted"),
            OutputProvenance(
                self._owner.install_plan.artifact.plan.plan_identity,
                self._owner.install_plan.bind_identity,
                run_identity,
                "runtime-instance-accepted-state",
            ),
            tuple(geometries.values()),
            tuple(fields),
            {
                "consumer_graph": self._owner.consumer_graph.identity.token,
                "runtime_plan": self._owner.runtime_plan.identity.token,
            },
            diagnostics=selected_diagnostics,
        )
        return snapshot, request


__all__ = ["RuntimeConsumerPublisher", "RuntimeOutputSnapshot"]
