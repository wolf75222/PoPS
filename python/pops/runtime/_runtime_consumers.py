"""Runtime-owned ConsumerGraph publication against accepted native state."""
from __future__ import annotations

import os
import tempfile
import math
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from pops.identity import Identity, make_identity
from pops.mesh._layout_plan_contracts import (
    CARTESIAN_CELL_AREA,
    POLAR_ANNULUS_CELL_AREA,
    NormalizedGeometry,
)
from pops.output.data import (
    _NATIVE_GEOMETRY_ARRAYS,
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
from pops.output._consumer_contracts import ConsumerKind, ParallelMode
from pops.output._writers.common import deterministic_target

from ._consumer import (
    AcceptedSideEffect,
    ConsumerPublisher,
    PreparedPublication,
    PublicationReceipt,
)
from ._component_execution_context import component_execution_data
from ._output_publisher import ConsumerOutputPublisher, OutputPreparation


def _block_name(reference: Any, names: tuple[str, ...]) -> str:
    block = getattr(reference, "block_ref", None)
    local_id = getattr(block, "local_id", None)
    if local_id in names:
        return local_id
    if len(names) == 1:
        return names[0]
    raise ValueError("consumer reference has no exact installed block owner")


def _conservative_names(owner: Any, block: str) -> tuple[str, ...]:
    """Read component order from the authenticated compiled artifact authority."""
    from pops.codegen._artifact_models import artifact_model_metadata

    rows = [
        row for row in artifact_model_metadata(owner._install_plan.artifact)
        if row.block_name == block
    ]
    if len(rows) != 1 or not rows[0].cons_names:
        raise ValueError("installed block %r has no exact conservative component order" % block)
    return rows[0].cons_names


def _identity_payload(value: Any, *, path: str = "layout") -> Any:
    """Project strict layout JSON into the float-free identity value language."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("%s contains a non-finite binary64 value" % path)
        return {"binary64": value.hex()}
    if isinstance(value, Mapping):
        return {
            key: _identity_payload(item, path="%s.%s" % (path, key))
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [
            _identity_payload(item, path="%s[%d]" % (path, index))
            for index, item in enumerate(value)
        ]
    return value


def _layout_identity(layout: Any) -> Identity:
    return make_identity("layout", _identity_payload(layout.to_data()))


_NATIVE_CELL_MEASURES = frozenset({
    CARTESIAN_CELL_AREA,
    POLAR_ANNULUS_CELL_AREA,
})


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


def _writer_snapshot_data(snapshot: OutputSnapshot, request: OutputRequest) -> dict[str, Any]:
    """Project the complete selected snapshot into the generated Writer POD vocabulary."""
    import numpy as np

    fields = snapshot.select(request)
    diagnostics = snapshot.select_diagnostics(request)
    geometry_keys = {
        (field.key.layout_identity.token, field.key.level) for field in fields
    }
    diagnostic_geometry_keys = {
        (diagnostic.key.layout_identity.token, diagnostic.key.level)
        for diagnostic in diagnostics
    }
    geometries = tuple(
        geometry for geometry in snapshot.geometries
        if geometry.key in geometry_keys
        or geometry.key in diagnostic_geometry_keys
    )
    if not geometries:
        raise ValueError("native Writer snapshot has no geometry for its exact selection")
    geometry_rows = []
    for geometry in geometries:
        dimension = len(geometry.cell_shape)
        patch_identity = make_identity("writer-geometry-domain", {
            "layout": geometry.layout_identity.token,
            "level": geometry.level,
            "boxes": [list(box) for box in geometry.boxes],
        }).token
        geometry_rows.append({
            "layout_identity": geometry.layout_identity.token,
            "layout_kind": geometry.layout_kind,
            "level": geometry.level,
            "dimension": dimension,
            "patch_identity": patch_identity,
            "origin": geometry.origin,
            "spacing": geometry.spacing,
            "cell_shape": geometry.cell_shape,
            "boxes": [
                {"lower": tuple(box[:dimension]), "upper": tuple(box[dimension:])}
                for box in geometry.boxes
            ],
            # LevelGeometry owns exact, immutable C-contiguous ABI buffers.  Keep the borrowed
            # arrays intact: the generated native marshaller validates dtype/shape again.
            "valid_cells": geometry.valid_cells,
            "coverage": geometry.coverage,
            "cell_volumes": geometry.cell_volumes,
        })
    field_rows = []
    for field in fields:
        # Serial Writer v1 receives every piece.  FieldPayload has already authenticated bounds,
        # dtype and non-overlap; the C++ Writer ABI additionally proves exact geometry coverage.
        # Densifying here only to repeat that proof was an O(N) allocation on every publication.
        pieces = []
        for piece in field.pieces:
            values = np.asarray(piece.values)
            if values.dtype != np.dtype(np.float64):
                raise TypeError("native Writer ABI v1 accepts only exact float64 field pieces")
            pieces.append({
                "lower": piece.lower,
                "upper": piece.upper,
                "patch_identity": make_identity("writer-field-piece", {
                    "field": field.key.identity.token,
                    "lower": list(piece.lower), "upper": list(piece.upper),
                }).token,
                "values": np.ascontiguousarray(values),
            })
        field_rows.append({
            "field_identity": field.key.identity.token,
            "reference_id": field.key.reference.qualified_id,
            "component_manifest_identity": field.key.component_manifest_identity.token,
            "layout_identity": field.key.layout_identity.token,
            "level": field.key.level,
            "state_id": field.key.state_id,
            "centering": field.centering,
            "units": field.units,
            "component_names": field.component_names,
            "dimension": len(field.global_shape),
            "global_shape": field.global_shape,
            "pieces": pieces,
        })
    diagnostic_rows = [{
        "diagnostic_identity": value.key.identity.token,
        "reference_id": value.key.reference.qualified_id,
        "component_manifest_identity": value.key.component_manifest_identity.token,
        "layout_identity": value.key.layout_identity.token,
        "level": value.key.level,
        "state_id": value.key.state_id,
        "reduction": value.key.reduction,
        "value": value.value,
        "units": value.units,
        "terms_json": json.dumps(
            {name: item.hex() for name, item in value.terms.items()},
            sort_keys=True, separators=(",", ":")),
    } for value in diagnostics]
    return {
        "geometries": geometry_rows,
        "fields": field_rows,
        "diagnostics": diagnostic_rows,
        "metadata_json": json.dumps(
            dict(snapshot.metadata), sort_keys=True, separators=(",", ":")),
        "selection_identity": request.identity.token,
    }


class _PreparedExternalWriter(PreparedPublication):
    """A verified native Writer temporary owned by one consumer transaction."""

    def __init__(self, effect: AcceptedSideEffect, preparation: OutputPreparation,
                 installed: Any, execution_context: Any) -> None:
        from pops.output.provider import consumer_format_data

        if preparation.request.consumer_id != effect.consumer_id:
            raise ValueError("native Writer request identity differs from its accepted effect")
        target_format = effect.target.output_format
        if not isinstance(target_format, Mapping):
            raise TypeError("accepted native Writer target must carry a format mapping")
        if consumer_format_data(
                preparation.format, where="resolved native Writer format") != dict(target_format):
            raise ValueError("resolved native Writer format differs from its accepted target")
        if effect.target.parallel_mode is not ParallelMode.SERIAL \
                or preparation.request.parallel:
            raise ValueError("native Writer ABI v1 requires one serial complete snapshot")
        self._effect = effect
        self._installed = installed
        self._target = Path(preparation.target)
        self._target.parent.mkdir(parents=True, exist_ok=True)
        if self._target.exists():
            raise FileExistsError("native Writer target collision: %s" % self._target)
        fd, temporary = tempfile.mkstemp(
            prefix=".%s." % self._target.name, suffix=".writer-stage",
            dir=str(self._target.parent))
        os.close(fd)
        os.unlink(temporary)
        self._temporary = Path(temporary)
        self._wire = _writer_snapshot_data(preparation.snapshot, preparation.request)
        self._execution = component_execution_data(execution_context)
        self._snapshot_identity = make_identity(
            "native-writer-snapshot", preparation.snapshot.to_data(preparation.request)).token
        clock = preparation.snapshot.clock
        if clock.stage != "accepted":
            raise ValueError("native Writer publishes only an accepted snapshot stage")
        self._request_data = {
            "snapshot": self._wire,
            "execution": self._execution,
            "temporary_path": str(self._temporary),
            "published_path": str(self._target),
            "snapshot_identity": self._snapshot_identity,
            "logical_time": {
                "clock_identity": clock.clock_id,
                "tick": clock.tick,
                "level": clock.level,
                "substep": clock.substep,
                "stage": clock.stage_index,
                "fraction_numerator": clock.fraction_numerator,
                "fraction_denominator": clock.fraction_denominator,
                "dt": float.fromhex(clock.dt_hex),
                "physical_time": float.fromhex(clock.time_hex),
            },
        }
        interface = installed.interface.to_data()
        self._interface_uri = interface["uri"]
        self._interface_version = interface["version"]
        self._published = False
        self._discarded = False
        try:
            receipt = self._invoke("verify")
            if not self._temporary.is_file():
                raise RuntimeError("native Writer verify did not create its exact temporary file")
            if receipt["bytes_written"] != self._temporary.stat().st_size:
                raise RuntimeError("native Writer verify receipt size differs from its temporary")
            if not receipt["content_digest"]:
                raise RuntimeError("native Writer verify returned no content digest")
            self._verified_receipt = dict(receipt)
        except BaseException:
            try:
                self._invoke("rollback")
            finally:
                self._temporary.unlink(missing_ok=True)
                self._target.unlink(missing_ok=True)
            raise

    @property
    def effect_identity(self) -> Identity:
        return self._effect.identity

    @property
    def payload_identity(self) -> Identity:
        return self._effect.payload.identity

    @property
    def temporary(self) -> Path:
        return self._temporary

    @property
    def target(self) -> Path:
        return self._target

    def _invoke(self, operation: str) -> Any:
        return self._installed.native_handle._invoke_component_operation(
            self._interface_uri, self._interface_version, operation,
            self._request_data)

    def publish(self) -> PublicationReceipt:
        if self._discarded:
            raise RuntimeError("discarded native Writer preparation cannot be published")
        if not self._published:
            if self._target.exists():
                raise FileExistsError("native Writer target collision: %s" % self._target)
            try:
                receipt = self._invoke("publish")
                self._published = self._target.is_file()
                if not self._published or self._temporary.exists():
                    raise RuntimeError(
                        "native Writer publish did not atomically consume its temporary")
                if dict(receipt) != self._verified_receipt:
                    raise RuntimeError(
                        "native Writer publish receipt differs from verified preparation")
            except BaseException:
                try:
                    self._invoke("rollback")
                finally:
                    self._temporary.unlink(missing_ok=True)
                    self._target.unlink(missing_ok=True)
                    self._published = False
                    self._discarded = True
                raise
        artifact = make_identity("native-writer-artifact", {
            "component_artifact": self._installed.artifact_identity.token,
            "snapshot": self._snapshot_identity,
            "target": str(self._target),
            "content_digest": self._verified_receipt["content_digest"],
        })
        return PublicationReceipt(
            self.effect_identity, self.payload_identity,
            "pops.output.external-writer.v1", artifact.token)

    def discard(self) -> None:
        if self._discarded:
            return
        if self._published:
            self.rollback()
            return
        try:
            self._invoke("discard")
        finally:
            self._temporary.unlink(missing_ok=True)
            self._discarded = True

    def rollback(self) -> None:
        if self._discarded:
            return
        try:
            self._invoke("rollback")
        finally:
            self._temporary.unlink(missing_ok=True)
            self._target.unlink(missing_ok=True)
            self._published = False
            self._discarded = True


class RuntimeConsumerPublisher(ConsumerPublisher):
    """One publisher for diagnostics, exact outputs, monitors and restart checkpoints."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner
        self._by_id = {row.qualified_id: row for row in owner._consumer_graph.nodes}
        self._pending: dict[str, tuple[DiagnosticPayload, ...]] = {}
        self._diagnostics: dict[str, DiagnosticPayload] = {}
        self._output = ConsumerOutputPublisher(self._resolve_output)
        self._external_writers: dict[str, Any] = {}
        exact_targets: dict[str, str] = {}
        from pops import interfaces
        for manifest in owner._consumer_graph.nodes:
            if manifest.kind is not ConsumerKind.SCIENTIFIC_OUTPUT:
                continue
            data = manifest.output_format_data
            if data.get("provider_id") != "pops.output.external-writer.v1":
                continue
            component_id = data.get("component_id")
            installed = owner._installed_components.get(component_id)
            if installed is None:
                raise ValueError(
                    "ScientificOutput names native Writer %r but that exact component is not "
                    "installed" % component_id)
            if installed.component_manifest.token != data.get("component_manifest_identity"):
                raise ValueError(
                    "ScientificOutput native Writer manifest identity differs from installation")
            if installed.interface != interfaces.Writer \
                    or dict(data.get("native_interface", {})) != interfaces.Writer.to_data():
                raise ValueError("ScientificOutput component does not implement exact Writer v1")
            if installed.native_handle is None:
                raise ValueError("ScientificOutput native Writer was installed but not loaded")
            target = Path(manifest.target_uri)
            canonical_target = target if target.suffix else target.with_suffix(data["extension"])
            collision_key = canonical_target.as_posix()
            previous = exact_targets.get(collision_key)
            if previous is not None:
                raise ValueError(
                    "two qualified native Writers select the same exact output target: %s "
                    "and %s" % (previous, manifest.qualified_id))
            exact_targets[collision_key] = manifest.qualified_id
            self._external_writers[manifest.qualified_id] = installed

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
        names = tuple(self._owner._component_manifests)
        values = []
        for quantity in manifest.quantities:
            block = _block_name(quantity.reference, names)
            engine = self._owner._executor_for_block(block)
            variables = _conservative_names(self._owner, block)
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
                self._owner._component_manifests[block].manifest_digest,
                self._owner.layout_identity(quantity.layout_id),
                0,
                "accepted",
                reduction,
            )
            values.append(DiagnosticPayload(key, cast(float, value), "unspecified", {}))
        return tuple(values)

    def _publish_diagnostics(self, effect: AcceptedSideEffect,
                             values: tuple[DiagnosticPayload, ...]) -> None:
        for value in values:
            self._diagnostics[value.key.identity.token] = value
            recorder = getattr(self._owner._executor, "record_program_diagnostic", None)
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
                    previous_value = previous[token]
                    if previous_value is None:
                        raise RuntimeError("diagnostic rollback lost its prior accepted payload")
                    self._diagnostics[token] = previous_value
                else:
                    self._diagnostics.pop(token, None)
            self._pending.pop(_effect.identity.token, None)

        self._pending[effect.identity.token] = values
        return _PreparedDiagnostic(
            effect, values, self._publish_diagnostics, self._discard_diagnostics, rollback)

    def _resolve_output(self, effect: AcceptedSideEffect) -> OutputPreparation:
        manifest = self._manifest(effect)
        snapshot, request = self._owner._output_snapshot(manifest, self.diagnostics)
        fmt = manifest.output_format
        target = _target(
            effect.target.uri, manifest.output_format_data, snapshot, request,
            manifest.handle.local_id, self._owner._output_root)
        communicator = self._owner._execution_context.communicator.handle \
            if effect.target.parallel_mode is not ParallelMode.SERIAL else None
        return OutputPreparation(fmt, snapshot, request, target, communicator)

    def prepare(self, effect: AcceptedSideEffect) -> PreparedPublication:
        if type(effect) is not AcceptedSideEffect:
            raise TypeError("RuntimeConsumerPublisher requires an exact AcceptedSideEffect")
        manifest = self._manifest(effect)
        if manifest.kind in (ConsumerKind.DIAGNOSTIC, ConsumerKind.MONITOR):
            return self._prepare_diagnostic(effect, manifest)
        if manifest.kind is ConsumerKind.SCIENTIFIC_OUTPUT:
            installed = self._external_writers.get(manifest.qualified_id)
            if installed is not None:
                return _PreparedExternalWriter(
                    effect, self._resolve_output(effect), installed,
                    self._owner._execution_context)
            return self._output.prepare(effect)
        if manifest.kind is ConsumerKind.CHECKPOINT:
            target = Path(effect.target.uri)
            if self._owner._output_root is not None:
                target = Path(self._owner._output_root) / target.name
            extension = manifest.operation_data["extension"]
            if target.suffix != extension:
                target = target.with_suffix(extension)
            return _PreparedCheckpoint(
                effect, self._owner, manifest.operation, target)
        raise TypeError("unsupported ConsumerKind %r" % manifest.kind)


class RuntimeOutputSnapshot:
    """Expose exact output values from one accepted RuntimeInstance snapshot."""

    def __init__(self, owner: Any) -> None:
        self._owner = owner
        self._geometry_cache: dict[tuple[str, int, int], LevelGeometry] = {}

    def _layout(self, layout_id: str) -> Any:
        rows = [row for row in self._owner._layout_plan.layouts
                if row.handle.qualified_id == layout_id]
        if len(rows) != 1:
            raise KeyError("consumer selected unknown layout %s" % layout_id)
        return rows[0]

    def _geometry(self, layout: Any, level: int) -> LevelGeometry:
        engine = self._owner._executor_for_layout(layout.handle.qualified_id)
        native_engine = getattr(engine, "_s", None)
        native_geometry = getattr(native_engine, "_output_geometry_snapshot", None)
        if not callable(native_geometry):
            raise RuntimeError(
                "scientific output requires the native output-geometry provider"
            )
        geometry = layout.geometry
        if type(geometry) is not NormalizedGeometry:
            raise TypeError("runtime output requires an exact normalized layout geometry")
        if geometry.dimension != 2:
            raise NotImplementedError(
                "the installed scientific-output provider supports rank-2 geometry; "
                "the normalized geometry has rank %d" % geometry.dimension)
        base_nx, base_ny = geometry.cells
        if int(engine.nx()) != base_nx:
            raise ValueError(
                "runtime x cell count does not match normalized layout geometry")
        if layout.adaptive:
            if base_nx != base_ny:
                raise NotImplementedError(
                    "adaptive scientific output requires the native square-grid geometry")
        elif int(engine.ny()) != base_ny:
            raise ValueError(
                "runtime y cell count does not match normalized layout geometry")
        scale = layout.levels[level].refinement
        nx, ny = base_nx * scale, base_ny * scale
        if geometry.cell_measure not in _NATIVE_CELL_MEASURES:
            raise NotImplementedError(
                "scientific output does not implement normalized cell measure %s"
                % geometry.cell_measure
            )
        epoch_provider = getattr(native_engine, "checkpoint_topology_epoch", None)
        topology_epoch = int(cast(Any, epoch_provider())) \
            if layout.adaptive and callable(epoch_provider) else 0
        layout_identity = _layout_identity(layout)
        cache_key = (layout_identity.token, level, topology_epoch)
        cached = self._geometry_cache.get(cache_key)
        if cached is not None:
            return cached
        spacing = (geometry.lengths[0] / nx, geometry.lengths[1] / ny)
        next_ratio = 0
        if layout.adaptive and level + 1 < len(layout.levels):
            next_ratio = (
                layout.levels[level + 1].refinement
                // layout.levels[level].refinement
            )
        if layout.adaptive:
            native = cast(Mapping[str, Any], native_geometry(
                level, geometry.lower, spacing, (ny, nx), next_ratio,
                geometry.cell_measure))
        else:
            native = cast(Mapping[str, Any], native_geometry(
                geometry.lower, spacing, (ny, nx), geometry.cell_measure))
        if int(native["topology_epoch"]) != topology_epoch:
            raise RuntimeError("native output geometry changed during snapshot construction")
        native_boxes = tuple(
            cast(tuple[int, int, int, int], tuple(int(item) for item in box))
            for box in native["boxes"]
        )
        result = LevelGeometry(
            layout_identity, "amr" if layout.adaptive else "uniform", level,
            cast(tuple[float, float], geometry.lower), spacing, (ny, nx),
            native_boxes,
            native["coverage"], native["cell_volumes"],
            coordinate_system=geometry.coordinate_system,
            cell_measure=geometry.cell_measure,
            axis_names=cast(tuple[str, str], geometry.axis_names),
            _native_valid_cells=native["valid_cells"],
            _native_arrays=_NATIVE_GEOMETRY_ARRAYS)
        # Retain only the current topology for this qualified level.  Regridding therefore cannot
        # grow the cache indefinitely, while every quantity in one accepted epoch shares buffers.
        for stale in tuple(self._geometry_cache):
            if stale[:2] == cache_key[:2] and stale != cache_key:
                del self._geometry_cache[stale]
        self._geometry_cache[cache_key] = result
        return result

    def _state(
        self, block: str, layout: Any, level: int, *, collective: bool
    ) -> tuple[tuple[ArrayPiece, ...], tuple[str, ...]]:
        import numpy as np

        engine = self._owner._executor_for_block(block)
        names = _conservative_names(self._owner, block)
        if layout.adaptive:
            communicator = self._owner._execution_context.communicator.handle
            size = 1 if communicator is None else int(communicator.Get_size())
            if collective and size > 1:
                raise NotImplementedError(
                    "collective adaptive output requires native rank-owned patch state; "
                    "the current AMR facade exposes only full-grid local/global buffers"
                )
            # A compiled Program can force the shared AmrRuntime engine for one
            # block, so block count is not a valid state-access discriminator.
            multi = bool(engine.uses_runtime_engine())
            getter = engine.block_level_state if multi else engine.level_state
            raw = getter(block, level) if multi else getter(level)
            n = int(engine.nx()) * layout.levels[level].refinement
            values = np.asarray(raw, dtype=np.float64).reshape(len(names), n, n)
            pieces = []
            for level_index, ilo, jlo, ihi, jhi in engine.patch_boxes():
                if int(level_index) != level:
                    continue
                lower = (int(jlo), int(ilo))
                upper = (int(jhi) + 1, int(ihi) + 1)
                pieces.append(ArrayPiece(
                    lower, upper,
                    np.ascontiguousarray(values[:, lower[0]:upper[0], lower[1]:upper[1]])))
            if not pieces:
                raise ValueError("selected adaptive field has no exact native patch pieces")
            return tuple(pieces), names
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

    def _field(
        self, reference: Any, layout: Any, level: int, *, collective: bool,
    ) -> tuple[tuple[ArrayPiece, ...], tuple[str, ...]]:
        """Read one resolved field by its authenticated native provider slot."""
        import numpy as np

        plans = self._owner._install_plan.artifact.plan.field_plans
        plan = plans.get(reference.local_id)
        if plan is None:
            raise ValueError(
                "scientific output field %r has no resolved install plan"
                % reference.local_id
            )
        engine = self._owner._executor_for_layout(layout.handle.qualified_id)
        if collective:
            communicator = self._owner._execution_context.communicator.handle
            size = 1 if communicator is None else int(communicator.Get_size())
            if size > 1:
                raise NotImplementedError(
                    "collective field output requires native rank-owned field patches"
                )
        slot = plan.native_options["provider_slot"]
        if layout.adaptive:
            raw = engine._s.field_potential_level_global(slot, level)
            n = int(engine.nx()) * layout.levels[level].refinement
        else:
            if level != 0:
                raise ValueError("uniform field output has only level zero")
            raw = engine._s.field_potential_global(slot)
            n = int(engine.nx())
        values = np.asarray(raw, dtype=np.float64).reshape(1, n, n)
        if layout.adaptive:
            pieces = []
            for level_index, ilo, jlo, ihi, jhi in engine.patch_boxes():
                if int(level_index) != level:
                    continue
                lower = (int(jlo), int(ilo))
                upper = (int(jhi) + 1, int(ihi) + 1)
                pieces.append(ArrayPiece(
                    lower, upper,
                    np.ascontiguousarray(values[:, lower[0]:upper[0], lower[1]:upper[1]])))
            if not pieces:
                raise ValueError("selected adaptive field has no exact native patch pieces")
        else:
            pieces = [ArrayPiece((0, 0), (n, n), values)]
        return tuple(pieces), (plan.operator.unknown.local_id,)

    def build(self, manifest: Any, diagnostics: tuple[DiagnosticPayload, ...]) \
            -> tuple[OutputSnapshot, OutputRequest]:
        geometries, fields, keys = {}, [], []
        names = tuple(self._owner._component_manifests)
        from pops.problem.handles import FieldHandle
        for quantity in manifest.quantities:
            layout = self._layout(quantity.layout_id)
            levels = quantity.levels or tuple(row.index for row in layout.levels)
            block = _block_name(quantity.reference, names)
            for level in levels:
                geometry = self._geometry(layout, level)
                geometries[geometry.key] = geometry
                if isinstance(quantity.reference, FieldHandle):
                    pieces, components = self._field(
                        quantity.reference,
                        layout,
                        level,
                        collective=manifest.parallel_mode is ParallelMode.COLLECTIVE,
                    )
                else:
                    pieces, components = self._state(
                        block,
                        layout,
                        level,
                        collective=manifest.parallel_mode is ParallelMode.COLLECTIVE,
                    )
                key = FieldKey(
                    quantity.reference,
                    self._owner._component_manifests[block].manifest_digest,
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
        engine = self._owner._executor
        logical_clock = manifest.schedule.domain.clock
        temporal = getattr(engine, "_temporal_restart_state", None)
        if temporal is None:
            raise RuntimeError("output snapshot requires accepted qualified temporal state")
        cursor = temporal.cursor_for_clock(logical_clock)
        last_dt_hex = temporal.controller_state.get("last_accepted_dt")
        accepted_dt = 0.0 if last_dt_hex is None else float.fromhex(last_dt_hex)
        run_identity = getattr(engine, "_last_run_identity", None)
        if type(run_identity) is not Identity:
            run_identity = make_identity("run", {
                "runtime": self._owner._runtime_plan.identity.token,
                "time": float(engine.time()).hex(),
                "macro_step": int(engine.macro_step()),
            })
        snapshot = OutputSnapshot(
            OutputClock.at(
                logical_clock.qualified_id, engine.time(), engine.macro_step(),
                stage="accepted", tick=int(cursor["tick"]), level=0, substep=0,
                stage_index=0, fraction=(1, 1), dt=accepted_dt),
            OutputProvenance(
                self._owner._install_plan.artifact.plan.plan_identity,
                self._owner._install_plan.bind_identity,
                run_identity,
                "runtime-instance-accepted-state",
            ),
            tuple(geometries.values()),
            tuple(fields),
            {
                "consumer_graph": self._owner._consumer_graph.identity.token,
                "runtime_plan": self._owner._runtime_plan.identity.token,
            },
            diagnostics=selected_diagnostics,
        )
        return snapshot, request


__all__ = ["RuntimeConsumerPublisher", "RuntimeOutputSnapshot"]
