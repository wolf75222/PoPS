"""Portable ParaView presentation recipes and explicit PVSM materialization.

The portable artifact is deliberately not a hand-written ``.pvsm``.  It is a canonical JSON
recipe plus a deterministic Python script which resolves its PVD relative to the script location.
Creating that pair requires only the Python standard library.  A real ParaView installation is
used only when :func:`materialize_paraview_state` is explicitly asked to call ``pvpython``.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar
from xml.etree import ElementTree as ET

from pops._manifest_protocol import strict_json_loads
from pops.identity import Identity, make_identity


_KIND = "pops-paraview-portable-state"
_SCHEMA_VERSION = 1
_PRESENTATION_KEYS = frozenset({
    "color_by", "component", "color_map", "representation", "show_scalar_bar",
})
_REPRESENTATIONS = frozenset({
    "Surface", "Surface With Edges", "Wireframe", "Points",
})
_ARRAY_KEYS = frozenset({"name", "type", "components", "component_names"})
_MANIFEST_KEYS = frozenset({
    "schema_version", "kind", "identity", "payload_sha256", "script_sha256", "payload",
})
_PAYLOAD_KEYS = frozenset({"pvd", "presentation", "cell_arrays", "script"})


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TypeError("%s must be non-empty canonical text" % where)
    return value


def _optional_text(value: Any, where: str) -> str | None:
    if value is None:
        return None
    return _text(value, where)


def _basename(value: Any, *, where: str, suffix: str) -> str:
    text = _text(value, where)
    path = Path(text)
    if path.name != text or text in {".", ".."} or path.suffix != suffix:
        raise ValueError("%s must be one local %s filename" % (where, suffix))
    return text


def _identity_token(value: Any, *, domain: str, where: str) -> str:
    try:
        identity = Identity.from_token(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be a valid PoPS identity token" % where) from exc
    if identity.domain != domain:
        raise ValueError("%s must use identity domain %r" % (where, domain))
    return identity.token


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _presentation(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _PRESENTATION_KEYS:
        raise TypeError(
            "portable ParaView presentation must contain exactly %s"
            % sorted(_PRESENTATION_KEYS)
        )
    color_by = _text(value["color_by"], "portable presentation color_by")
    component = _optional_text(value["component"], "portable presentation component")
    color_map = _text(value["color_map"], "portable presentation color_map")
    representation = _text(
        value["representation"], "portable presentation representation")
    if representation not in _REPRESENTATIONS:
        raise ValueError(
            "portable presentation representation must be one of %s"
            % sorted(_REPRESENTATIONS)
        )
    show_scalar_bar = value["show_scalar_bar"]
    if type(show_scalar_bar) is not bool:
        raise TypeError("portable presentation show_scalar_bar must be an exact bool")
    return {
        "color_by": color_by,
        "component": component,
        "color_map": color_map,
        "representation": representation,
        "show_scalar_bar": show_scalar_bar,
    }


def _cell_arrays(value: Any, presentation: Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("portable ParaView cell_arrays must be a sequence")
    result = []
    for index, row in enumerate(value):
        where = "portable cell_arrays[%d]" % index
        if not isinstance(row, Mapping) or set(row) != _ARRAY_KEYS:
            raise TypeError("%s must contain exactly %s" % (where, sorted(_ARRAY_KEYS)))
        name = _text(row["name"], where + " name")
        vtk_type = _text(row["type"], where + " type")
        components = row["components"]
        if isinstance(components, bool) or type(components) is not int or components < 1:
            raise TypeError("%s components must be an integer >= 1" % where)
        raw_names = row["component_names"]
        if isinstance(raw_names, (str, bytes, bytearray)) \
                or not isinstance(raw_names, Sequence):
            raise TypeError("%s component_names must be a sequence" % where)
        component_names = [
            _text(item, "%s component_names[%d]" % (where, item_index))
            for item_index, item in enumerate(raw_names)
        ]
        if component_names and len(component_names) != components:
            raise ValueError(
                "%s component_names must be empty or name every component" % where)
        if len(component_names) != len(set(component_names)):
            raise ValueError("%s component_names contains duplicates" % where)
        result.append({
            "name": name,
            "type": vtk_type,
            "components": components,
            "component_names": component_names,
        })
    if not result:
        raise ValueError("portable ParaView state requires at least one CellData array")
    names = [row["name"] for row in result]
    if len(names) != len(set(names)):
        raise ValueError("portable ParaView CellData array names must be unique")
    selected = [row for row in result if row["name"] == presentation["color_by"]]
    if len(selected) != 1:
        raise ValueError("portable presentation color_by does not name one CellData array")
    component = presentation["component"]
    if component is not None and component not in selected[0]["component_names"]:
        raise ValueError(
            "portable presentation component is not declared by the selected array")
    return result


@dataclass(frozen=True, slots=True)
class PortableState:
    """Request a relocatable JSON/Python presentation recipe, with no ParaView dependency."""

    __pops_ir_immutable__: ClassVar[bool] = True

    def to_data(self) -> dict[str, Any]:
        return {"schema_version": 1, "mode": "portable"}


@dataclass(frozen=True, slots=True)
class MaterializedPVSM:
    """Request an actual PVSM from an explicitly selected real ``pvpython`` executable."""

    pvpython: str | None = None
    __pops_ir_immutable__: ClassVar[bool] = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "pvpython", _optional_text(
            self.pvpython, "MaterializedPVSM.pvpython"))

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "mode": "materialized_pvsm",
            "pvpython": self.pvpython,
        }


@dataclass(frozen=True, slots=True)
class PortableStateDocuments:
    """Pure deterministic bytes that a transactional writer can stage itself."""

    identity: Identity
    manifest: bytes
    script: bytes
    manifest_file: str
    script_file: str
    pvd_file: str

    def __post_init__(self) -> None:
        if type(self.identity) is not Identity \
                or self.identity.domain != "paraview-portable-state":
            raise TypeError("portable state documents require their exact identity")
        for name in ("manifest", "script"):
            if not isinstance(getattr(self, name), bytes) or not getattr(self, name):
                raise TypeError("portable state %s must be non-empty bytes" % name)
        _basename(self.manifest_file, where="portable manifest_file", suffix=".json")
        _basename(self.script_file, where="portable script_file", suffix=".py")
        _basename(self.pvd_file, where="portable pvd_file", suffix=".pvd")


@dataclass(frozen=True, slots=True)
class PortableStateBundle:
    """Authenticated paths for one relocatable portable-state bundle."""

    manifest: Path
    script: Path
    pvd: Path
    identity: Identity
    presentation: Mapping[str, Any]
    cell_arrays: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        for name in ("manifest", "script", "pvd"):
            value = getattr(self, name)
            if not isinstance(value, Path) or not value.is_absolute():
                raise TypeError("PortableStateBundle.%s must be an absolute Path" % name)
        if len({self.manifest.parent, self.script.parent, self.pvd.parent}) != 1:
            raise ValueError("portable state bundle files must share one directory")
        if type(self.identity) is not Identity \
                or self.identity.domain != "paraview-portable-state":
            raise TypeError("PortableStateBundle.identity has the wrong domain")
        object.__setattr__(self, "presentation", MappingProxyType(dict(self.presentation)))
        object.__setattr__(self, "cell_arrays", tuple(
            MappingProxyType(dict(row)) for row in self.cell_arrays))


def _render_script(
    *, manifest_file: str, identity: str, payload_sha256: str,
) -> bytes:
    manifest_literal = json.dumps(manifest_file, ensure_ascii=True)
    identity_literal = json.dumps(identity, ensure_ascii=True)
    digest_literal = json.dumps(payload_sha256, ensure_ascii=True)
    source = f'''#!/usr/bin/env pvpython
"""Open this relocatable PoPS ParaView bundle and apply its presentation recipe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET


_MANIFEST_FILE = {manifest_literal}
_EXPECTED_IDENTITY = {identity_literal}
_EXPECTED_PAYLOAD_SHA256 = {digest_literal}


def _canonical_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\\n").encode("ascii")


def _load_recipe():
    root = Path(__file__).resolve().parent
    manifest_path = root / _MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    if manifest.get("schema_version") != 1 or manifest.get("kind") != {_KIND!r}:
        raise RuntimeError("unsupported PoPS portable ParaView recipe")
    if manifest.get("identity") != _EXPECTED_IDENTITY:
        raise RuntimeError("portable ParaView recipe identity changed")
    payload = manifest.get("payload")
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    if digest != _EXPECTED_PAYLOAD_SHA256 or manifest.get("payload_sha256") != digest:
        raise RuntimeError("portable ParaView recipe payload failed authentication")
    pvd = root / payload["pvd"]["file"]
    if not pvd.is_file():
        raise FileNotFoundError("portable ParaView PVD is missing: " + str(pvd))
    xml = ET.parse(pvd).getroot()
    if xml.get("type") != "Collection" or xml.get("pops_identity") != payload["pvd"]["identity"]:
        raise RuntimeError("portable ParaView PVD identity differs from its recipe")
    return payload, pvd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-state", metavar="OUTPUT_PVSM")
    args = parser.parse_args()
    payload, pvd = _load_recipe()

    from paraview.simple import (
        ColorBy, GetActiveViewOrCreate, GetAnimationScene, GetColorTransferFunction,
        OpenDataFile, Render, ResetCamera, SaveState, Show,
    )

    config = payload["presentation"]
    reader = OpenDataFile(str(pvd))
    if reader is None:
        raise RuntimeError("ParaView could not open " + str(pvd))
    reader.UpdatePipeline()
    view = GetActiveViewOrCreate("RenderView")
    display = Show(reader, view)
    scene = GetAnimationScene()
    scene.UpdateAnimationUsingDataTimeSteps()
    scene.GoToLast()
    timesteps = list(getattr(reader, "TimestepValues", ()))
    if timesteps:
        scene.AnimationTime = timesteps[-1]
        reader.UpdatePipeline(time=timesteps[-1])
    display.SetRepresentationType(config["representation"])
    color = ("CELLS", config["color_by"])
    if config["component"] is not None:
        color = color + (config["component"],)
    ColorBy(display, color)
    lut = GetColorTransferFunction(config["color_by"])
    if lut.ApplyPreset(config["color_map"], True) is False:
        raise RuntimeError("unknown ParaView color preset: " + config["color_map"])
    display.RescaleTransferFunctionToDataRange(True)
    display.SetScalarBarVisibility(view, config["show_scalar_bar"])
    ResetCamera(view)
    Render(view)
    if args.save_state is not None:
        target = Path(args.save_state).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        SaveState(str(target))


if __name__ == "__main__":
    main()
'''
    result = source.encode("utf-8")
    compile(result, "<pops-portable-paraview-state>", "exec")
    return result


def build_portable_paraview_state(
    *,
    pvd_file: str,
    pvd_identity: str,
    presentation: Mapping[str, Any],
    cell_arrays: Sequence[Mapping[str, Any]],
    manifest_file: str,
    script_file: str,
) -> PortableStateDocuments:
    """Build canonical recipe/script bytes without touching the filesystem or importing ParaView."""

    checked_pvd = _basename(pvd_file, where="portable pvd_file", suffix=".pvd")
    checked_manifest = _basename(
        manifest_file, where="portable manifest_file", suffix=".json")
    checked_script = _basename(script_file, where="portable script_file", suffix=".py")
    if len({checked_pvd, checked_manifest, checked_script}) != 3:
        raise ValueError("portable PVD, manifest and script filenames must be distinct")
    checked_identity = _identity_token(
        pvd_identity, domain="paraview-pvd", where="portable pvd_identity")
    checked_presentation = _presentation(presentation)
    checked_arrays = _cell_arrays(cell_arrays, checked_presentation)
    payload = {
        "pvd": {"file": checked_pvd, "identity": checked_identity},
        "presentation": checked_presentation,
        "cell_arrays": checked_arrays,
        "script": {"file": checked_script},
    }
    identity = make_identity("paraview-portable-state", payload)
    payload_sha256 = _sha256(_json_bytes(payload))
    script = _render_script(
        manifest_file=checked_manifest,
        identity=identity.token,
        payload_sha256=payload_sha256,
    )
    manifest = _json_bytes({
        "schema_version": _SCHEMA_VERSION,
        "kind": _KIND,
        "identity": identity.token,
        "payload_sha256": payload_sha256,
        "script_sha256": _sha256(script),
        "payload": payload,
    })
    return PortableStateDocuments(
        identity, manifest, script, checked_manifest, checked_script, checked_pvd)


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".pops-paraview-state-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FileExistsError(
                    "portable ParaView state collision at %s" % path) from None
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)


def _pvd_identity(path: Path) -> str:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ValueError("portable ParaView PVD is not readable XML: %s" % path) from exc
    if root.tag != "VTKFile" or root.attrib.get("type") != "Collection":
        raise ValueError("portable ParaView input must be a VTK Collection PVD")
    return _identity_token(
        root.attrib.get("pops_identity"),
        domain="paraview-pvd",
        where="portable PVD pops_identity",
    )


def write_portable_paraview_state(
    pvd: os.PathLike[str] | str,
    *,
    presentation: Mapping[str, Any],
    cell_arrays: Sequence[Mapping[str, Any]],
    target: os.PathLike[str] | str | None = None,
) -> PortableStateBundle:
    """Write a canonical recipe and script next to an already-published PVD."""

    pvd_path = Path(pvd).expanduser().resolve()
    if pvd_path.suffix != ".pvd" or not pvd_path.is_file():
        raise FileNotFoundError("portable ParaView input PVD does not exist: %s" % pvd_path)
    identity = _pvd_identity(pvd_path)
    manifest = (
        pvd_path.with_suffix(".view.json")
        if target is None else Path(target).expanduser().resolve()
    )
    if manifest.suffix != ".json":
        raise ValueError("portable ParaView target must end in .json")
    if manifest.parent != pvd_path.parent:
        raise ValueError("portable ParaView recipe must be written next to its PVD")
    script = manifest.with_suffix(".py")
    documents = build_portable_paraview_state(
        pvd_file=pvd_path.name,
        pvd_identity=identity,
        presentation=presentation,
        cell_arrays=cell_arrays,
        manifest_file=manifest.name,
        script_file=script.name,
    )
    # The manifest is the bundle entrypoint, so publish it only after its referenced script.
    _publish_exact(script, documents.script)
    _publish_exact(manifest, documents.manifest)
    return read_portable_paraview_state(manifest)


def read_portable_paraview_state(
    manifest: os.PathLike[str] | str,
) -> PortableStateBundle:
    """Authenticate a portable recipe, its exact generated script and its colocated PVD."""

    manifest_path = Path(manifest).expanduser().resolve()
    raw = manifest_path.read_bytes()
    decoded = strict_json_loads(raw, where="portable ParaView state JSON")
    if not isinstance(decoded, Mapping) or set(decoded) != _MANIFEST_KEYS:
        raise ValueError("portable ParaView state has an unsupported manifest schema")
    if decoded["schema_version"] != _SCHEMA_VERSION or decoded["kind"] != _KIND:
        raise ValueError("portable ParaView state has an unsupported schema version or kind")
    if _json_bytes(dict(decoded)) != raw:
        raise ValueError("portable ParaView state JSON is not canonical")
    payload = decoded["payload"]
    if not isinstance(payload, Mapping) or set(payload) != _PAYLOAD_KEYS:
        raise ValueError("portable ParaView state has an unsupported payload schema")
    pvd_row, script_row = payload["pvd"], payload["script"]
    if not isinstance(pvd_row, Mapping) or set(pvd_row) != {"file", "identity"}:
        raise ValueError("portable ParaView state has an invalid PVD reference")
    if not isinstance(script_row, Mapping) or set(script_row) != {"file"}:
        raise ValueError("portable ParaView state has an invalid script reference")
    documents = build_portable_paraview_state(
        pvd_file=pvd_row["file"],
        pvd_identity=pvd_row["identity"],
        presentation=payload["presentation"],
        cell_arrays=payload["cell_arrays"],
        manifest_file=manifest_path.name,
        script_file=script_row["file"],
    )
    if documents.manifest != raw or decoded["identity"] != documents.identity.token:
        raise ValueError("portable ParaView state identity or manifest differs from its payload")
    script_path = manifest_path.parent / documents.script_file
    if script_path.read_bytes() != documents.script:
        raise ValueError("portable ParaView script differs from its authenticated recipe")
    if _sha256(documents.script) != decoded["script_sha256"]:
        raise ValueError("portable ParaView script digest differs from its manifest")
    pvd_path = manifest_path.parent / documents.pvd_file
    if _pvd_identity(pvd_path) != pvd_row["identity"]:
        raise ValueError("portable ParaView PVD differs from its authenticated recipe")
    return PortableStateBundle(
        manifest_path,
        script_path.resolve(),
        pvd_path.resolve(),
        documents.identity,
        payload["presentation"],
        tuple(payload["cell_arrays"]),
    )


def _pvpython(value: str | None) -> Path:
    candidate = shutil.which("pvpython" if value is None else value)
    if candidate is None and value is not None:
        path = Path(value).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            candidate = str(path.resolve())
    if candidate is None:
        raise RuntimeError(
            "materializing a PVSM requires an executable pvpython")
    return Path(candidate).resolve()


def _run(command: list[str], *, timeout: int, phase: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        stdout = getattr(exc, "stdout", None) or getattr(exc, "output", None) or ""
        stderr = getattr(exc, "stderr", None) or ""
        detail = "\n".join(
            item.strip() for item in (stdout, stderr) if isinstance(item, str) and item.strip())
        raise RuntimeError(
            "%s failed%s" % (phase, "\n" + detail if detail else "")) from exc


def _require_pvsm(path: Path) -> None:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ValueError("pvpython produced an invalid PVSM document") from exc
    if root.tag == "ServerManagerState":
        return
    children = root.findall("./ServerManagerState")
    if root.tag == "GenericParaViewApplication" and len(children) == 1:
        return
    raise ValueError("pvpython produced an unexpected PVSM state document")


_LOAD_STATE_SCRIPT = r'''
import sys
from paraview.simple import LoadState
LoadState(sys.argv[1], data_directory=sys.argv[2], restrict_to_data_directory=True)
'''


def materialize_paraview_state(
    recipe: PortableStateBundle | os.PathLike[str] | str,
    *,
    configuration: MaterializedPVSM | None = None,
    pvpython: str | None = None,
    target: os.PathLike[str] | str | None = None,
    timeout: int = 120,
) -> Path:
    """Create and reload a real PVSM from an authenticated portable recipe."""

    if type(configuration) not in {type(None), MaterializedPVSM}:
        raise TypeError("configuration must be an exact MaterializedPVSM or None")
    if configuration is not None and pvpython is not None:
        raise ValueError("select pvpython either directly or through MaterializedPVSM")
    selected = configuration.pvpython if configuration is not None else pvpython
    selected = _optional_text(selected, "materialize_paraview_state pvpython")
    if isinstance(timeout, bool) or type(timeout) is not int or timeout < 1:
        raise ValueError("materialize_paraview_state timeout must be an integer >= 1")
    bundle = (
        recipe if type(recipe) is PortableStateBundle
        else read_portable_paraview_state(recipe)
    )
    if type(bundle) is not PortableStateBundle:
        raise TypeError("recipe must be an exact PortableStateBundle or manifest path")
    executable = _pvpython(selected)
    version = _run(
        [str(executable), "--version"],
        timeout=min(timeout, 15),
        phase="pvpython version preflight",
    )
    if not (version.stdout or version.stderr).strip():
        raise RuntimeError("pvpython version preflight returned no version evidence")
    output = (
        bundle.pvd.with_suffix(".pvsm")
        if target is None else Path(target).expanduser().resolve()
    )
    if output.suffix != ".pvsm":
        raise ValueError("materialized ParaView state target must end in .pvsm")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix=".pops-materialized-pvsm-", dir=output.parent) as work:
        generated = Path(work) / output.name
        _run(
            [str(executable), str(bundle.script), "--save-state", str(generated)],
            timeout=timeout,
            phase="pvpython portable-state materialization",
        )
        if not generated.is_file():
            raise RuntimeError("pvpython did not create the requested PVSM")
        _require_pvsm(generated)
        _run(
            [
                str(executable), "-c", _LOAD_STATE_SCRIPT,
                str(generated), str(bundle.manifest.parent),
            ],
            timeout=timeout,
            phase="pvpython generated-state reload",
        )
        payload = generated.read_bytes()
    _publish_exact(output, payload)
    _require_pvsm(output)
    return output


__all__ = [
    "MaterializedPVSM",
    "PortableState",
    "PortableStateBundle",
    "PortableStateDocuments",
    "build_portable_paraview_state",
    "materialize_paraview_state",
    "read_portable_paraview_state",
    "write_portable_paraview_state",
]
