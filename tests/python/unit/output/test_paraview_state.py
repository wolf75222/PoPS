"""Portable ParaView presentation state without a simulation-time ParaView dependency."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from pops.identity import make_identity
from pops.output.paraview_state import (
    MaterializedPVSM,
    PortableState,
    build_portable_paraview_state,
    materialize_paraview_state,
    read_portable_paraview_state,
    write_portable_paraview_state,
)


def _presentation(**updates):
    value = {
        "color_by": "U",
        "component": "rho",
        "color_map": "Viridis",
        "representation": "Surface With Edges",
        "show_scalar_bar": True,
    }
    value.update(updates)
    return value


def _arrays():
    return [
        {
            "name": "U",
            "type": "Float64",
            "components": 2,
            "component_names": ["rho", "energy"],
        },
        {
            "name": "temperature",
            "type": "Float32",
            "components": 1,
            "component_names": [],
        },
    ]


def _pvd(path: Path, *, identity=None) -> str:
    token = (
        make_identity("paraview-pvd", {"case": path.stem}).token
        if identity is None else identity
    )
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="Collection" version="0.1" '
        'pops_identity="%s"><Collection/></VTKFile>\n' % token,
        encoding="ascii",
    )
    return token


def _bundle(tmp_path: Path):
    root = tmp_path / "bundle"
    root.mkdir()
    pvd = root / "solution.pvd"
    _pvd(pvd)
    return write_portable_paraview_state(
        pvd,
        presentation=_presentation(),
        cell_arrays=_arrays(),
    )


def test_state_policies_are_immutable_and_canonical():
    portable = PortableState()
    assert portable.to_data() == {"schema_version": 1, "mode": "portable"}
    with pytest.raises((AttributeError, TypeError)):
        portable.mode = "changed"

    materialized = MaterializedPVSM("/opt/paraview/bin/pvpython")
    assert materialized.to_data() == {
        "schema_version": 1,
        "mode": "materialized_pvsm",
        "pvpython": "/opt/paraview/bin/pvpython",
    }
    with pytest.raises((AttributeError, TypeError)):
        materialized.pvpython = "changed"
    with pytest.raises(TypeError, match="canonical text"):
        MaterializedPVSM(" pvpython")


def test_portable_documents_are_deterministic_and_import_no_paraview_at_build_time():
    identity = make_identity("paraview-pvd", {"series": "portable"}).token
    arguments = {
        "pvd_file": "solution.pvd",
        "pvd_identity": identity,
        "presentation": _presentation(),
        "cell_arrays": _arrays(),
        "manifest_file": "solution.view.json",
        "script_file": "solution.view.py",
    }
    first = build_portable_paraview_state(**arguments)
    second = build_portable_paraview_state(**arguments)

    assert first == second
    compile(first.script, "solution.view.py", "exec")
    manifest = json.loads(first.manifest)
    assert manifest["identity"] == first.identity.token
    assert manifest["payload"]["pvd"] == {
        "file": "solution.pvd",
        "identity": identity,
    }
    assert manifest["payload"]["presentation"] == _presentation()
    assert first.manifest.endswith(b"\n")
    assert b"from paraview.simple import" in first.script
    assert b"Path(__file__).resolve().parent" in first.script
    assert b"/opt/" not in first.script


def test_portable_bundle_roundtrips_after_whole_directory_relocation(tmp_path):
    bundle = _bundle(tmp_path)
    assert bundle.manifest.name == "solution.view.json"
    assert bundle.script.name == "solution.view.py"
    assert bundle.pvd.name == "solution.pvd"
    assert bundle.presentation["color_by"] == "U"
    with pytest.raises(TypeError):
        bundle.presentation["color_by"] = "temperature"

    relocated_root = tmp_path / "relocated"
    shutil.copytree(bundle.manifest.parent, relocated_root)
    relocated = read_portable_paraview_state(relocated_root / bundle.manifest.name)

    assert relocated.identity == bundle.identity
    assert relocated.manifest.parent == relocated_root.resolve()
    assert relocated.script.parent == relocated_root.resolve()
    assert relocated.pvd.parent == relocated_root.resolve()
    script = relocated.script.read_text(encoding="utf-8")
    assert str(bundle.manifest.parent) not in script


def test_portable_bundle_rejects_tampering_and_noncanonical_json(tmp_path):
    bundle = _bundle(tmp_path)
    bundle.script.write_text(
        bundle.script.read_text(encoding="utf-8") + "# tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="script differs"):
        read_portable_paraview_state(bundle.manifest)

    other = tmp_path / "other"
    other.mkdir()
    pvd = other / "solution.pvd"
    _pvd(pvd)
    bundle = write_portable_paraview_state(
        pvd, presentation=_presentation(), cell_arrays=_arrays())
    decoded = json.loads(bundle.manifest.read_bytes())
    bundle.manifest.write_text(json.dumps(decoded, indent=2), encoding="ascii")
    with pytest.raises(ValueError, match="not canonical"):
        read_portable_paraview_state(bundle.manifest)

    third = tmp_path / "third"
    third.mkdir()
    pvd = third / "solution.pvd"
    _pvd(pvd)
    bundle = write_portable_paraview_state(
        pvd, presentation=_presentation(), cell_arrays=_arrays())
    _pvd(bundle.pvd, identity=make_identity("paraview-pvd", {"other": True}).token)
    with pytest.raises(ValueError, match="differs"):
        read_portable_paraview_state(bundle.manifest)


def test_portable_recipe_validates_field_component_and_local_names():
    identity = make_identity("paraview-pvd", {"series": "validation"}).token
    base = {
        "pvd_file": "solution.pvd",
        "pvd_identity": identity,
        "cell_arrays": _arrays(),
        "manifest_file": "solution.view.json",
        "script_file": "solution.view.py",
    }
    with pytest.raises(ValueError, match="color_by"):
        build_portable_paraview_state(
            **base, presentation=_presentation(color_by="missing"))
    with pytest.raises(ValueError, match="component"):
        build_portable_paraview_state(
            **base, presentation=_presentation(component="missing"))
    with pytest.raises(ValueError, match="one local"):
        build_portable_paraview_state(
            **dict(base, pvd_file="nested/solution.pvd"),
            presentation=_presentation(),
        )


def _fake_pvpython(path: Path) -> Path:
    path.write_text(
        """#!/usr/bin/env python3
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

log = Path(__file__).with_suffix('.log')
if sys.argv[1:] == ['--version']:
    log.write_text('version\\n', encoding='utf-8')
    print('ParaView 6.1.1')
elif len(sys.argv) == 4 and sys.argv[2] == '--save-state':
    script = Path(sys.argv[1])
    state = Path(sys.argv[3])
    assert script.is_file()
    assert script.with_suffix('.json').is_file()
    assert (script.parent / 'solution.pvd').is_file()
    state.write_text(
        '<GenericParaViewApplication><ServerManagerState version="6.1.1"/>'
        '</GenericParaViewApplication>',
        encoding='utf-8',
    )
    with log.open('a', encoding='utf-8') as stream:
        stream.write('materialize\\n')
elif len(sys.argv) == 5 and sys.argv[1] == '-c' and 'LoadState' in sys.argv[2]:
    state, data_directory = Path(sys.argv[3]), Path(sys.argv[4])
    assert ET.parse(state).getroot().find('./ServerManagerState') is not None
    assert (data_directory / 'solution.pvd').is_file()
    with log.open('a', encoding='utf-8') as stream:
        stream.write('load\\n')
else:
    raise SystemExit('unexpected fake pvpython invocation: %r' % (sys.argv,))
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def test_materialize_uses_real_process_boundary_and_reloads_generated_state(tmp_path):
    bundle = _bundle(tmp_path)
    executable = _fake_pvpython(tmp_path / "pvpython")
    target = tmp_path / "render" / "solution.pvsm"

    result = materialize_paraview_state(
        bundle.manifest,
        configuration=MaterializedPVSM(str(executable)),
        target=target,
    )

    assert result == target.resolve()
    assert result.is_file()
    assert executable.with_suffix(".log").read_text(encoding="utf-8").splitlines() == [
        "version", "materialize", "load",
    ]


def test_materialize_fails_closed_without_pvpython_or_on_target_collision(tmp_path):
    bundle = _bundle(tmp_path)
    missing = tmp_path / "missing-pvpython"
    with pytest.raises(RuntimeError, match="requires an executable pvpython"):
        materialize_paraview_state(bundle, pvpython=str(missing))

    executable = _fake_pvpython(tmp_path / "pvpython")
    target = tmp_path / "occupied.pvsm"
    target.write_text("not a state", encoding="utf-8")
    with pytest.raises(FileExistsError, match="collision"):
        materialize_paraview_state(bundle, pvpython=str(executable), target=target)


def test_write_refuses_to_split_the_relocatable_bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    pvd = root / "solution.pvd"
    _pvd(pvd)
    with pytest.raises(ValueError, match="next to its PVD"):
        write_portable_paraview_state(
            pvd,
            presentation=_presentation(),
            cell_arrays=_arrays(),
            target=tmp_path / "elsewhere" / "solution.view.json",
        )
