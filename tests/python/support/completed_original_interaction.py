"""IR19 fixture: original Stage physics plus a completed-source observation.

No interaction array getter exists: saved I arrays are COMPUTED_REFERENCE only.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.evolved_stage_amr import (
    ACCEPTANCE, CONTROLS, build as original_build, check_saved,
)
from tests.python.integration.runtime.test_public_evolved_stage_amr import capture as original_capture
from tests.python.integration.runtime.test_public_captured_diffusion import bounded_bytes
from tests.python.support.spatial_interaction_receipts import (
    _allgather_array_values, array_wire_decode, array_wire_encode,
)

CELLS = 8
WORKSPACE = 32 * 1024**2
SCHEMA = "pops.completed-original-interaction-native-fixture@1"
KINDS = ("sum", "abs_sum", "min", "max")


def build(width):
    """Reuse the original physical equations, seven controls and partial layout."""
    from pops.fields import CellMidpoint, CellVolumeMeasure, DirectSpatialInteraction, SpatialInteractionKernel
    from pops.model import FieldSpace
    from pops.time._program.serialization import _json_ready

    case, layout = original_build(CELLS, width)
    program = case._time_registry.program
    name = "T0" if width == 1 else "T1"
    keeper = next(node for node in program._values if node.op == "store_history" and node.name == name)
    source, storage = keeper.inputs[0], keeper.block
    owner = next(time for time in program._time_states.values() if time.block is storage)
    output = FieldSpace("completed-original-map", components=("I",), sampling="cell_center",
        frame=owner.n.space.frame, support=owner.n.space.support, clock=owner.n.space.clock)
    interaction = program.spatial_interaction(source,
        SpatialInteractionKernel(2, lambda x, y: .25+x[0]*y[1]-2*y[0]+.5*x[1]),
        output_space=output, measure=CellVolumeMeasure(), quadrature=CellMidpoint(),
        realization=DirectSpatialInteraction(WORKSPACE), source_scope="completed_original", owner_block=storage)
    for kind, reduced in (("sum", program.sum_component(interaction, 0)),
            ("abs_sum", program.abs_sum_component(interaction, 0)),
            ("min", program.min(interaction)), ("max", program.max(interaction))):
        program.record_scalar("ir19."+name+"."+kind, reduced)
    assert source.block is source.space is source.state_ref is None
    assert interaction.block is interaction.state_ref is None
    return case, layout, {"source_name":name, "source_ssa_name":source.name, "tuple_component":source.attrs["component"],
        "storage":storage.qualified_id, "source_value_id":source.id, "interaction_value_id":interaction.id,
        "binding":_json_ready(interaction.attrs["closed_field_source"]),
        "source_point":source.point.to_data(), "kernel":"0.25+x0*y1-2*y0+0.5*x1",
        "rate_physics":"original helper unchanged; interaction does not enter F",
        "evolved_partition":"T1,T0" if width == 2 else "T0"}


def digest(path):
    return hashlib.sha256(bounded_bytes(path)).hexdigest()


def capture(world, runtime, width, *, initial=False):
    rows, original_metadata = original_capture(world, runtime, width, histories=not initial)
    native = runtime._executor
    count = len(rows)
    with collective_check(world):
        assert tuple(runtime.spatial_shape()) == (CELLS,CELLS) and count == 2
    geometry, owners = [], []
    for level in range(count):
        shape = (CELLS*2**level,)*2
        spacing = (1./shape[0],)*2
        raw = collective_call(world, lambda level=level, shape=shape, spacing=spacing:
            native._output_geometry_snapshot(level, (0.,0.), spacing, shape,
                (2,2) if level+1 < count else (0,0), "pops://cell-measures/cartesian-area@1"))
        with collective_check(world):
            valid = np.asarray(raw["valid_cells"], dtype=np.bool_)
            coverage = np.asarray(raw["coverage"], dtype=np.bool_)
            np.testing.assert_array_equal(rows[level]["active"], valid & ~coverage)
            y, x = np.meshgrid((np.arange(shape[1])+.5)*spacing[1],
                (np.arange(shape[0])+.5)*spacing[0], indexing="ij")
            raw = dict(raw)
            raw.update(origin=(0.,0.), spacing=spacing,
                coordinate_authority="derived-centers-from-bound-native-Cartesian-metadata",
                centers=np.stack((x,y)), cell_measure="pops://cell-measures/cartesian-area@1")
            geometry.append(raw)
        pieces = {}
        for name in (*tuple("Q%d" % index for index in range(width)), "forcing"):
            pieces[name] = collective_call(world, lambda name=name, level=level:
                native.output_state_local_pieces(name, level))
        owners.append(_allgather_array_values(world, pieces))
        if not initial:
            for name in (*tuple("T%d" % index for index in range(width)), *(("z",) if width == 2 else ())):
                for slot in (0,1):
                    rows[level]["raw_"+name+"_slot%d" % slot] = np.asarray(collective_call(world,
                        lambda name=name, level=level, slot=slot: native.history_global(name, level, slot))).copy()
    auxiliary = collective_call(world, lambda: tuple(bytes(part) for part in
        native.capture_auxiliary_checkpoint_accepted_state()))
    return {"rows":rows, "metadata":original_metadata, "geometry":geometry,
        "owners":owners, "rank_auxiliary":_allgather_array_values(world, auxiliary),
        "epoch":collective_call(world, native.checkpoint_topology_epoch),
        "volume_fraction_authority":"null-original-provider-mask; non-EB geometry (no kappa getter/array)",
        "source_role":"UNAVAILABLE_BEFORE_SOLVE" if initial else "NATIVE_GLOBAL_FIELD_HISTORY_SAME_ISSUED_COMPONENT"}


def same(left, right):
    assert array_wire_encode(left) == array_wire_encode(right)


def reference(image, source_name):
    """Independent midpoint integral from actual saved composite T and measures."""
    sources = []
    for row, geometry in zip(image["rows"], image["geometry"], strict=True):
        active = np.asarray(row["active"])
        centers = np.asarray(geometry["centers"])
        values = np.asarray(row[source_name]).reshape(active.shape)
        volumes = np.asarray(geometry["cell_volumes"])
        sources.append((centers[:,active], values[active], volumes[active]))
    coordinates = np.concatenate([row[0] for row in sources], axis=1)
    values = np.concatenate([row[1] for row in sources])
    volumes = np.concatenate([row[2] for row in sources])
    assert np.isfinite(values).all() and np.isfinite(volumes).all() and np.all(volumes > 0)
    assert abs(float(volumes.sum())-1) < 1e-14
    output, selected = [], []
    for row, geometry in zip(image["rows"], image["geometry"], strict=True):
        centers = np.asarray(geometry["centers"])
        x = centers.reshape(2,-1)
        # No PoPS operator/helper is used to evaluate this explicit kernel.
        kernel = .25+x[0,:,None]*coordinates[1,None,:]-2*coordinates[0,None,:]+.5*x[1,:,None]
        result = np.sum(kernel*(values*volumes)[None,:], axis=1).reshape(np.asarray(row["active"]).shape)
        output.append(result)
        selected.append(result[np.asarray(row["active"])])
    active_values = np.concatenate(selected)
    return output, {"sum":float(active_values.sum()), "abs_sum":float(np.abs(active_values).sum()),
        "min":float(active_values.min()), "max":float(active_values.max())}


def check(image, initial, previous, width, step, source_name):
    physical = check_saved(image["rows"], CELLS, width, previous, step, initial["rows"])
    expected, reductions = reference(image, source_name)
    native = dict(image["metadata"][2])
    error = {}
    for kind in KINDS:
        value = native["ir19."+source_name+"."+kind]
        assert np.isfinite(value)
        error[kind] = abs(value-reductions[kind])
        assert error[kind] <= ACCEPTANCE*max(1.,abs(reductions[kind]))
    assert reductions["min"] < 0 < reductions["max"]
    native_residual = [value for name, value in native.items() if name.endswith(".rel_residual")]
    assert len(native_residual) == 1 and 0 <= native_residual[0] <= CONTROLS["tolerance"]
    return expected, {"original_stage":physical, "native_original_F_composite_relative":native_residual[0],
        "interaction_reduction_errors":error, "independent_reference_reductions":reductions,
        "native_reductions":{kind:native["ir19."+source_name+"."+kind] for kind in KINDS}}


def save_phase(directory, phase, image, expected):
    """Typed wire preserves rank-local binary/array shape; NPZ is separately reloadable."""
    wire_path = directory/(phase+"-array-wire.json")
    envelope = array_wire_encode(image)
    wire_path.write_text(json.dumps(envelope, sort_keys=True)+"\n")
    decoded = array_wire_decode(json.loads(bounded_bytes(wire_path)))
    same(image, decoded)
    data = {}
    for level, (row, geometry) in enumerate(zip(decoded["rows"], decoded["geometry"], strict=True)):
        for key, value in row.items():
            data["level_%d_%s" % (level,key)] = np.asarray(value).copy()
        for key in ("valid_cells", "coverage", "cell_volumes", "centers"):
            data["level_%d_geometry_%s" % (level,key)] = np.asarray(geometry[key]).copy()
        if expected is not None:
            data["level_%d_I_COMPUTED_REFERENCE" % level] = expected[level]
    npz_path = directory/(phase+"-observations.npz")
    np.savez(npz_path, **data)
    with np.load(BytesIO(bounded_bytes(npz_path)), allow_pickle=False) as stored:
        assert set(stored.files) == set(data)
        for key, value in data.items():
            assert stored[key].shape == value.shape and stored[key].dtype == value.dtype
            assert stored[key].tobytes() == value.tobytes()
    return {"wire":{"path":str(wire_path),"sha256":digest(wire_path)},
        "npz":{"path":str(npz_path),"sha256":digest(npz_path)},
        "source_role":image["source_role"], "I_array_role":"COMPUTED_REFERENCE_ONLY"}


def origins(artifact, directory):
    from pops.codegen.compile_provenance import artifact_sidecar_path
    components = [("block-"+row.name,row.model) for row in artifact.blocks]
    components += [("program-"+row.layout_id,row.program) for row in artifact.layout_programs]
    result = []
    for index, (name, component) in enumerate(components):
        binary = Path(component.so_path)
        sidecar = Path(artifact_sidecar_path(str(binary)))
        entry = {"component":name,"DSO":{"path":str(binary),"sha256":digest(binary)},
            "sidecar":{"path":str(sidecar),"sha256":digest(sidecar)}}
        if name.startswith("program-"):
            assert component._generated_cpp is not None and component.compile_command
            entry.update(program_hash=component.program_hash, command=component.compile_command)
            for extension, dump in (("cpp",component.dump_cpp),("ir.json",component.dump_ir)):
                path = Path(dump(directory/("program-%d.%s" % (index,extension))))
                entry[extension] = {"path":str(path),"sha256":digest(path)}
        result.append(entry)
    return result
