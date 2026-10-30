"""N7a pilot (spec 0.3 IR-T1, first part): the FaceFlux builder recovers M05 and M10.

The M05 and M10 faces of ``models_M01_M14.py`` are written once more as face
schemes over typed maps (``examples/api_v02/faceflux_pilot.py``). The builder
of ``pops.lib.numerics`` must derive what the b6 author writes by hand: the
halo widths with the b6 formulas (IR14), the face identities (IR17) and the
expected class (IR21), and lower to the very same ``StencilFlux`` (IR24).

These tests establish authoring, derivation and generated-source identity
(levels E and G of K6), never a compiled or native claim.
"""
import ast
import inspect
import pathlib
import re

import numpy as np
import pytest

import pops
import examples.api_v02.models_M01_M14 as cases
from examples.api_v02.faceflux_pilot import navier_stokes_scheme, scharfetter_gummel_scheme
from pops.lib.numerics import FaceFlux, Incidence, realize_face_flux
from pops.model.hash_data import canonical_hash_data


REPO = pathlib.Path(__file__).resolve().parents[4]
NS = ("rho", "mx", "my", "E")


def _m05(components=NS, name="M05_compressible_Navier_Stokes", **parameters):
    frame, _ = cases._domain("M05_NS", (16, 8))
    model = pops.Model(name, frame=frame)
    U = model.state("conservative", components=components)
    return realize_face_flux(navier_stokes_scheme(U, **parameters), U, frame=model.frame)


def _m10(name="M10_Nernst_Planck_Poisson", **parameters):
    lengths = (2*np.pi, 2*np.pi)
    frame, _ = cases._domain("M10_periodic", (16, 8), lengths=lengths)
    model = pops.Model(name, frame=frame)
    U = model.state("density", components=("n",))
    phi = model.field("solved_potential")
    return realize_face_flux(scharfetter_gummel_scheme(U, phi, **parameters), U,
                             frame=model.frame), phi


def test_m05_halo_is_derived_and_equals_b6():
    realized, b6 = _m05(), cases.navier_stokes_face()
    assert realized.ghost_extent == b6.ghost_extent == (2, 2)
    # One depth per state, as build_m05 stores it with StateStorage(ghost_depth=...).
    assert realized.ghost_depth == max(b6.ghost_extent) == 2
    # The per-side widths that b6 folds into one number. Its formula puts the
    # +1 of the face's normal axis on every axis, so the high side reads 2
    # where the transverse reads of the viscous terms reach one cell.
    assert realized.halo() == ((1, 1), (2, 2))
    assert realized.stencil_flux.radius == b6.radius == 1


def test_m10_state_and_field_halos_are_derived_and_equal_b6():
    (realized, phi), b6 = _m10(), cases.scharfetter_gummel_face()
    assert realized.ghost_extent == b6.ghost_extent == (1, 1)
    assert realized.field_ghost_extents == b6.field_ghost_extents == ((1, 1),)
    assert len(realized.fields) == 1 and realized.fields[0] is phi
    assert realized.halo() == realized.halo(phi) == ((1, 1), (1, 1))
    assert realized.stencil_flux.field_components == b6.field_components == (1,)


@pytest.mark.parametrize("build,b6", (
    (lambda: _m05(), cases.navier_stokes_face),
    (lambda: _m10()[0], cases.scharfetter_gummel_face),
))
def test_builder_lowers_to_the_b6_stencil_flux_exactly(build, b6):
    # Same rows (expression DAGs), same sample keys and leaves, same derived
    # extents: the IR24 substitution receives the b6 declaration itself.
    assert build().stencil_flux.to_data() == b6().to_data()


def test_halo_follows_the_reads_and_is_never_declared():
    frame, _ = cases._domain("wide", (8, 8))
    model = pops.Model("wide", frame=frame)
    U = model.state("q", components=("q",))
    q = U["q"]

    def wide(face):
        # A wider face: reads two cells beyond the right cell along the normal.
        far = face.right_cell.shift(face.axis, 2).value(q)
        return FaceFlux({"q": (face.right(q)-face.left(q)+far)/face.distance},
                        frequency=1/face.distance**2)
    realized = realize_face_flux(wide, U, frame=model.frame)
    assert realized.halo() == ((1, 1), (3, 3))
    assert realized.ghost_extent == realized.stencil_flux.ghost_extent == (3, 3)
    assert realized.stencil_flux.radius == 2
    # Two paths to one cell are one read: the right cell, reached from the left one.
    def twice(face):
        again = face.left_cell.shift(face.axis, 1).value(q)
        return FaceFlux({"q": (face.right(q)+again-2*face.left(q))/face.distance},
                        frequency=1/face.distance**2)
    assert [identity["reads"] for identity in
            realize_face_flux(twice, U, frame=model.frame).identities()] == [
        ((-1, 0), (0, 0)), ((0, -1), (0, 0))]

    def two_point(face):
        return FaceFlux({"q": face.normal_gradient(q)}, frequency=1/face.distance**2)
    assert realize_face_flux(two_point, U, frame=model.frame).ghost_extent == (1, 1)
    parameters = tuple(inspect.signature(realize_face_flux).parameters)
    assert parameters == ("scheme", "state", "frame")


def test_case_schemes_write_no_radius_halo_dimension_or_component_index():
    tree = ast.parse((REPO/"examples/api_v02/faceflux_pilot.py").read_text(encoding="utf-8"))
    forbidden = {"radius", "ghost_depth", "ghost_extent", "dimension", "components",
                 "StencilFlux", "StateStorage", "sample", "field_components"}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names |= {node.arg for node in ast.walk(tree) if isinstance(node, ast.keyword) and node.arg}
    assert not names & forbidden
    # Components are read by name through the state handle, never by position.
    subscripts = [node for node in ast.walk(tree) if isinstance(node, ast.Subscript)
                  and isinstance(node.value, ast.Name) and node.value.id == "U"]
    assert subscripts and all(isinstance(node.slice, (ast.Constant, ast.Name)) and
                              not isinstance(getattr(node.slice, "value", ""), int)
                              for node in subscripts)


def _native_assembly(J, axis):
    # include/pops/numerics/fv/stencil_flux.hpp:150-153@b6: the face of index i
    # along an axis has cell i as its right cell, and
    # net(cell) = J(cell) - J(cell + e_axis), on a periodic box.
    return J - np.roll(J, -1, axis=axis)


def _derived_assembly(identity, J):
    out = np.zeros_like(J)
    axis = identity["axis"]
    for incidence, weight in identity["targets"]:
        offset = incidence.offset(J.ndim)
        # The face of index i delivers weight * J[i] to the cell i + offset.
        out += weight*np.roll(J, offset[axis], axis=axis)
    return out


@pytest.mark.parametrize("realized", (_m05(), _m10()[0]))
def test_face_identities_match_a_transcription_of_the_native_assembly(realized):
    # The normal and the offsets come from the incidence maps; the deliveries
    # are the builder's own (D6), so this is an arithmetic check. The executed
    # evidence of conservation is the b6 mass_defect invariant that the native
    # test asserts on both routes.
    identities = realized.identities()
    assert [i["axis"] for i in identities] == [0, 1]
    for identity in identities:
        axis = identity["axis"]
        e = tuple(int(d == axis) for d in range(2))
        assert identity["normal"] == e
        assert identity["left_offset"] == tuple(-x for x in e)
        assert identity["right_offset"] == (0, 0)
        assert identity["targets"] == ((Incidence(axis, "left"), -1), (Incidence(axis, "right"), +1))
        assert identity["conservation"] is True
        J = np.random.default_rng(axis).standard_normal((7, 5))
        np.testing.assert_array_equal(_derived_assembly(identity, J), _native_assembly(J, axis))
        # A conservative face transfer leaves the total unchanged (IR17).
        assert abs(_derived_assembly(identity, J).sum()) < 1e-12
    # The conventions that b6 records by hand are the derived ones.
    data = realized.stencil_flux.to_data()
    assert (data["orientation"], data["sampling"]) == ("positive_axis", "right_incident_cell")


def _rows_by_name(realized):
    flux = realized.stencil_flux
    rows = tuple(dict(zip((*realized.components, "frequency"), row)) for row in flux.rows)
    return tuple({name: canonical_hash_data(expr) for name, expr in row.items()} for row in rows)


def test_identities_and_lowering_survive_a_permutation_of_the_components():
    original, permuted = _m05(), _m05(components=("E", "my", "rho", "mx"))
    assert permuted.ghost_extent == original.ghost_extent
    assert permuted.halo() == original.halo()
    strip = lambda realized: [{k: v for k, v in i.items() if k != "components"}
                              for i in realized.identities()]
    assert strip(permuted) == strip(original)
    assert permuted.identities()[0]["components"] == ("E", "my", "rho", "mx")
    # The same reads by name land on the permuted component slots ...
    by_name = lambda realized: [(offset, realized.components[c], leaf.name)
                                for offset, c, leaf in realized.stencil_flux.samples]
    assert by_name(permuted) == by_name(original)
    assert ({c for _, c, _ in permuted.stencil_flux.samples}
            == {c for _, c, _ in original.stencil_flux.samples} == {0, 1, 2, 3})
    # ... and every face density, keyed by its component name, is the same expression.
    assert _rows_by_name(permuted) == _rows_by_name(original)
    assert permuted.stencil_flux.to_data()["rows"] != original.stencil_flux.to_data()["rows"]


@pytest.mark.parametrize("realized", (_m05(), _m10()[0]))
def test_reached_class_is_face_incidence(realized):
    report = realized.lowering_class()
    assert report["expected"] == report["reached"] == "face_incidence"
    assert report["reached_by"] == "construction of the face vocabulary"
    assert "StencilFlux" in report["substitution"]


def test_m05_class_report_names_the_composed_transverse_shifts():
    assert _m05().lowering_class()["composed_shifts"] == ((-1, 0), (0, -1), (0, 1), (1, 0))
    assert _m10()[0].lowering_class()["composed_shifts"] == ()


def _emit(spec):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    source = emit_cpp_program(resolved.time, model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    # b6 writes the process-local serial of the authoring model into some
    # history labels (python/pops/model/ownership.py:560@b6), so two builds of
    # the same case in one process differ there and only there.
    return re.sub(r"#authoring=\d+", "#authoring=N", source)


def _graph_hash(spec):
    # The semantic identity of a program, as CF4 compares a preset with its
    # hand-written program (tests/python/integration/native_loader/
    # test_normalized_program_execution.py:184-221@b6).
    return pops.resolve(pops.validate(spec.case), layout=spec.layout).time.to_graph().graph_hash


def test_two_b6_builds_differ_only_by_the_authoring_serial():
    assert _emit(cases.build_m10()) == _emit(cases.build_m10())
    assert _emit(cases.build_m05(cells=(16, 8))) == _emit(cases.build_m05(cells=(16, 8)))


def test_m05_program_built_with_the_builder_emits_the_b6_source(monkeypatch):
    b6, b6_hash = _emit(cases.build_m05(cells=(16, 8))), _graph_hash(cases.build_m05(cells=(16, 8)))
    monkeypatch.setattr(cases, "navier_stokes_face",
                        lambda **parameters: _m05(name="probe", **parameters).stencil_flux)
    assert _emit(cases.build_m05(cells=(16, 8))) == b6
    assert _graph_hash(cases.build_m05(cells=(16, 8))) == b6_hash
    # The comparison sees the face: another viscosity changes the source.
    monkeypatch.setattr(cases, "navier_stokes_face",
                        lambda **parameters: _m05(name="probe", viscosity=.02).stencil_flux)
    assert _emit(cases.build_m05(cells=(16, 8))) != b6


def test_m10_program_built_with_the_builder_emits_the_b6_source(monkeypatch):
    b6, b6_hash = _emit(cases.build_m10()), _graph_hash(cases.build_m10())
    monkeypatch.setattr(cases, "scharfetter_gummel_face",
                        lambda **parameters: _m10(name="probe", **parameters)[0].stencil_flux)
    assert _emit(cases.build_m10()) == b6
    assert _graph_hash(cases.build_m10()) == b6_hash
    monkeypatch.setattr(cases, "scharfetter_gummel_face",
                        lambda **parameters: _m10(name="probe", mobility=.2)[0].stencil_flux)
    assert _emit(cases.build_m10()) != b6


def test_scheme_must_cover_every_component_once():
    frame, _ = cases._domain("cover", (8, 8))
    model = pops.Model("cover", frame=frame)
    U = model.state("conservative", components=NS)
    with pytest.raises(ValueError, match="CoverageNotExact"):
        realize_face_flux(lambda face: FaceFlux({"rho": face.left(U["rho"])}, frequency=1.),
                          U, frame=model.frame)


def test_a_scheme_that_reads_other_cells_when_realized_is_refused():
    frame, _ = cases._domain("drifting", (8, 8))
    model = pops.Model("drifting", frame=frame)
    U = model.state("q", components=("q",))
    q, calls = U["q"], []

    def drifting(face):
        calls.append(face.axis)
        value = face.right(q)-face.left(q)
        if len(calls) > 2:  # after one recording call per axis: the realization pass
            value = value+face.right_cell.shift(face.axis, 1).value(q)
        return FaceFlux({"q": value}, frequency=1.)
    with pytest.raises(ValueError, match="must read the same cells"):
        realize_face_flux(drifting, U, frame=model.frame)
    assert len(calls) > 2


def test_reads_of_a_foreign_state_field_or_frame_are_refused():
    frame, _ = cases._domain("own", (8, 8))
    model = pops.Model("own", frame=frame)
    U = model.state("q", components=("q",))
    other_model = pops.Model("other", frame=frame)
    other = other_model.state("q", components=("q",))
    with pytest.raises(TypeError, match="own state"):
        realize_face_flux(lambda face: FaceFlux({"q": face.left(other["q"])}, frequency=1.),
                          U, frame=model.frame)
    chi = other_model.field("chi")
    with pytest.raises(TypeError, match="own state or solved fields"):
        realize_face_flux(lambda face: FaceFlux({"q": face.left(U["q"])*face.right(chi)},
                                                frequency=1.), U, frame=model.frame)
    elsewhere, _ = cases._domain("elsewhere", (8, 8), lengths=(2., 1.))
    with pytest.raises(ValueError, match="frame"):
        realize_face_flux(lambda face: FaceFlux({"q": face.left(U["q"])}, frequency=1.),
                          U, frame=elsewhere)


def test_offsets_are_integer_constants():
    frame, _ = cases._domain("offset", (8, 8))
    model = pops.Model("offset", frame=frame)
    U = model.state("q", components=("q",))
    with pytest.raises(TypeError, match="StructuredOffsetNotConstant"):
        realize_face_flux(
            lambda face: FaceFlux({"q": face.right_cell.shift(face.axis, 1.5).value(U["q"])},
                                  frequency=1.),
            U, frame=model.frame)
