"""Final exact-ranked field-view seam for generated Program cell kernels."""

from pops.codegen.program_emit_kernels import (
    _PROGRAM_CPP_TEMPLATE,
    _apply_in_arg,
    _emit_cell_compare_kernel,
    _emit_where_kernel,
    _kernel_open,
)


def _source(lines: list[str]) -> str:
    return "\n".join(lines)


def test_kernel_open_uses_native_ranked_fields_views_and_indices() -> None:
    source = _source(_kernel_open("out", "state"))

    assert "pops::FieldView<pops::Real, pops::kNativeDimension> outA" in source
    assert "pops::FieldView<const pops::Real, pops::kNativeDimension> stateA" in source
    assert "ctx.aux()" not in source
    assert "auxA" not in source
    assert "const pops::CellIndex<pops::kNativeDimension>& index" in source
    assert "std::as_const(state).fab(li).view()" in source
    assert "Array4" not in source
    assert "int i, int j" not in source


def test_model_free_cell_kernels_use_the_same_exact_ranked_seam() -> None:
    compare = _source(_emit_cell_compare_kernel("field", "mask", ">", 0))
    where = _source(_emit_where_kernel("mask", "left", "right", "out"))

    for source in (compare, where):
        assert "FieldView" in source
        assert "CellIndex<pops::kNativeDimension>" in source
        assert "(index," in source
        assert "Array4" not in source
        assert "int i, int j" not in source


def test_program_template_and_apply_input_have_no_unranked_storage_route() -> None:
    assert "pops/mesh/storage/field_view.hpp" in _PROGRAM_CPP_TEMPLATE
    assert "pops/mesh/storage/fab2d.hpp" not in _PROGRAM_CPP_TEMPLATE
    assert _apply_in_arg({7: "in"}, type("Value", (), {"id": 7})()) == (
        "const_cast<pops::MultiFab<pops::kNativeDimension>&>(in)"
    )


def test_program_template_has_a_private_nonthrowing_install_diagnostic_writer() -> None:
    assert "#include <pops/runtime/program/program_abi.hpp>" in _PROGRAM_CPP_TEMPLATE
    assert "namespace {{" in _PROGRAM_CPP_TEMPLATE
    assert "void write_program_install_diagnostic(" in _PROGRAM_CPP_TEMPLATE
    assert "ProgramInstallDiagnostic* diagnostic" in _PROGRAM_CPP_TEMPLATE
    assert "ProgramInstallErrorCode code" in _PROGRAM_CPP_TEMPLATE
    assert ") noexcept {{" in _PROGRAM_CPP_TEMPLATE
    assert "size + 1 < sizeof(diagnostic->message)" in _PROGRAM_CPP_TEMPLATE
    assert "diagnostic->message[size] = '\\0';" in _PROGRAM_CPP_TEMPLATE
