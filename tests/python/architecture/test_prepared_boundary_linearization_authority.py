"""The System boundary linearization has one prepared execution authority."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SYSTEM_PROGRAM = ROOT / "src" / "runtime" / "system" / "system_program.cpp"


def _first_overload(source: str, signature: str) -> str:
    start = source.index(signature)
    end = source.index(signature, start + len(signature))
    return source[start:end]


def test_boundary_residual_refuses_a_missing_prepared_session():
    source = SYSTEM_PROGRAM.read_text(encoding="utf-8")
    body = _first_overload(source, "void System::block_boundary_residual_into_at(")

    assert "if (!block.boundary_session)" in body
    assert "persistent prepared boundary session" in body
    assert "block.boundary_residual_at_point_prepared(" in body
    assert "block.boundary_residual_at_point;" not in body


def test_boundary_jvp_refuses_a_missing_prepared_session():
    source = SYSTEM_PROGRAM.read_text(encoding="utf-8")
    body = _first_overload(source, "void System::block_boundary_jvp_into_at(")

    assert "if (!block.boundary_session)" in body
    assert "persistent prepared boundary session" in body
    assert "block.boundary_jvp_at_point_prepared(" in body
    assert "block.boundary_jvp_at_point;" not in body
