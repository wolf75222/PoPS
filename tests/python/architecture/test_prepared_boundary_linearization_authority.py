"""The System boundary linearization has one prepared execution authority."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SYSTEM_PROGRAM = ROOT / "src" / "runtime" / "system" / "system_program.cpp"


def _function(source: str, signature: str) -> str:
    start = source.index(signature)
    opening_brace = source.index("{", start)
    depth = 0
    for offset in range(opening_brace, len(source)):
        token = source[offset]
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            if depth == 0:
                return source[start : offset + 1]
    raise AssertionError(f"unterminated C++ function {signature}")


def test_boundary_residual_requires_complete_prepared_authority():
    source = SYSTEM_PROGRAM.read_text(encoding="utf-8")
    body = _function(source, "void System<Dim>::block_boundary_residual_into_at(")

    assert "if (&transport.lane() != &lane)" in body
    assert "if (!selected.boundary || !selected.boundary_residual_at_point_prepared)" in body
    assert "complete prepared boundary authority" in body
    assert "selected.boundary_residual_at_point_prepared(" in body
    assert "selected.boundary_residual_at_point;" not in body


def test_boundary_jvp_requires_complete_prepared_authority():
    source = SYSTEM_PROGRAM.read_text(encoding="utf-8")
    body = _function(source, "void System<Dim>::block_boundary_jvp_into_at(")

    assert "if (&transport.lane() != &lane)" in body
    assert "if (!selected.boundary || !selected.boundary_jvp_at_point_prepared)" in body
    assert "complete prepared boundary authority" in body
    assert "selected.boundary_jvp_at_point_prepared(" in body
    assert "selected.boundary_jvp_at_point;" not in body
