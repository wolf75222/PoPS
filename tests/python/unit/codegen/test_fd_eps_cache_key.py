"""ADC-617 codegen finite-difference epsilon (fd_eps) + compile-cache participation.

Two families:
  * wave_speeds_from_jacobian(eig='fd', fd_eps=...): the emitted C++ literal changes AND model_hash
    changes; the default (fd_eps=None) is byte-identical + hash-stable vs the pre-617 emission.
  * solve_local_nonlinear(fd_eps=...): the emitted Newton kernel literal changes AND the program IR
    hash changes; the default is byte-identical.

Emit-level + hash-level only (no _pops runtime needed); self-skips without the DSL.
"""
import pytest

pops = pytest.importorskip("pops")
Model = pytest.importorskip("pops.physics.facade").Model
from pops.codegen.compile_emit import model_hash  # noqa: E402


def _fd_model(fd_eps=None):
    m = Model("jacfd" if fd_eps is None else "jacfd_eps")
    q1, q2 = m.conservative_vars("q1", "q2")
    m.flux(x=[0.5 * q1 * q1, 0.5 * q2 * q2], y=[0.5 * q2 * q2, 0.5 * q1 * q1])
    m.wave_speeds_from_jacobian(eig="fd", fd_eps=fd_eps)
    m.primitive_vars(q1, q2)
    m.conservative_from([q1, q2])
    return m


def test_wave_speeds_default_fd_eps_emits_historical_literal():
    src = _fd_model()._m.emit_cpp_brick()
    # The default keeps the exact historical literal (1e-6 relative + 1e-30 floor), byte-identical.
    assert "pops::Real(1e-6) * (U[0] < 0 ? -U[0] : U[0]) + pops::Real(1e-30)" in src


def test_wave_speeds_fd_eps_override_changes_literal_and_hash():
    default = _fd_model()
    override = _fd_model(fd_eps=1e-4)
    src_d = default._m.emit_cpp_brick()
    src_o = override._m.emit_cpp_brick()
    assert "pops::Real(1e-06)" not in src_d  # the default is the verbatim 1e-6, not the repr form
    assert "pops::Real(0.0001)" in src_o, "the configured fd_eps replaces the emitted literal"
    assert src_d != src_o
    # The cache key MUST bust: fd_eps enters the ws_jac part of model_hash.
    assert model_hash(default._m) != model_hash(override._m)


def test_wave_speeds_default_fd_eps_hash_is_stable():
    # None -> the ws_jac hash part is byte-identical to a model with no fd_eps segment (no spurious
    # cache miss for existing models). Two default models hash identically and deterministically.
    assert model_hash(_fd_model()._m) == model_hash(_fd_model()._m)


def test_wave_speeds_fd_eps_rejected_on_numeric_path():
    m = Model("jacnum")
    q1, q2 = m.conservative_vars("q1", "q2")
    m.flux(x=[q1, q2], y=[q2, q1])
    with pytest.raises(ValueError, match="fd_eps only applies to eig='fd'"):
        m.wave_speeds_from_jacobian(eig="numeric", fd_eps=1e-5)


# --- solve_local_nonlinear (program IR) --------------------------------------

def _solve_program(adctime, fd_eps=None):
    """A minimal Program with a solve_local_nonlinear node carrying fd_eps (trivial residual so no
    compiled model is needed): r(U) = U - U0. The node stores tol / max_iter / fd_eps."""
    P = adctime.Program("p_default" if fd_eps is None else "p_eps")
    U = P.state("blk")

    def residual(P, Uit, U0):
        return P.linear_combine("r", Uit - U0)

    W = P.solve_local_nonlinear(name="W", residual=residual, initial_guess=U, tol=1e-12,
                                max_iter=20, fd_eps=fd_eps)
    P.commit("blk", W)
    return P


def test_solve_local_nonlinear_fd_eps_changes_program_ir_hash():
    adctime = pytest.importorskip("pops.time")
    default = _solve_program(adctime)
    override = _solve_program(adctime, fd_eps=1e-5)
    # fd_eps is a hashed IR node attribute -> the program hash busts when it changes.
    assert default._ir_hash() != override._ir_hash()
    # Two default programs hash identically (fd_eps=None deterministic).
    assert _solve_program(adctime)._ir_hash() == _solve_program(adctime)._ir_hash()


def test_solve_local_nonlinear_fd_eps_rejected_out_of_domain():
    adctime = pytest.importorskip("pops.time")
    P = adctime.Program("bad")
    U = P.state("blk")

    def residual(P, Uit, U0):
        return P.linear_combine("r", Uit - U0)

    with pytest.raises(ValueError, match="fd_eps"):
        P.solve_local_nonlinear(name="W", residual=residual, initial_guess=U, fd_eps=0.0)


def main():
    test_wave_speeds_default_fd_eps_emits_historical_literal()
    test_wave_speeds_fd_eps_override_changes_literal_and_hash()
    test_wave_speeds_default_fd_eps_hash_is_stable()
    test_wave_speeds_fd_eps_rejected_on_numeric_path()
    test_solve_local_nonlinear_fd_eps_changes_program_ir_hash()
    test_solve_local_nonlinear_fd_eps_rejected_out_of_domain()
    print("OK  ADC-617 fd_eps cache key")


if __name__ == "__main__":
    main()
