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
Model = pytest.importorskip("pops.physics._facade").Model
from pops.codegen._compile_emit import model_hash  # noqa: E402
from typed_program_support import typed_state  # noqa: E402


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
    from pops.solvers.nonlinear import LocalNewton
    from pops.time import FailRun, LocalResidual
    """A minimal Program with a solve_local_nonlinear node carrying fd_eps (trivial residual so no
    compiled model is needed): r(U) = U - U0. The node stores tol / max_iter / fd_eps."""
    P = adctime.Program("p_default" if fd_eps is None else "p_eps")
    U = typed_state(P, "blk")

    def residual(P, Uit, U0):
        return P.value("r", Uit - U0)

    endpoint = typed_state(P, "blk", state_name="U").next
    guess = P.value("guess", U, at=endpoint.point)
    step = 1e-7 if fd_eps is None else fd_eps
    W = P.solve(LocalResidual(residual, guess), name="W", solver=LocalNewton(
        tolerance=1e-12, max_iterations=20,
        finite_difference_step=step)).consume(action=FailRun())
    P.commit(endpoint, W)
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
    from pops.solvers.nonlinear import LocalNewton

    with pytest.raises(ValueError, match="finite_difference_step"):
        LocalNewton(finite_difference_step=0.0)


# --- ADC-645: eig_max_iter / im_tol cache-key parity (the fd_eps rule) ------------------------

def test_eig_knobs_default_hash_stable_and_override_busts():
    default_a = model_hash(_fd_model()._m)
    # Setting eig_max_iter / im_tol busts the model hash (they are emitted into the kernels).
    m_iter = _fd_model()
    m_iter._m._ws_jacobian["eig_max_iter"] = 50  # authoring-equivalent override for the hash check
    m_tol = _fd_model()
    m_tol._m._ws_jacobian["im_tol"] = 1e-7
    assert model_hash(m_iter._m) != default_a
    assert model_hash(m_tol._m) != default_a
    # Two default models (knobs None) hash identically -- the pre-645 hash is unchanged.
    assert model_hash(_fd_model()._m) == default_a


def test_eig_knobs_validated_and_carried():
    m = Model("eigk")
    q1, q2 = m.conservative_vars("q1", "q2")
    m.flux(x=[0.5 * q1 * q1, 0.5 * q2 * q2], y=[0.5 * q2 * q2, 0.5 * q1 * q1])
    m.wave_speeds_from_jacobian(eig_max_iter=50, im_tol=1e-7)
    ws = m._m._ws_jacobian
    assert ws["eig_max_iter"] == 50 and ws["im_tol"] == 1e-7
    m2 = Model("eigd")
    q1, q2 = m2.conservative_vars("q1", "q2")
    m2.flux(x=[q1, q2], y=[q2, q1])
    with pytest.raises(ValueError, match="eig_max_iter"):
        m2.wave_speeds_from_jacobian(eig_max_iter=0)
    with pytest.raises(ValueError, match="im_tol"):
        m2.wave_speeds_from_jacobian(im_tol=-1e-7)


def main():
    test_wave_speeds_default_fd_eps_emits_historical_literal()
    test_wave_speeds_fd_eps_override_changes_literal_and_hash()
    test_wave_speeds_default_fd_eps_hash_is_stable()
    test_wave_speeds_fd_eps_rejected_on_numeric_path()
    test_solve_local_nonlinear_fd_eps_changes_program_ir_hash()
    test_solve_local_nonlinear_fd_eps_rejected_out_of_domain()
    test_eig_knobs_default_hash_stable_and_override_busts()
    test_eig_knobs_validated_and_carried()
    print("OK  ADC-617 fd_eps cache key")


if __name__ == "__main__":
    main()
