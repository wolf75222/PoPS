"""Named auxiliary-field regression on the native runtime seams.

The final user lifecycle supplies fields through ``Case.field`` and ``pops.bind``.  The native
runtime still owns the named auxiliary channel installed by that lifecycle, including uniform,
polar and AMR execution.  These tests keep its shape, rejection, persistence, regrid and isolation
oracles without restoring the retired AOT backend.
"""
import os
import shutil
import tempfile

import numpy as np

from pops.codegen import Production
from pops.mesh import PolarMesh
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
from pops.physics._facade import Model
from pops.physics._model import HyperbolicModel
from pops.physics.aux import AUX_NAMED_BASE, AUX_NAMED_MAX, aux_total_n_aux
import pops.runtime._engine_descriptors as engine
from pops.runtime._system import AmrSystem, System
from tests.python.support.requirements import repo_include


INCLUDE = repo_include()


def build_decay_model():
    """Build ``dn/dt = -kappa*n`` where kappa is a named auxiliary field."""
    model = Model("kappadecay")
    (density,) = model.conservative_vars("n")
    zero = 0.0 * density
    model.flux(x=[zero], y=[zero])
    model.eigenvalues(x=[zero], y=[zero])
    model.primitive_vars(n=density)
    model.conservative_from([density])
    kappa = model.aux_field("kappa")
    model.source([-(kappa * density)])
    return model


def test_form():
    """Channel width, emitted reads and declaration rejections require no compiler."""
    assert aux_total_n_aux([], []) == 3
    assert aux_total_n_aux([], ["kappa"]) == 6
    assert aux_total_n_aux([], ["kappa", "sigma"]) == 7
    assert aux_total_n_aux(["B_z"], ["kappa"]) == 6
    assert AUX_NAMED_BASE == 5

    model = HyperbolicModel("decay")
    (density,) = model.conservative_vars("n")
    kappa = model.aux_field("kappa")
    model.set_source([-(kappa * density)])
    source = model.emit_cpp_source(name="GenDecaySrc")
    assert "static constexpr int n_aux = 6;" in source
    assert "const pops::Real kappa = a.extra_field(0);" in source

    plain = HyperbolicModel("plain")
    (plain_density,) = plain.conservative_vars("n")
    plain.set_source([0.0 * plain_density])
    plain_source = plain.emit_cpp_source(name="GenPlainSrc")
    assert "n_aux" not in plain_source
    assert plain._total_n_aux() == 3

    rejected = HyperbolicModel("rejected")
    rejected.conservative_vars("n")
    for name in ("B_z", "T_e", "phi", "grad_x"):
        try:
            rejected.aux_field(name)
        except ValueError:
            pass
        else:
            raise AssertionError("canonical auxiliary name %r must be rejected" % name)

    rejected.aux_field("kappa")
    try:
        rejected.aux_field("kappa")
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate auxiliary names must be rejected")

    rejected.aux_field("a")
    rejected.aux_field("b")
    rejected.aux_field("c")
    try:
        rejected.aux_field("d")
    except ValueError:
        pass
    else:
        raise AssertionError("more than %d named auxiliary fields must be rejected" % AUX_NAMED_MAX)


def test_facade_rejects():
    """The low-level setter refuses canonical fields and unknown blocks explicitly."""
    runtime = System(n=8, L=1.0, periodic=True)
    field = np.ones((8, 8))

    try:
        runtime.set_aux_field("block", "B_z", field)
    except ValueError as exc:
        assert "set_magnetic_field" in str(exc)
    else:
        raise AssertionError("B_z must use the dedicated magnetic-field path")

    try:
        runtime.set_aux_field("block", "T_e", field)
    except ValueError as exc:
        assert "set_electron_temperature_from" in str(exc)
    else:
        raise AssertionError("T_e must use the dedicated temperature path")

    try:
        runtime.set_aux_field("block", "phi", field)
    except ValueError:
        pass
    else:
        raise AssertionError("derived canonical fields cannot be written as named auxiliaries")

    try:
        runtime.set_aux_field("missing", "kappa", field)
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("an unknown block must be rejected")


def _have_compiler():
    return bool(
        (shutil.which("c++") or shutil.which("g++") or shutil.which("clang++"))
        and os.path.isdir(INCLUDE)
    )


def _compile_decay(directory, filename, *, target="system"):
    return build_decay_model().compile(
        os.path.join(directory, filename),
        include=INCLUDE,
        backend=Production(),
        target=target,
    )


def _spatial():
    return engine.Spatial(limiter=FirstOrder(), flux=Rusanov())


def test_end_to_end():
    """A production package reads constant and spatial kappa and preserves it while stepping."""
    if not _have_compiler():
        return

    n = 16
    directory = tempfile.mkdtemp()
    try:
        compiled = _compile_decay(directory, "kappadecay.so")
        assert compiled.aux_extra_names == ["kappa"]
        assert compiled.n_aux == 6

        runtime = System(n=n, L=1.0, periodic=True)
        runtime.set_poisson(rhs="charge_density", solver="geometric_mg")
        runtime.add_equation(
            "decay", model=compiled, spatial=_spatial(), time=engine.Explicit())
        runtime.set_density("decay", np.ones((n, n)))

        before = runtime.aux_field("decay", "kappa")
        assert before.shape == (n, n)
        assert float(np.max(np.abs(before))) == 0.0

        constant = 2.0
        runtime.set_aux_field("decay", "kappa", constant * np.ones((n, n)))
        runtime.solve_fields()
        residual = np.asarray(runtime.eval_rhs("decay"))
        assert float(np.max(np.abs(residual + constant))) < 1e-12
        assert float(np.max(np.abs(runtime.aux_field("decay", "kappa") - constant))) < 1e-12

        x = (np.arange(n) + 0.5) / float(n)
        X, Y = np.meshgrid(x, x, indexing="xy")
        spatial_kappa = 1.0 + 3.0 * np.exp(-30.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))
        runtime.set_aux_field("decay", "kappa", spatial_kappa)
        runtime.solve_fields()
        residual = np.asarray(runtime.eval_rhs("decay"))
        assert float(np.max(np.abs(residual + spatial_kappa))) < 1e-12

        for _ in range(5):
            runtime.step_cfl(0.4)
        assert float(np.max(np.abs(
            runtime.aux_field("decay", "kappa") - spatial_kappa))) < 1e-12

        try:
            runtime.set_aux_field("decay", "sigma", np.ones((n, n)))
        except ValueError as exc:
            assert "sigma" in str(exc) and "kappa" in str(exc)
        else:
            raise AssertionError("an undeclared auxiliary field must be rejected")
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_polar_named_aux():
    """The polar executor widens and reads the same named auxiliary channel."""
    if not _have_compiler():
        return

    directory = tempfile.mkdtemp()
    try:
        compiled = _compile_decay(directory, "kpolar.so")
        nr, ntheta = 16, 16
        runtime = System(mesh=PolarMesh(r_min=0.3, r_max=1.0, nr=nr, ntheta=ntheta))
        runtime.add_equation(
            "decay", model=compiled, spatial=_spatial(), time=engine.Explicit())
        runtime.set_density("decay", np.ones((ntheta, nr)))

        before = runtime.aux_field("decay", "kappa")
        assert before.shape == (ntheta, nr)
        assert float(np.max(np.abs(before))) == 0.0

        constant = 3.0
        runtime.set_aux_field("decay", "kappa", constant * np.ones((ntheta, nr)))
        residual = np.asarray(runtime.eval_rhs("decay"))
        assert float(np.max(np.abs(residual + constant))) < 1e-12
        assert float(np.max(np.abs(runtime.aux_field("decay", "kappa") - constant))) < 1e-12
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _bump_density(n, lo, hi, base, peak):
    density = np.full((n, n), float(base))
    density[lo:hi, lo:hi] = float(peak)
    return density


def test_amr_named_aux_single_block_regrid():
    """Named auxiliaries persist on fine patches reconstructed at every step."""
    if not _have_compiler():
        return

    directory = tempfile.mkdtemp()
    try:
        n = 24
        lo, hi = n // 3, 2 * n // 3

        reference = AmrSystem(n=n, L=1.0, periodic=True, regrid_every=1)
        reference.add_equation(
            "decay",
            model=_compile_decay(directory, "amr0.so", target="amr_system"),
            spatial=_spatial(),
            time=engine.Explicit(),
        )
        reference.set_poisson(rhs="charge_density", solver="geometric_mg")
        reference.set_refinement(2.0)
        reference.set_density("decay", _bump_density(n, lo, hi, 1.0, 5.0))
        initial_mass = reference.mass("decay")
        for _ in range(3):
            reference.step(1e-2)
        assert abs(reference.mass("decay") - initial_mass) < 1e-10

        runtime = AmrSystem(n=n, L=1.0, periodic=True, regrid_every=1)
        runtime.add_equation(
            "decay",
            model=_compile_decay(directory, "amr1.so", target="amr_system"),
            spatial=_spatial(),
            time=engine.Explicit(),
        )
        runtime.set_poisson(rhs="charge_density", solver="geometric_mg")
        runtime.set_refinement(2.0)
        density = _bump_density(n, lo, hi, 1.0, 5.0)
        runtime.set_density("decay", density)
        runtime.set_aux_field("decay", "kappa", 2.0 * np.ones((n, n)))
        masses = [runtime.mass("decay")]
        for _ in range(5):
            runtime.step(1e-2)
            masses.append(runtime.mass("decay"))

        assert runtime.n_patches() > 0
        assert all(masses[index + 1] < masses[index] - 1e-9
                   for index in range(len(masses) - 1))
        ratio = np.asarray(runtime.density("decay")) / density
        assert float(np.std(ratio)) < 1e-2
        assert float(np.mean(ratio)) < 0.95
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _constant_decay_model(name, coefficient):
    model = Model(name)
    (density,) = model.conservative_vars("n")
    zero = 0.0 * density
    model.flux(x=[zero], y=[zero])
    model.eigenvalues(x=[zero], y=[zero])
    model.primitive_vars(n=density)
    model.conservative_from([density])
    model.source([-(float(coefficient) * density)])
    return model


def test_amr_named_aux_multiblock_regrid():
    """A shared AMR channel reaches fine patches without leaking into another block."""
    if not _have_compiler():
        return

    directory = tempfile.mkdtemp()
    try:
        n = 24
        lo, hi = n // 3, 2 * n // 3
        decay = _compile_decay(directory, "amrdecay.so", target="amr_system")
        plain = _constant_decay_model("plaindecay", 1.0).compile(
            os.path.join(directory, "amrplain.so"),
            include=INCLUDE,
            backend=Production(),
            target="amr_system",
        )

        runtime = AmrSystem(n=n, L=1.0, periodic=True, regrid_every=1)
        runtime.add_equation(
            "decay", model=decay, spatial=_spatial(), time=engine.Explicit())
        runtime.add_equation(
            "plain", model=plain, spatial=_spatial(), time=engine.Explicit())
        runtime.set_poisson(rhs="charge_density", solver="geometric_mg")
        runtime.set_refinement(2.0)
        runtime.set_density("decay", _bump_density(n, lo, hi, 1.0, 5.0))
        runtime.set_density("plain", np.ones((n, n)))
        runtime.set_aux_field("decay", "kappa", 50.0 * np.ones((n, n)))
        decay_before, plain_before = runtime.mass("decay"), runtime.mass("plain")
        for _ in range(5):
            runtime.step(1e-2)
        decay_after, plain_after = runtime.mass("decay"), runtime.mass("plain")

        assert runtime.n_patches() > 0
        assert decay_after < 0.5 * decay_before
        assert 0.90 < plain_after / plain_before < 0.99
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_amr_named_aux_rejections():
    """AMR rejects canonical fields and unknown blocks before compilation."""
    runtime = AmrSystem(n=8, L=1.0, periodic=True)
    field = np.ones((8, 8))
    for name, expected in (("B_z", "set_magnetic_field"), ("phi", "CANONICAL")):
        try:
            runtime.set_aux_field("block", name, field)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("AMR canonical field %r must be rejected" % name)

    try:
        runtime.set_aux_field("missing", "kappa", field)
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("an unknown AMR block must be rejected")
