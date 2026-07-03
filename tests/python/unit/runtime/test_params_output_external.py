"""Spec 5 (sec.5.12 / 5.14 / 5.17): the pops.params / pops.output / pops.external surface.

Typed scalar params (compile-time vs runtime, typed dtype, typed domain), typed output /
checkpoint / format / level policies, and typed compiled-brick references with manifest +
native id. All inert; the runtime consumes them. Needs only `import pops`.
"""

import pytest

pops = pytest.importorskip("pops")

from pops.math import Real, Integer, Bool  # noqa: E402
from pops.params import (RuntimeParam, ConstParam, DerivedParam,  # noqa: E402
                         Positive, NonNegative, Range, In, Interval, OneOf, Constant)
from pops.output import (OutputPolicy, CheckpointPolicy, HDF5, Plotfile,  # noqa: E402
                         AllLevels, CoarseOnly, SelectedLevels)
from pops.external import CompiledBrickRef, ExternalBrick  # noqa: E402
import pops.descriptors as _desc  # noqa: E402


def test_math_dtypes():
    assert Real.name == "Real" and str(Integer) == "Integer" and repr(Bool) == "Bool"


def test_runtime_and_const_params():
    a = RuntimeParam("alpha", dtype=Real, default=1.0, domain=Positive())
    assert a.name == "alpha" and a.capabilities().to_dict()["runtime"] is True
    assert a.options()["dtype"] == "Real"
    a.validate()  # default 1.0 satisfies Positive
    g = ConstParam("gamma", value=5.0 / 3.0)
    assert g.capabilities().to_dict()["in_cache_key"] is True and g.value == 5.0 / 3.0
    assert DerivedParam("Te", expression="p/rho").category == "derived_param"


def test_param_domain_rejects_bad_default():
    bad = RuntimeParam("nu", dtype=Real, default=-1.0, domain=Positive())
    with pytest.raises(ValueError):
        bad.validate()


def test_constraints():
    Positive().check(2.0)
    with pytest.raises(ValueError):
        Positive().check(0.0, who="alpha")
    NonNegative().check(0.0)
    Range(0.0, 1.0).check(0.5)
    with pytest.raises(ValueError):
        Range(0.0, 1.0).check(2.0)
    with pytest.raises(ValueError):
        Range(1.0, 0.0)  # lo > hi
    In("a", "b").check("a")
    with pytest.raises(ValueError):
        In("a", "b").check("z")


def test_interval_and_oneof_aliases():
    # ADC-541: Interval / OneOf are the readable aliases of Range / In (thin subclasses so an
    # isinstance(..., Range/In) check keeps recognising them, with their own name for diagnostics).
    iv = Interval(0.0, 1.0)
    assert isinstance(iv, Range)
    assert iv.name == "Interval"
    iv.check(0.5)
    with pytest.raises(ValueError):
        iv.check(2.0)
    assert iv.options() == {"lo": 0.0, "hi": 1.0}
    oo = OneOf("roe", "hll")
    assert isinstance(oo, In)
    assert oo.name == "OneOf"
    oo.check("roe")
    with pytest.raises(ValueError):
        oo.check("nope")


def test_runtime_param_rejects_string_domain():
    # ADC-541 / Spec 5 sec.7: domain="positive" (a bare string) is the anti-pattern a typed domain
    # replaces; it is rejected at construction naming the typed alternative.
    with pytest.raises(TypeError) as exc:
        RuntimeParam("nu", dtype=Real, domain="positive")
    msg = str(exc.value)
    assert "String algorithm selector rejected" in msg
    assert "Positive()" in msg


def test_runtime_param_check_bind_four_part_message():
    # ADC-541: a bound value outside the domain is refused at the BIND phase with the 4-part
    # diagnostic -- param name / expected domain / received value / phase.
    p = RuntimeParam("alpha", dtype=Real, default=1.0, domain=Interval(0.0, 2.0))
    assert p.check_bind(1.5) is True  # in domain
    assert p.check_bind(None) is True  # falls back to the in-domain default
    with pytest.raises(ValueError) as exc:
        p.check_bind(5.0)
    msg = str(exc.value)
    assert "alpha" in msg          # (1) param name
    assert "Interval" in msg       # (2) expected domain
    assert "5.0" in msg            # (3) received value
    assert "bind" in msg           # (4) phase


def test_runtime_param_check_bind_requires_a_value_without_default():
    # A runtime param with no default MUST be supplied at bind; a missing value is refused.
    p = RuntimeParam("beta", dtype=Real, domain=Positive())  # no default
    with pytest.raises(ValueError, match="a value is required at the bind phase"):
        p.check_bind(None)


def test_runtime_param_compile_phase_domain_error_is_four_part():
    # The declared DEFAULT is validated at the COMPILE phase with the same 4-part diagnostic.
    bad = RuntimeParam("nu", dtype=Real, default=-1.0, domain=Positive())
    with pytest.raises(ValueError) as exc:
        bad.validate()
    msg = str(exc.value)
    assert "nu" in msg and "Positive" in msg and "-1.0" in msg and "compile" in msg


def test_constant_with_unit():
    c = Constant("c", 2.998e8, unit="m/s")
    assert c.options()["unit"] == "m/s" and c.value == 2.998e8


def test_output_and_checkpoint_policies():
    out = OutputPolicy(format=HDF5(parallel=True), cadence=20, fields=["phi", "E"],
                       levels=AllLevels(), require_parallel=True)
    assert out.options()["format"] == "HDF5" and out.options()["levels"] == "all"
    assert out.requirements().to_dict()["parallel_io"] is True
    assert HDF5(parallel=True).requirements().to_dict()["parallel_io"] is True
    assert Plotfile().capabilities().to_dict()["per_level"] is True
    assert SelectedLevels(0, 1).options()["levels"] == (0, 1)
    assert CoarseOnly().options()["levels"] == "coarse"
    chk = CheckpointPolicy(restartable=True, require_bit_identical=True)
    assert chk.options()["restartable"] is True


def test_external_brick_ref_resolves_from_json_manifest(tmp_path):
    _desc._clear_external_catalog()
    manifest = tmp_path / "bricks.json"
    # ADC-611 : le schema strict versionne exige schema_version + chaque champ d'entree.
    # ADC-544 : le schema passe a la v2 (les champs v2 sont optionnels; native_id defaut = id).
    manifest.write_text(
        '{"schema_version": 2, "bricks": [{"id": "my_ext_hll", "category": "riemann", '
        '"requirements": "physical_flux,wave_speeds", "capabilities": ""}]}', encoding="utf-8")
    ref = CompiledBrickRef(manifest=str(manifest), native_id="my_ext_hll",
                           expect_category="riemann")
    assert ref.available()  # registers + resolves
    d = ref.resolve()
    assert d.brick_type == "external_cpp"
    assert "physical_flux" in d.requirements.get("capabilities", [])
    assert ExternalBrick is CompiledBrickRef
    _desc._clear_external_catalog()


def test_external_brick_ref_missing_is_explainable(tmp_path):
    _desc._clear_external_catalog()
    ref = CompiledBrickRef(manifest=str(tmp_path / "none.json"), native_id="nope")
    av = ref.available()
    assert not av.ok and "could not be resolved" in av.reason


# --- ADC-541: const params participate in the cache key; runtime params do not recompile --------
def _scalar_model(name, param):
    """A minimal scalar model whose x-flux reads @p param (rho advected at speed param)."""
    from pops.physics.model import HyperbolicModel
    m = HyperbolicModel(name)
    (rho,) = m.conservative_vars("rho")
    m.set_flux(x=[param * rho], y=[rho])
    m.set_eigenvalues(x=[rho], y=[rho])
    return m


def test_const_param_change_invalidates_cache():
    # ADC-541: a compile-time constant inlines into the formula, so changing its VALUE is a genuine
    # recompile (a distinct model_hash / cache key). ConstParam.capabilities advertises this.
    from pops.codegen.compile_emit import model_hash
    from pops.physics.model import Param
    assert ConstParam("gamma", value=1.4).capabilities().to_dict()["in_cache_key"] is True
    slow = _scalar_model("scal_ct", Param("c", 0.25, kind="const"))
    fast = _scalar_model("scal_ct", Param("c", 4.0, kind="const"))
    assert model_hash(slow) != model_hash(fast), "a const param value must recompile"


def test_runtime_param_change_does_not_recompile():
    # A runtime param reads as rparam(<name>) in the formula (its VALUE is not in the model hash),
    # so changing it is NOT a recompile while the ABI holds. RuntimeParam.capabilities advertises it.
    from pops.codegen.compile_emit import model_hash
    from pops.physics.model import Param
    assert RuntimeParam("nu", dtype=Real).capabilities().to_dict()["runtime"] is True
    assert RuntimeParam("nu", dtype=Real).capabilities().to_dict()["compile_time"] is False
    slow = _scalar_model("scal_rt", Param("nu", 0.25, kind="runtime"))
    fast = _scalar_model("scal_rt", Param("nu", 4.0, kind="runtime"))
    assert model_hash(slow) == model_hash(fast), "a runtime param value must not recompile"


def test_params_public_surface_has_no_kind_or_domain_string():
    # ADC-541 / Spec 5 sec.7: the public pops.params typed params carry NO kind= / domain= string
    # spelling -- the kind is the TYPE (RuntimeParam vs ConstParam) and the domain is a typed object.
    import inspect as _inspect
    for cls in (RuntimeParam, ConstParam):
        sig = _inspect.signature(cls.__init__)
        assert "kind" not in sig.parameters, "%s must not take a kind= string" % cls.__name__
    # domain= is a typed constraint slot: a bare string is refused (verified above); confirm the
    # type distinction is what carries the compile-time vs runtime choice.
    assert RuntimeParam("a").capabilities().to_dict()["runtime"] is True
    assert ConstParam("g", 1.0).capabilities().to_dict()["compile_time"] is True
