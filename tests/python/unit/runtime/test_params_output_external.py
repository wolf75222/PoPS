"""Spec 5 (sec.5.12 / 5.14 / 5.17): the pops.params / pops.output / pops.external surface.

Typed scalar params (compile-time vs runtime, typed dtype, typed domain), typed output /
checkpoint / format / level policies, and typed compiled-brick references with manifest +
native id. All inert; the runtime consumes them. Needs only `import pops`.
"""

import json

import pytest

pops = pytest.importorskip("pops")

from pops.math import Real, Integer, Bool  # noqa: E402
from pops.params import (MISSING, RuntimeParam, ConstParam, DerivedParam,  # noqa: E402
                         ParamInvalidation, ParamPhase, ParamStorage,
                         Positive, NonNegative, Range, In, Interval, OneOf, Constant)
from pops.output import (OutputPolicy, CheckpointPolicy, HDF5, Plotfile,  # noqa: E402
                         AllLevels, CoarseOnly, SelectedLevels)
from pops.external import CompiledBrickRef, ExternalBrick  # noqa: E402
from pops.model import Handle, Module, OperatorHandle, OwnerPath  # noqa: E402
from pops.ir.expr import Var  # noqa: E402
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
    module = Module("derived")
    alpha = module.param(a)
    derived = DerivedParam(
        "alpha2", expression=Var("alpha", "param") * Var("alpha", "param"),
        depends_on=(alpha,), phase=ParamPhase.PerBlock,
        storage=ParamStorage.DerivedCache,
        invalidation=ParamInvalidation.OnDependencies)
    assert derived.category == "derived_param"


def test_param_domain_rejects_bad_default():
    with pytest.raises(ValueError, match="compile"):
        RuntimeParam("nu", dtype=Real, default=-1.0, domain=Positive())


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
    assert iv.options() == {
        "lo": {"kind": "binary64", "value": (0.0).hex()},
        "hi": {"kind": "binary64", "value": (1.0).hex()},
    }
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
    assert p.check_bind(MISSING) is True  # falls back to the in-domain default
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
        p.check_bind(MISSING)


def test_runtime_param_compile_phase_domain_error_is_four_part():
    # The declared DEFAULT is validated at the COMPILE phase with the same 4-part diagnostic.
    with pytest.raises(ValueError) as exc:
        RuntimeParam("nu", dtype=Real, default=-1.0, domain=Positive())
    msg = str(exc.value)
    assert "nu" in msg and "Positive" in msg and "-1.0" in msg and "compile" in msg


def test_constant_with_unit():
    c = Constant("c", 2.998e8, unit="m/s")
    assert c.options()["unit"] == "m/s" and c.value == 2.998e8


def test_output_and_checkpoint_policies():
    phi = Handle("phi", kind="field", owner=OwnerPath.shared("runtime.output-policy"))
    electric = Handle("E", kind="field", owner=OwnerPath.shared("runtime.output-policy"))
    out = OutputPolicy(format=HDF5(parallel=True), cadence=20, fields=[phi, electric],
                       levels=AllLevels(), require_parallel=True)
    assert out.options()["format"] == "HDF5" and out.options()["levels"] == "all"
    assert out.requirements().to_dict()["parallel_io"] is True
    assert HDF5(parallel=True).requirements().to_dict()["parallel_io"] is True
    assert Plotfile().capabilities().to_dict()["per_level"] is True
    assert SelectedLevels(0, 1).options()["levels"] == (0, 1)
    assert CoarseOnly().options()["levels"] == "coarse"
    chk = CheckpointPolicy(restartable=True, require_bit_identical=True)
    assert chk.options()["restartable"] is True


def test_output_policy_rejects_string_field_references():
    with pytest.raises(TypeError, match="declaration Handle"):
        OutputPolicy(fields=["phi"])


def test_output_policy_rejects_non_writable_semantic_handles():
    owner = OwnerPath.shared("runtime.output-policy-kinds")
    operator = OperatorHandle("rhs", kind="local_rate", owner=owner)
    parameter = Handle("alpha", kind="parameter", owner=owner)
    for reference in (operator, parameter, Handle("block", kind="block", owner=owner)):
        with pytest.raises(TypeError, match="writable state/field/aux"):
            OutputPolicy(fields=[reference])

    from pops.runtime._output_driver import _field_names
    with pytest.raises(TypeError, match="writable state/field/aux"):
        _field_names([operator])


def test_external_brick_ref_resolves_from_json_manifest(tmp_path):
    _desc._clear_external_catalog()
    from pops.runtime.bricks import abi_key
    manifest = tmp_path / "bricks.json"
    manifest.write_text(json.dumps({
        "schema_version": _desc.BRICK_MANIFEST_SCHEMA_VERSION,
        "abi_key": abi_key(), "annotations": {},
        "bricks": [{
            "id": "my_ext_hll", "category": "riemann", "native_id": "my_ext_hll",
            "requirements": "physical_flux,wave_speeds", "capabilities": "",
            "supported_layouts": "", "supported_platforms": "", "params": "", "options": "",
            "exported_symbols": "",
        }],
    }), encoding="utf-8")
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


def test_const_param_change_invalidates_cache():
    slow = ConstParam("c", value=0.25)
    fast = ConstParam("c", value=4.0)
    assert slow.capabilities().to_dict()["in_cache_key"] is True
    assert slow.artifact_data() != fast.artifact_data()


def test_runtime_param_change_does_not_recompile():
    missing = RuntimeParam("nu", dtype=Real)
    slow = RuntimeParam("nu", dtype=Real, default=0.25)
    fast = RuntimeParam("nu", dtype=Real, default=4.0)
    assert missing.capabilities().to_dict()["runtime"] is True
    assert missing.capabilities().to_dict()["compile_time"] is False
    assert missing.artifact_data() == slow.artifact_data() == fast.artifact_data()


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
