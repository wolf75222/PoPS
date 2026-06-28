"""Spec 5 (sec.5.12 / 5.17): the pops.params / pops.external surface.

Typed scalar params (compile-time vs runtime, typed dtype, typed domain) and typed
compiled-brick references with manifest + native id. All inert; the runtime consumes them.
Needs only `import pops`.

The general pops.output OutputPolicy / CheckpointPolicy surface was REMOVED (decorative API with
no codegen / runtime wiring; ADC-509). The WIRED AMR-output surface (pops.mesh.amr.AMROutput /
CheckpointPolicy) is covered by test_mesh_descriptors.py / test_inspect_amr.py.
"""
import sys

import pytest

pops = pytest.importorskip("pops")

from pops.math import Real, Integer, Bool  # noqa: E402
from pops.params import (RuntimeParam, ConstParam, DerivedParam,  # noqa: E402
                         Positive, NonNegative, Range, In, Constant)
from pops.external import CompiledBrickRef, ExternalBrick  # noqa: E402
import pops.descriptors as _desc  # noqa: E402


def test_math_dtypes():
    assert Real.name == "Real" and str(Integer) == "Integer" and repr(Bool) == "Bool"


def test_runtime_and_const_params():
    a = RuntimeParam("alpha", dtype=Real, default=1.0, domain=Positive())
    assert a.name == "alpha" and a.capabilities()["runtime"] is True
    assert a.options()["dtype"] == "Real"
    a.validate()  # default 1.0 satisfies Positive
    g = ConstParam("gamma", value=5.0 / 3.0)
    assert g.capabilities()["in_cache_key"] is True and g.value == 5.0 / 3.0
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


def test_constant_with_unit():
    c = Constant("c", 2.998e8, unit="m/s")
    assert c.options()["unit"] == "m/s" and c.value == 2.998e8


def test_output_policy_surface_removed():
    """C4 (ADC-509): the decorative general output/checkpoint policy surface is GONE. The removed
    symbols are absent from pops.output and from the pops top level; the WIRED AMR-output home
    (pops.mesh.amr) is untouched (asserted positively here)."""
    import pops.output as out_pkg
    removed = ("OutputPolicy", "CheckpointPolicy", "HDF5", "Plotfile",
               "AllLevels", "CoarseOnly", "SelectedLevels")
    for sym in removed:
        assert not hasattr(out_pkg, sym), "pops.output.%s must be removed (decorative API)" % sym
        assert sym not in getattr(out_pkg, "__all__", []), "pops.output.__all__ must drop %s" % sym
        assert not hasattr(pops, sym), "pops.%s must not be a top-level export" % sym
    # The wired narrower AMR home stays (sec.8.11): AMROutput + AMR-local level/checkpoint policies.
    from pops.mesh.amr import AMROutput, CheckpointPolicy as AmrCheckpoint, AllLevels as AmrAllLevels
    assert AMROutput(fields=["phi"], levels=AmrAllLevels()).options()["levels"] == "all"
    assert AmrCheckpoint(restartable=True).options()["restartable"] is True


def test_external_brick_ref_resolves_from_json_manifest(tmp_path):
    _desc._clear_external_catalog()
    manifest = tmp_path / "bricks.json"
    manifest.write_text(
        '{"bricks": [{"id": "my_ext_hll", "category": "riemann", '
        '"requirements": "physical_flux,wave_speeds"}]}', encoding="utf-8")
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
