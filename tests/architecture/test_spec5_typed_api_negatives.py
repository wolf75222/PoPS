"""Spec 5 (sec.15): the typed-object API rejects strings and stays inspectable.

Spec 5 ("Python describes with typed objects, C++ executes; no YAML disguised as
Python") makes a set of architecture promises about the central descriptor packages:

* a free-string algorithm selector is rejected, not silently accepted
  (``pops.descriptors.reject_string_selector``);
* a typed descriptor constructor does NOT take a ``kind="..."`` string -- the type IS
  the kind, so ``HLL(kind=...)`` / ``RuntimeParam(..., kind=...)`` is a ``TypeError``;
* every catalog descriptor is inspectable (``.inspect()`` -> dict) and self-validating
  (``.validate()`` does not raise / returns truthy);
* the compiled-brick load path refuses a brick without a real manifest with a clear error.

These are negative / contract tests for the typed surface. They IMPORT ``pops`` and so
need the compiled ``_pops`` extension; if it cannot be loaded the module is skipped (like
``test_public_imports.py``), so the source-only architecture checks still run bare.
"""
import pytest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

# Skip the whole module if the native extension cannot be loaded in this interpreter.
# importorskip is too strict here (pops/_bootstrap raises a custom ImportError whose .name
# does not match "pops._pops"), so catch any import failure and skip at module level.
try:
    import pops._pops  # noqa: F401
except Exception as _exc:  # pragma: no cover - exercised only without a built extension
    pytest.skip("compiled _pops extension not importable: %s" % _exc,
                allow_module_level=True)


def test_reject_string_selector_raises_and_is_actionable():
    # A free string for an algorithm selector is a TypeError that names the param, echoes
    # the rejected value, and points at the typed alternative.
    import pops.descriptors as descriptors

    with pytest.raises(TypeError) as excinfo:
        descriptors.reject_string_selector("hll", "riemann", suggestion="HLL()")
    message = str(excinfo.value)
    assert "riemann" in message            # names the rejected parameter
    assert "hll" in message                # echoes the rejected value
    assert "HLL()" in message              # points at the typed alternative


def test_reject_string_selector_does_not_touch_a_real_typed_object():
    # The guard only fires on the string branch: passing the typed object as the suggestion
    # (a real descriptor instance) still raises -- the helper ALWAYS raises by design -- but
    # the real typed object itself is untouched and remains a usable descriptor.
    import pops.descriptors as descriptors
    from pops.numerics.riemann import HLL

    typed = HLL()
    # The real typed object is not rejected on its own: it is a valid descriptor.
    assert typed.validate()
    assert typed.inspect()["native_id"] == typed.native_id
    # Used as the suggestion it is rendered, not mutated, and the guard still raises.
    with pytest.raises(TypeError):
        descriptors.reject_string_selector("hll", "riemann", suggestion=typed)
    assert typed.native_id == "pops::HLLFlux"  # untouched


def test_typed_descriptor_constructors_reject_a_kind_string():
    # The TYPE is the kind: a typed descriptor constructor must not accept a kind="..."
    # selector string. CPython raises TypeError("unexpected keyword argument 'kind'").
    from pops.numerics.riemann import HLL
    from pops.physics.model import Param
    from pops.params import RuntimeParam

    with pytest.raises(TypeError):
        HLL(kind="x")
    with pytest.raises(TypeError):
        Param("a", 1.0, kind="runtime")
    with pytest.raises(TypeError):
        RuntimeParam("a", kind="runtime")


def test_catalog_descriptors_are_inspectable_and_validate():
    # Every catalogued numerics/diagnostics descriptor that constructs with no required
    # args exposes .inspect() -> dict and a .validate() that does not raise / returns truthy.
    import pops.diagnostics as diagnostics
    import pops.numerics.reconstruction as reconstruction
    import pops.numerics.riemann as riemann
    import pops.numerics.variables as variables

    checked = 0
    for module in (riemann, reconstruction, variables, diagnostics):
        for name in getattr(module, "__all__", ()):
            obj = getattr(module, name)
            if not callable(obj):
                continue
            try:
                instance = obj()
            except TypeError:
                # Constructors that need arguments (e.g. User(...)) are out of scope here.
                continue
            if not (hasattr(instance, "inspect") and hasattr(instance, "validate")):
                continue
            view = instance.inspect()
            assert isinstance(view, dict), "%s.%s().inspect() must return a dict" % (
                module.__name__, name)
            assert view, "%s.%s().inspect() returned an empty dict" % (module.__name__, name)
            assert instance.validate(), "%s.%s().validate() must be truthy" % (
                module.__name__, name)
            checked += 1
    assert checked >= 8, ("expected to inspect the riemann/reconstruction/variables/diagnostics "
                          "catalogs")


def test_public_top_level_assembly_is_not_a_descriptor_because_it_is_not_public():
    # Corrective clean break: the old top-level assembly façade is not public. Descriptors remain
    # descriptors, but there is no public pops.Case/pops.Problem object pretending to be one.
    import pops
    from pops.descriptors import Descriptor

    assert not hasattr(pops, "Case")
    assert not hasattr(pops, "Problem")
    assert isinstance(Descriptor(), Descriptor)


def test_optimization_math_rejects_a_bare_string():
    # Spec 5 sec.14.2 / #20-21: the codegen Optimization math= / fuse= selectors are TYPED objects;
    # a bare string is rejected at construction (not silently mis-set and crashed later), while the
    # typed StrictMath() / FastMath() / ... usage keeps working.
    from pops.codegen import Optimization, FastMath, StrictMath

    with pytest.raises(TypeError) as excinfo:
        Optimization(math="fast")
    message = str(excinfo.value)
    assert "optimization math" in message and "fast" in message
    assert "StrictMath()" in message and "FastMath()" in message
    with pytest.raises(TypeError):
        Optimization(fuse="conservative")
    # Typed usage is intact and the default stays StrictMath.
    assert isinstance(Optimization().math, StrictMath)
    assert Optimization(math=FastMath()).options()["math"] == "FastMath"


def test_output_policy_rejects_string_format_and_invalid_cadence():
    from pops.output import OutputPolicy
    from pops.time.schedule import subcycle, when

    with pytest.raises(TypeError) as excinfo:
        OutputPolicy(format="hdf5")
    assert "format" in str(excinfo.value) and "HDF5()" in str(excinfo.value)

    with pytest.raises(ValueError, match="valid output/checkpoint cadence"):
        OutputPolicy(cadence=when(lambda: True))
    with pytest.raises(ValueError, match="valid output/checkpoint cadence"):
        OutputPolicy(cadence=subcycle(2))


def test_output_runtime_public_validation_uses_configuration_errors():
    """TASK-005/TASK-056: output/checkpoint validation must not expose placeholders."""
    files = [
        "python/pops/output/policies.py",
        "python/pops/runtime/_output_driver.py",
        "python/pops/runtime/_system_io.py",
    ]
    offenders = []
    for rel in files:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        if "NotImplementedError" in text:
            offenders.append(rel)
    assert not offenders, (
        "output/checkpoint public validation must raise TypeError/ValueError with clear messages, "
        "not NotImplementedError:\n%s" % "\n".join(offenders)
    )


def test_public_authoring_validation_uses_configuration_errors():
    """TASK-005/TASK-039/040: public DSL/runtime validation must not expose placeholders."""
    files = [
        "python/pops/physics/_authoring_riemann.py",
        "python/pops/physics/board.py",
        "python/pops/time/values.py",
        "python/pops/time/program_authoring.py",
        "python/pops/time/program_local.py",
        "python/pops/runtime/_system_install.py",
    ]
    offenders = []
    for rel in files:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        if "NotImplementedError" in text:
            offenders.append(rel)
    assert not offenders, (
        "public authoring/runtime validation must raise TypeError/ValueError with clear messages, "
        "not NotImplementedError:\n%s" % "\n".join(offenders)
    )


def test_compiled_model_check_runtime_does_not_recreate_legacy_runtime_assembly():
    """TASK-001/003: CompiledModel.check_runtime must not route through private add-equation APIs."""
    from pops.codegen.loader import CompiledModel

    source = (REPO_ROOT / "python/pops/codegen/loader.py").read_text(encoding="utf-8")
    start = source.index("    def check_runtime(")
    end = source.index("    def inspect_amr(", start)
    body = source[start:end]
    for token in ("_add_equation", "Explicit(", "from pops import System"):
        assert token not in body

    compiled = CompiledModel(
        so_path="/tmp/missing.so",
        backend="production",
        adder="add_native_block",
        cons_names=["rho"],
        cons_roles=["Density"],
        prim_names=[],
        n_vars=1,
        gamma=None,
        n_aux=3,
        params={},
        caps={},
        abi_key="abi",
        model_hash="model",
        cxx="c++",
        std="20",
    )
    with pytest.raises(ValueError) as excinfo:
        compiled.check_runtime()
    msg = str(excinfo.value)
    assert "compile_problem" in msg and "System.install" in msg


def test_disc_domain_rejects_string_transport_mode():
    from pops.mesh.geometry import DiscDomain

    with pytest.raises(TypeError) as excinfo:
        DiscDomain(center=(0.0, 0.0), radius=0.2, mode="cutcell")
    assert "mode" in str(excinfo.value) and "CutCell()" in str(excinfo.value)


def test_physics_riemann_rejects_string_selector():
    import pops

    model = pops.physics.Model("riemann_string_guard")
    with pytest.raises(TypeError) as excinfo:
        model.riemann("hllc")
    assert "riemann" in str(excinfo.value) and "HLLC()" in str(excinfo.value)


def test_compiled_brick_without_a_manifest_is_a_clear_error():
    # The compiled-brick load path refuses a brick whose manifest does not exist. read_manifest
    # on a missing .json raises FileNotFoundError (an OSError); resolving a CompiledBrickRef whose
    # manifest is absent raises an explainable error before any runtime install.
    from pops.external import CompiledBrickRef, read_manifest

    with pytest.raises(OSError):  # FileNotFoundError is an OSError
        read_manifest("/nonexistent_pops_brick_manifest.json")

    ref = CompiledBrickRef(manifest="/nonexistent_pops_brick_manifest.json",
                           native_id="missing_brick")
    with pytest.raises((OSError, ValueError)):
        ref.resolve()
    # validate() turns the unresolvable manifest into an explainable ValueError, not a crash.
    with pytest.raises(ValueError) as excinfo:
        ref.validate()
    assert "missing_brick" in str(excinfo.value)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
