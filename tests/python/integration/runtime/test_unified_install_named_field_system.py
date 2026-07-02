"""ADC-479 Spec 5 C1-System: named elliptic fields lower on the uniform System.

PURE-PYTHON tests of the install-seam routing for a NAMED elliptic field (m.elliptic_field): a
field a block's model DECLARES routes through the shared system elliptic solver (set_poisson), while
an UNDECLARED field name is rejected LOUD against the declared set (never a silent drop). The
declared set is collected from CompiledModel.elliptic_field_names / a raw model's _elliptic_fields,
so the codegen->install carry is exercised without a real Kokkos compile.

Runs both under pytest and as a plain script; the CI runner executes it via the __main__ guard.
"""
import sys

try:
    import pops
    from pops.runtime._system_unified_install import _SystemUnifiedInstall
    from pops.codegen.loader import CompiledModel
except Exception as exc:  # noqa: BLE001
    print("skip test_unified_install_named_field_system (pops unavailable: %s)" % exc)
    sys.exit(0)


from tests.python.support.assertions import _check


class _RawModel:
    """A raw physics/dsl model stand-in exposing the _elliptic_fields mapping (m.elliptic_field)."""

    def __init__(self, fields=()):
        self._elliptic_fields = {n: {} for n in fields}


class _SolverHarness(_SystemUnifiedInstall):
    """Captures set_poisson calls without a real System (no engine, no Kokkos)."""

    def __init__(self):
        self.calls = []

    def set_poisson(self, **kw):
        self.calls.append(kw)


def test_compiled_model_carries_elliptic_field_names():
    """C1: CompiledModel carries the declared named-field set (elliptic_field_names)."""
    cm = CompiledModel(
        so_path="/fake.so", backend="production", adder="add_native_block",
        cons_names=["rho"], cons_roles=["Density"], prim_names=["rho"], n_vars=1,
        gamma=None, n_aux=0, params={}, caps={}, abi_key="k", model_hash="h",
        cxx="c++", std="c++20", elliptic_field_names=["psi", "chi"])
    _check(cm.elliptic_field_names == ["psi", "chi"], "CompiledModel keeps the declared names")
    # default: empty when none declared
    cm0 = CompiledModel(
        so_path="/fake.so", backend="production", adder="add_native_block",
        cons_names=["rho"], cons_roles=["Density"], prim_names=["rho"], n_vars=1,
        gamma=None, n_aux=0, params={}, caps={}, abi_key="k", model_hash="h",
        cxx="c++", std="c++20")
    _check(cm0.elliptic_field_names == [], "no declared field -> empty list")
    print("ok test_compiled_model_carries_elliptic_field_names")


def test_declared_elliptic_fields_collected_from_handle_and_instances():
    """C1: the declared set is gathered from the compiled handle's model AND the per-instance models,
    reading elliptic_field_names (CompiledModel) or _elliptic_fields (raw model) without compiling."""
    handle = type("CP", (), {"model": _RawModel(fields=("psi",))})()
    instances = {"ne": {"model": _RawModel(fields=("chi",))},
                 "ni": {"model": _RawModel(fields=("psi",))}}
    declared = _SystemUnifiedInstall._declared_elliptic_fields(handle, instances)
    _check(declared == {"psi", "chi"}, "union of handle + instance declared fields (got %r)" % declared)
    # a CompiledModel exposes elliptic_field_names instead of _elliptic_fields
    cm = CompiledModel(
        so_path="/fake.so", backend="production", adder="add_native_block",
        cons_names=["rho"], cons_roles=["Density"], prim_names=["rho"], n_vars=1,
        gamma=None, n_aux=0, params={}, caps={}, abi_key="k", model_hash="h",
        cxx="c++", std="c++20", elliptic_field_names=["theta"])
    declared2 = _SystemUnifiedInstall._declared_elliptic_fields(
        type("CP", (), {"model": None})(), {"b": {"model": cm}})
    _check(declared2 == {"theta"}, "CompiledModel.elliptic_field_names collected (got %r)" % declared2)
    print("ok test_declared_elliptic_fields_collected_from_handle_and_instances")


def test_install_solver_routes_default_poisson_field():
    """C1: the default Poisson field still routes to set_poisson (regression)."""
    h = _SolverHarness()
    h._install_solver("phi", "geometric_mg", frozenset())
    _check(h.calls and h.calls[0]["solver"] == "geometric_mg", "default field -> set_poisson")
    print("ok test_install_solver_routes_default_poisson_field")


def test_install_solver_routes_declared_named_field():
    """C1: a DECLARED named elliptic field routes through the shared elliptic solver (set_poisson)."""
    h = _SolverHarness()
    h._install_solver("psi", "geometric_mg", frozenset({"psi"}))
    _check(h.calls and h.calls[0]["solver"] == "geometric_mg",
           "a declared named field routes to set_poisson (shared solver)")
    print("ok test_install_solver_routes_declared_named_field")


def test_install_solver_rejects_undeclared_field():
    """C1: an UNDECLARED field name is a typo -- rejected LOUD, naming the declared set."""
    h = _SolverHarness()
    try:
        h._install_solver("psii", "geometric_mg", frozenset({"psi"}))
        raise AssertionError("an undeclared field name must raise")
    except ValueError as exc:
        _check("psii" in str(exc), "the reject names the offending field")
        _check("psi" in str(exc), "the reject names the declared set")
        _check("elliptic_field" in str(exc), "the reject points at m.elliptic_field")
    _check(not h.calls, "no set_poisson call on a rejected field")
    print("ok test_install_solver_rejects_undeclared_field")


def test_case_validate_accepts_named_field():
    """C1: a Case with a named non-Poisson field VALIDATES (the whitelist reject is removed)."""
    class _M:
        def validate(self, context=None):
            return True

        def requirements(self):
            return {}

        def capabilities(self):
            return {}

    class _F:
        def validate(self, context=None):
            return True

        def requirements(self):
            return {}

    case = pops.Case().block("ne", physics=_M())
    case._fields["psi"] = _F()
    _check(case.validate() is True, "a named non-Poisson field validates (C1-System)")
    print("ok test_case_validate_accepts_named_field")


def _run_all():
    funcs = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for fn in funcs:
        fn()
    print("\nall %d test(s) passed" % len(funcs))


if __name__ == "__main__":
    _run_all()
