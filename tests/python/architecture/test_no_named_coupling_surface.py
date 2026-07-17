"""ADC-595: a new coupling needs no new public C++ method (source-only guard).

The named inter-species couplings (ionization / collision / thermal exchange) used to be hard-coded
C++ methods (``System::add_ionization`` / ``add_collision`` / ``add_thermal_exchange``) with a pybind
``.def`` each; the raw coupled-source bytecode ABI was a PUBLIC ``add_coupled_source`` binding. After
ADC-595 the named couplings are Python PRESETS lowering to the generic coupled source, and the raw
bytecode ABI is an INTERNAL escape hatch (``_add_coupled_source``); a coupling registers through the one
typed ``add_coupling_operator``. This source-only test pins that surface so a new coupling cannot
re-introduce a bespoke C++ method:

  - ``include/pops/runtime/system.hpp`` declares none of the three named coupling methods;
  - ``python/bindings/core/init/init_system.cpp`` binds none of them, and the raw bytecode ABI is bound
    only as the INTERNAL ``_add_coupled_source`` (no public ``add_coupled_source`` def);
  - ``engine.Ionization`` / ``Collision`` / ``ThermalExchange`` survive only as preset descriptors that
    carry data (no ``add_*`` C++ dispatch method);
  - the typed ``add_coupling_operator`` entry IS present (the one generic registration path).

The test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
import ast
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POPS = REPO_ROOT / "python" / "pops"
BINDINGS = REPO_ROOT / "python" / "bindings"
INCLUDE = REPO_ROOT / "include" / "pops"

_NAMED = ("add_ionization", "add_collision", "add_thermal_exchange")


def _read(rel_root, rel):
    return (rel_root / rel).read_text(encoding="utf-8")


def test_system_header_declares_no_named_coupling_method():
    src = _read(INCLUDE, "runtime/system.hpp")
    # A method declaration looks like "void add_ionization(" -- a mere comment mention (the removal
    # note) uses the bare name without a "(", so we key on the call-signature shape.
    for name in _NAMED:
        assert not re.search(r"\bvoid\s+" + name + r"\s*\(", src), (
            "System::%s must be removed (ADC-595): the named couplings are Python presets, not C++ "
            "methods" % name)


def test_amr_system_header_declares_no_named_coupling_method():
    src = _read(INCLUDE, "runtime/amr_system.hpp")
    for name in _NAMED:
        assert not re.search(r"\bvoid\s+" + name + r"\s*\(", src), (
            "AmrSystem::%s must not exist (ADC-595)" % name)


def test_init_system_binds_no_named_coupling_and_internalizes_raw_abi():
    src = _read(BINDINGS, "core/init/init_system.cpp")
    for name in _NAMED:
        assert ('"%s"' % name) not in src, (
            'init_system.cpp must not .def("%s") (ADC-595): the named couplings are presets' % name)
    # The raw bytecode ABI is INTERNAL (leading underscore); no public add_coupled_source def.
    assert '"_add_coupled_source"' in src, (
        "the raw coupled-source bytecode ABI must be bound as the INTERNAL _add_coupled_source "
        "(ADC-595)")
    assert '"add_coupled_source"' not in src, (
        "the raw coupled-source bytecode ABI must not be PUBLIC (rename to _add_coupled_source, "
        "ADC-595): end users register through add_coupling / add_coupling_operator")
    # The one generic typed registration path IS present.
    assert '"add_coupling_operator"' in src, (
        "the typed add_coupling_operator entry must be bound (the single generic registration path)")


def test_init_amr_internalizes_raw_abi():
    src = _read(BINDINGS, "core/init/init_amr.cpp")
    assert '"_add_coupled_source"' in src and '"add_coupled_source"' not in src, (
        "the AMR raw coupled-source bytecode ABI must be INTERNAL (_add_coupled_source), ADC-595")
    assert '"add_coupling_operator"' in src, "the AMR typed add_coupling_operator entry must be bound"


def test_named_couplings_survive_only_as_preset_descriptors():
    tree = ast.parse(_read(POPS, "runtime/_bricks_scheme.py"),
                     filename="_bricks_scheme.py")
    classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    for name in ("Ionization", "Collision", "ThermalExchange"):
        assert name in classes, "%s must survive as a preset descriptor class" % name
        methods = {n.name for n in classes[name].body if isinstance(n, ast.FunctionDef)}
        # A descriptor carries data only (an __init__); it must expose no add_* dispatch method.
        dispatch = {m for m in methods if m.startswith("add_")}
        assert not dispatch, (
            "%s must be a plain preset descriptor (data only), not carry a C++ dispatch method: %s"
            % (name, sorted(dispatch)))
