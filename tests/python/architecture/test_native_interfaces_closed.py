"""CF2 and FC6b (spec 0.3; Linear N2): the substitution protocol accepts natives only through the
closed list of generic interfaces.

The native interfaces of b6 are declared once, in ``schemas/component_catalog.v2.json``
("native_interface_abis"), and generated into ``python/pops/_generated_component_interfaces.py``.
``frozen_engine/cf2_interfaces.toml`` states the closed list of FC6b and where each b6 interface
stands: mapped to an entry of the list, kept as the binding of a solver substitution, or an
exception with the issue that takes it out of the protocol. An interface added outside the list
fails; the exceptions only shrink (MG6). The CF0 placement list names, for each native still in a
core path, the entry of the closed list that will bind it.

The test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
import ast
import json
import tomllib

from tests.python.architecture.frozen_engine import ratchet, scan

INTERFACES = scan.HERE / "cf2_interfaces.toml"
PLACEMENT = scan.HERE / "cf0_native_placement.toml"
CATALOG = scan.REPO_ROOT / "schemas" / "component_catalog.v2.json"
GENERATED = scan.REPO_ROOT / "python" / "pops" / "_generated_component_interfaces.py"

# FC6b, verbatim in its order: linear action, preconditioner apply, Krylov driver over the
# vector-space interface, multigrid cycle over generated level operators, FFT with a derived symbol
# (one entry, IR15), dense and banded factorization, loop over a generated body, and the
# level-transfer class (reflux ledger and restriction with generated weights).
FC6B_CLOSED_LIST = (
    "linear_action", "preconditioner_apply", "krylov_driver", "multigrid_cycle", "fft",
    "dense_banded_factorization", "loop_over_generated_body", "level_transfer",
)

# The interfaces of b6 outside the closed list (cf2_interfaces.toml [[exception]]). This set only
# shrinks (MG6); never add a name here.
B6_INTERFACE_EXCEPTIONS = frozenset({
    "numerical_flux", "ghost_boundary", "field_boundary_closure", "boundary_flux",
    "accepted_initial_ghost", "transfer", "tagger", "clustering", "field_topology", "writer",
})


def _interfaces():
    return tomllib.loads(INTERFACES.read_text(encoding="utf-8"))


def catalog_interfaces():
    """Names of the native interfaces that the protocol of the tree declares."""
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    return [row["name"] for row in catalog["native_interface_abis"]]


def generated_interfaces():
    """Names in the generated NATIVE_COMPONENT_INTERFACES tuple, read from its source."""
    tree = ast.parse(GENERATED.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "NATIVE_COMPONENT_INTERFACES"
                for target in node.targets):
            rows = ast.literal_eval(node.value)
            return [row["name"] for row in rows]
    raise AssertionError("NATIVE_COMPONENT_INTERFACES not found in %s" % GENERATED.name)


def classify(declared, table):
    """(unclassified, classified twice, useless exceptions) of the declared interface names."""
    mapped = set(table.get("mapped", {}))
    binding = set(table.get("binding", {}))
    exceptions = [row["name"] for row in table.get("exception", [])]
    homes = {}
    for group, names in (("mapped", mapped), ("binding", binding), ("exception", exceptions)):
        for name in names:
            homes.setdefault(name, []).append(group)
    unclassified = sorted(set(declared) - set(homes))
    twice = sorted(name for name, groups in homes.items() if len(groups) > 1)
    useless = sorted(set(homes) - set(declared))
    return unclassified, twice, useless


def test_closed_list_is_the_list_of_fc6b():
    table = _interfaces()
    assert tuple(table["closed"]["interfaces"]) == FC6B_CLOSED_LIST, (
        "cf2_interfaces.toml [closed] must be the closed list of FC6b, in its order, with one FFT "
        "entry; changing it is a change of spec 0.3, not of this file")


def test_every_declared_interface_is_in_the_closed_list_or_a_named_exception():
    table = _interfaces()
    declared = catalog_interfaces()
    unclassified, twice, useless = classify(declared, table)
    assert not unclassified, (
        "CF2/FC6b: the protocol declares native interfaces outside the closed list of FC6b: %s. "
        "A native enters only through %s (cf2_interfaces.toml)." % (unclassified, FC6B_CLOSED_LIST))
    assert not twice, "an interface has more than one standing in cf2_interfaces.toml: %s" % twice
    assert not useless, (
        "MG6: cf2_interfaces.toml names interfaces that the protocol no longer declares; remove "
        "them: %s" % useless)


def test_mapped_interfaces_target_the_closed_list():
    table = _interfaces()
    wrong = {name: target for name, target in table.get("mapped", {}).items()
             if target not in FC6B_CLOSED_LIST}
    assert not wrong, "mapped interfaces must name an entry of the FC6b closed list: %s" % wrong


def test_binding_is_only_the_solver_binding():
    assert set(_interfaces().get("binding", {})) <= {"field_solver"}, (
        "FC6b keeps field_solver only as the binding of a solver substitution to its reference")


def test_interface_exceptions_only_shrink_and_name_their_issue():
    rows = _interfaces().get("exception", [])
    names = {row["name"] for row in rows}
    assert names <= B6_INTERFACE_EXCEPTIONS, (
        "MG6: new interface exceptions are refused: %s" % sorted(names - B6_INTERFACE_EXCEPTIONS))
    wrong = [row for row in rows if row.get("issue") not in ratchet.KNOWN_ISSUES
             or not row.get("reason")]
    assert not wrong, "each interface exception names a known issue and a reason: %s" % wrong


def test_generated_interfaces_match_the_catalog():
    assert generated_interfaces() == catalog_interfaces(), (
        "python/pops/_generated_component_interfaces.py drifted from the catalogue; the closed-list "
        "check reads the catalogue")


def test_placement_entries_bind_through_the_closed_list():
    natives = tomllib.loads(PLACEMENT.read_text(encoding="utf-8"))["native"]
    wrong = [(row["id"], row["interface"]) for row in natives
             if row["interface"] not in FC6B_CLOSED_LIST + ("none",)]
    assert not wrong, "CF0 placement entries must name an FC6b interface (or none): %s" % wrong


def test_classification_refuses_an_interface_outside_the_list():
    """Control: a physics-level interface added to the catalogue is caught."""
    table = _interfaces()
    unclassified, _, _ = classify(catalog_interfaces() + ["riemann_solver"], table)
    assert unclassified == ["riemann_solver"]
    _, _, useless = classify([name for name in catalog_interfaces() if name != "tagger"], table)
    assert useless == ["tagger"]
