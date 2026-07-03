"""ADC-527 / ADC-625: the typed DescriptorProtocol result objects exist and are the ONE form.

Source-only guards (no ``import pops`` / no ``_pops``) on the result-object module:

* ``python/pops/descriptors_report.py`` defines the four typed result objects (RequirementSet,
  CapabilitySet, LoweredDescriptor, ValidationReport) plus Requirement / ValidationIssue ;
* ADC-625: RequirementSet / CapabilitySet / LoweredDescriptor are TYPED objects, NOT ``dict``
  subclasses -- ``to_dict()`` is the only mapping bridge, so a caller reads them through the typed
  accessors (no dict emulation crutch can come back) ;
* ``pops.descriptors`` re-exports them, and ``pops._inspect`` defines the ``inspect`` dispatcher ;
* the base ``Descriptor`` returns the typed objects (no bare ``{}`` / ``dict()`` literal for
  requirements / capabilities / lower).

The test reads the source tree only; it does not import ``pops`` or ``_pops``.
"""
import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POPS = REPO_ROOT / "python" / "pops"

RESULT_OBJECTS = ("Requirement", "RequirementSet", "CapabilitySet", "LoweredDescriptor",
                  "ValidationIssue", "ValidationReport")
TYPED_RESULT_OBJECTS = ("RequirementSet", "CapabilitySet", "LoweredDescriptor")


def _classes(path):
    tree = ast.parse(path.read_text(), str(path))
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def test_descriptors_report_defines_the_typed_result_objects():
    path = POPS / "descriptors_report.py"
    assert path.exists(), "python/pops/descriptors_report.py must exist (ADC-527 result objects)"
    classes = _classes(path)
    for name in RESULT_OBJECTS:
        assert name in classes, "descriptors_report.py must define %r (ADC-527)" % name


def test_result_objects_are_typed_not_dict_subclasses():
    # ADC-625: the ONE final form is a typed object, never a dict subclass. A dict base would let
    # the dict-emulation crutch (x[key] / x.get / iteration) creep back; forbid it structurally.
    classes = _classes(POPS / "descriptors_report.py")
    for name in TYPED_RESULT_OBJECTS:
        bases = [b.id for b in classes[name].bases if isinstance(b, ast.Name)]
        assert "dict" not in bases, (
            "%s must NOT subclass dict (ADC-625): it is a typed object; the only mapping bridge is "
            "to_dict()" % name)


def test_descriptors_reexports_result_objects():
    src = (POPS / "descriptors.py").read_text()
    for name in RESULT_OBJECTS:
        assert name in src, "pops.descriptors must re-export %r (the one descriptor home)" % name


def test_inspect_dispatcher_module_exists():
    path = POPS / "_inspect.py"
    assert path.exists(), "python/pops/_inspect.py must define the pops.inspect(obj) dispatcher"
    tree = ast.parse(path.read_text(), str(path))
    fns = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "inspect" in fns, "pops._inspect must define inspect(obj)"
    # And the top-level pops package re-exports it.
    init = (POPS / "__init__.py").read_text()
    assert "from ._inspect import inspect" in init, "pops.__init__ must re-export inspect"
    assert '"inspect"' in init, "pops.__all__ must list inspect"


def test_base_descriptor_returns_typed_objects_not_bare_dicts():
    # The base Descriptor.requirements/capabilities/lower must return the typed objects (ADC-527),
    # not a bare {} literal -- so every Descriptor subclass is auto-conform.
    src = (POPS / "_descriptor_protocol.py").read_text()
    tree = ast.parse(src, "_descriptor_protocol.py")
    base = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "Descriptor")
    method_src = {}
    for node in base.body:
        if isinstance(node, ast.FunctionDef):
            method_src[node.name] = ast.get_source_segment(src, node) or ""
    for method, typed in (("requirements", "RequirementSet"),
                          ("capabilities", "CapabilitySet"),
                          ("lower", "LoweredDescriptor")):
        assert typed in method_src.get(method, ""), (
            "Descriptor.%s must return a %s (ADC-527), not a bare dict" % (method, typed))
