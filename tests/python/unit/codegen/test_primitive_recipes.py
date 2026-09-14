"""Canonical primitive recipes retain exact physical inputs without executing callbacks."""
import pytest

from pops._ir.expr import Var
from pops.codegen.resolved_operations import build_resolved_operations
from pops.model import Module
from pops.model.primitive_recipes import resolve_primitive_recipe
from pops.physics._facade import Model


def test_derived_density_primitive_has_exact_input_and_omits_unused_field():
    facade = Model("density_recipe")
    rho, unused_state = facade.conservative_vars("rho", "q_unused")
    facade.aux("unused")
    density = facade.primitive("density_only", 2 * rho)
    composed = facade.primitive("density_composed", density + 1)
    facade.source_term("forcing", (composed, 0))
    module = facade.module
    assert resolve_primitive_recipe(module, "density_only") is facade._m.prim_defs["density_only"]
    assert resolve_primitive_recipe(module, "density_composed") is facade._m.prim_defs["density_composed"]
    operation, = build_resolved_operations(module).operations
    assert [(access.kind, access.components, access.complete) for access in operation.inputs] == [
        ("state", ("rho",), False)]
    assert resolve_primitive_recipe(module, "unknown") is None
    with pytest.raises(TypeError):
        module.primitive_recipes()["density_only"] = rho
    before = module.module_hash()
    module.freeze()
    assert module.module_hash() == before
    assert module.primitive_recipes()["density_only"] is facade._m.prim_defs["density_only"]


def test_recipe_cycles_callbacks_and_foreign_qualified_references_are_refused_atomically():
    module = Module("recipes")
    with pytest.raises(ValueError, match="recipe cycle"):
        module.set_primitive_recipes({"a": Var("b", "prim"), "b": Var("a", "prim")})
    assert not module.primitive_recipes()
    with pytest.raises(TypeError, match="callbacks are forbidden"):
        module.set_primitive_recipes({"a": lambda: 1})
    assert not module.primitive_recipes()
    foreign = Module("foreign")
    state = foreign.state_space("U", ("rho",))
    rho, = foreign.state_symbols(state)
    with pytest.raises(ValueError, match="foreign qualified"):
        module.set_primitive_recipes({"a": rho})
    assert not module.primitive_recipes()


def test_recipe_source_changes_module_hash_and_does_not_capture_external_mapping():
    first, second = Module("same"), Module("same")
    recipes = {"density": Var("rho", "cons")}
    first.set_primitive_recipes(recipes)
    second.set_primitive_recipes({"density": 2 * Var("rho", "cons")})
    assert first.module_hash() != second.module_hash()
    recipes["density"] = Var("momentum", "cons")
    assert resolve_primitive_recipe(first, "density").name == "rho"
    with pytest.raises(ValueError, match="already declared"):
        first.set_primitive_recipes(recipes)
    with pytest.raises(TypeError, match="exact source Module"):
        resolve_primitive_recipe(object(), "density")


def test_legacy_formula_manifest_projection_is_stable_without_relaxing_reference_resolution():
    from pops.model import ModuleManifest

    def manifest():
        facade = Model("formula_snapshot")
        rho, = facade.conservative_vars("rho")
        density = facade.primitive("density", 2 * rho)
        facade.source_term("forcing", (density,))
        return facade.module.manifest().to_dict()

    first = manifest()
    assert first == manifest()
    assert first["expressions"]["operators"]["forcing"]
    assert first["expressions"]["primitives"]
    assert ModuleManifest.from_dict(first).to_dict() == first
    with pytest.raises(TypeError, match="free-name Var"):
        Var("rho", "cons").resolve_references(lambda handle: handle._resolved())
