"""ADC-655: CompiledModel rejects hidden authoring state and authenticates compile results."""
from __future__ import annotations

import gc
import weakref
from types import MappingProxyType

import pytest

from pops.codegen._compiled_model_identity import model_compile_identity
from pops.codegen._orchestration_compile import compile_install_model
from pops.codegen.loader import CompiledModel
from pops.problem._snapshot import AuthoringSnapshot, attach_problem_snapshot


class _SourceModel:
    def __init__(self, name="source", *, returned_hash=None, extra=None):
        self.name = name
        self.returned_hash = returned_hash
        self.extra = extra

    def _model_hash(self):
        return "structural:%s" % self.name

    def compile(self, *, backend, target, **kwargs):
        identity_source = _SourceModel(self.returned_hash or self.name)
        compiled = _compiled(
            target=target,
            model_hash=identity_source._model_hash(),
            identity=model_compile_identity(identity_source),
        )
        if self.extra is not None:
            compiled.extension = self.extra
        return compiled


def _compiled(*, target="system", model_hash="structural:source", identity=None):
    return CompiledModel(
        so_path="<compiled-model-boundary>",
        backend="production",
        cons_names=("u",),
        cons_roles=("Scalar",),
        prim_names=("u",),
        n_vars=1,
        gamma=None,
        n_aux=0,
        params={},
        caps={"cpu": True, "amr": target == "amr_system"},
        abi_key="abi",
        model_hash=model_hash,
        cxx="c++",
        std="c++23",
        target=target,
        definition_identity=identity,
    )


def _compile(model):
    return compile_install_model("fluid", model, "production", "system", {})


def test_compile_result_identity_must_match_exact_source_model():
    with pytest.raises(ValueError, match="different structural model"):
        _compile(_SourceModel("expected", returned_hash="other"))


def test_compile_result_cannot_omit_structural_identity():
    source = _SourceModel()
    source.compile = lambda **kwargs: _compiled(model_hash=source._model_hash())

    with pytest.raises(TypeError, match="definition_identity must be a mapping"):
        _compile(source)


def test_compile_result_identity_and_model_hash_must_agree():
    source = _SourceModel()

    def broken_compile(*, backend, target, **kwargs):
        return _compiled(
            target=target,
            model_hash="tampered",
            identity=model_compile_identity(source),
        )

    source.compile = broken_compile
    with pytest.raises(ValueError, match="model_hash disagrees"):
        _compile(source)


def test_matching_identity_is_scalar_only_and_does_not_retain_source():
    source = _SourceModel()
    source_ref = weakref.ref(source)
    compiled = _compile(source)
    del source
    gc.collect()

    assert source_ref() is None
    assert dict(compiled.definition_identity) == {
        "protocol": "pops.compiled-model-identity.v1",
        "model_hash": "structural:source",
        "module_hash": None,
    }


def test_subclass_slot_cannot_hide_an_authoring_builder():
    class HiddenLoader(CompiledModel):
        __slots__ = ("hidden",)

        def __getattribute__(self, name):
            if name == "hidden":
                return "spoofed-safe-value"
            return super().__getattribute__(name)

    source = _SourceModel()
    base = _compiled(identity=model_compile_identity(source))
    hidden = HiddenLoader(
        base.so_path, base.backend, base.cons_names, base.cons_roles, base.prim_names,
        base.n_vars, base.gamma, base.n_aux, base.params, base.caps,
        base.abi_key, base.model_hash, base.cxx, base.std,
        definition_identity=base.definition_identity,
    )
    hidden.hidden = source
    source.compile = lambda **kwargs: hidden

    with pytest.raises(TypeError, match="exact CompiledModel"):
        _compile(source)


def test_added_field_with_noop_freeze_is_rejected():
    class NoOpFrozen:
        def __init__(self):
            self.payload = []

        def freeze(self):
            return self

    with pytest.raises(TypeError, match=r"freeze\(\) did not make retained state immutable"):
        _compile(_SourceModel(extra=NoOpFrozen()))


def test_added_nested_mapping_cannot_hide_authoring_state():
    with pytest.raises(TypeError, match="extension.*unsupported live object"):
        _compile(_SourceModel(extra={"nested": [_SourceModel("hidden")]}))


def test_subclass_noop_seal_cannot_bypass_canonical_boundary():
    class NoOpSealLoader(CompiledModel):
        def _seal(self):
            return None

        def __getattribute__(self, name):
            if name == "_sealed" and object.__getattribute__(self, "__dict__").get("armed"):
                return True
            return super().__getattribute__(name)

        def __setattr__(self, name, value):
            if name == "params" and object.__getattribute__(self, "__dict__").get("armed"):
                return
            super().__setattr__(name, value)

    source = _SourceModel()
    loader = NoOpSealLoader(
        "<noop-seal>", "production", "add_native_block", ("u",), ("Scalar",), ("u",),
        1, None, 0, {}, {"cpu": True}, "abi", source._model_hash(), "c++", "c++23",
        definition_identity=model_compile_identity(source),
    )
    object.__setattr__(loader, "armed", True)

    attach_problem_snapshot(loader, AuthoringSnapshot({"model": "source"}))

    stored = object.__getattribute__(loader, "__dict__")
    assert stored["_sealed"] is True
    assert isinstance(stored["params"], MappingProxyType)
    with pytest.raises(AttributeError, match="immutable"):
        loader.caps = {}
