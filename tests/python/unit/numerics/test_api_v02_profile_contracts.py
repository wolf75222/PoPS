"""Coverage is an equation contract, never an implicit numerical PASS."""

import ast
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from examples.api_v02.profile_contracts import (
    SELECTED_PROFILES,
    ComponentBinding,
    ImplementationManifest,
    MissingComponentsError,
    NumericalReceipt,
    matching_receipts,
    named_profiles,
    selected_profile,
)

ROOT = Path(__file__).resolve().parents[4]


def _source(symbol):
    module, name = symbol.split(":")
    path = ROOT.joinpath(*module.split(".")).with_suffix(".py")
    return path, name


def _manifest(contract):
    # A fixture explicitly binds review obligations to the selected source.
    # This is not an execution receipt or an automatic mathematical proof.
    path, _ = _source(contract.builder.symbol)
    return ImplementationManifest(
        contract.key,
        tuple(
            ComponentBinding(
                item.identifier, contract.builder.symbol, sha256(path.read_bytes()).hexdigest()
            )
            for item in contract.required_components
        ),
    )


def test_exact_28_models_12_methods_and_referenced_profiles_exist():
    assert set(SELECTED_PROFILES) == {
        *(f"M{i:02}" for i in range(1, 29)),
        *(f"T{i:02}" for i in range(1, 13)),
    }
    for identifier, contract in SELECTED_PROFILES.items():
        assert contract.required_components
        assert contract.assumptions
        assert contract.unestablished_properties
        names = [profile.name for profile in named_profiles(identifier)]
        assert len(names) == len(set(names))
        for profile in named_profiles(identifier):
            path, symbol = _source(profile.builder.symbol)
            definitions = {
                node.name: node
                for node in ast.parse(path.read_bytes()).body
                if isinstance(node, ast.FunctionDef)
            }
            assert symbol in definitions, profile.builder.symbol
            function = definitions[symbol]
            if function.args.kwarg is None:
                arguments = (
                    function.args.posonlyargs + function.args.args + function.args.kwonlyargs
                )
                assert set(profile.builder.kwargs()) <= {argument.arg for argument in arguments}
        for receiver in contract.numerical_receivers:
            assert (ROOT / receiver).is_file(), receiver


def test_missing_equation_is_rejected_without_demanding_a_universal_theorem():
    contract = selected_profile("M09")
    manifest = _manifest(contract)
    missing_midpoint = replace(
        manifest,
        bindings=tuple(binding for binding in manifest.bindings if binding.component != "midpoint"),
    )
    with pytest.raises(MissingComponentsError) as error:
        contract.require_components(missing_midpoint, source_root=ROOT)
    assert error.value.missing == ("midpoint",)
    assert contract.require_components(manifest, source_root=ROOT) is manifest
    assert (
        contract.unestablished_properties
    )  # Still present after implementation coverage succeeds.
    assert (
        matching_receipts(
            contract,
            (),
            source_sha256="a" * 64,
            artifact_sha256="b" * 64,
            parameters_sha256="c" * 64,
            backend="Serial",
            ranks=1,
        )
        == ()
    )


def test_changed_equation_source_invalidates_old_manifest():
    contract = selected_profile("M01")
    manifest = _manifest(contract)
    stale = replace(manifest.bindings[0], source_sha256="0" * 64)
    with pytest.raises(ValueError, match="digest changed"):
        contract.require_components(
            replace(manifest, bindings=(stale, *manifest.bindings[1:])), source_root=ROOT
        )


def test_wrong_variant_duplicates_and_unknown_components_are_not_coverage():
    contract = selected_profile("M21")
    manifest = _manifest(contract)
    with pytest.raises(ValueError, match="different selected profile"):
        contract.missing_components(replace(manifest, profile="M21:legacy-slab-fixed-kperp"))
    with pytest.raises(ValueError, match="unambiguous"):
        contract.missing_components(
            replace(manifest, bindings=(*manifest.bindings, manifest.bindings[0]))
        )
    bad = replace(manifest.bindings[0], component="universal-convergence-theorem")
    with pytest.raises(ValueError, match="unknown component"):
        contract.missing_components(replace(manifest, bindings=(bad, *manifest.bindings[1:])))


def test_hyqmom_nonhyperbolicity_is_a_model_property_not_a_missing_api_component():
    contract = selected_profile("M16")
    assert any("nonhyperbolic" in property for property in contract.model_properties)
    assert not any(
        "hyperbolic" in component.identifier for component in contract.required_components
    )
    assert contract.missing_components(_manifest(contract)) == ()


def test_receipts_do_not_transfer_across_code_artifact_parameters_backend_or_ranks():
    contract = selected_profile("M23")
    receipt = NumericalReceipt(
        contract.key,
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "Serial",
        1,
        "receipt://explicit-test-fixture",
        ("original-divB", "every-field-oracle"),
        "pass",
    )
    failed = replace(receipt, outcome="fail", checks=("original-divB",))
    identity = dict(
        source_sha256="a" * 64,
        artifact_sha256="b" * 64,
        parameters_sha256="c" * 64,
        backend="Serial",
        ranks=1,
    )
    assert matching_receipts(contract, (receipt, failed), **identity) == (receipt, failed)
    for key, value in (
        ("source_sha256", "d" * 64),
        ("artifact_sha256", "d" * 64),
        ("parameters_sha256", "d" * 64),
        ("backend", "CUDA"),
        ("ranks", 2),
    ):
        assert matching_receipts(contract, (receipt,), **dict(identity, **{key: value})) == ()
    with pytest.raises(ValueError, match="name the checks"):
        replace(receipt, checks=())


def test_old_reduced_profiles_remain_explicitly_named():
    for identifier in (
        "M09",
        "M19",
        "M20",
        "M21",
        "M23",
        "M24",
        "M28",
        "T02",
        "T03",
        "T04",
        "T06",
        "T09",
        "T10",
        "T11",
        "T12",
    ):
        contract = selected_profile(identifier)
        assert contract.compatibility_profiles, identifier
        assert all(profile.scope for profile in contract.compatibility_profiles)
    m24 = selected_profile("M24")
    assert m24.builder.kwargs()["pressure_relaxation"] > 0
    assert m24.builder.kwargs()["velocity_relaxation"] > 0
