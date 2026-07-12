"""ADC-652: cadence inputs retain exact authoring data until the native boundary."""
from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

import pytest

from pops.ir import ScalarLiteral
from pops.identity import make_identity
from pops.model.bind_schema import BindSchema
from pops.runtime._bound_snapshot import build_uniform_snapshot
from pops.runtime._system_unified_install import _SystemUnifiedInstall
from pops.time import CompiledTime


@pytest.mark.parametrize("field", ["substeps", "stride"])
def test_compiled_time_rejects_bool_counts(field):
    with pytest.raises(ValueError, match="positive int"):
        CompiledTime(**{field: True})


@pytest.mark.parametrize("cfl", [Fraction(1, 3), Decimal("0.125")])
def test_compiled_time_retains_exact_cfl_until_runtime_lowering(cfl):
    cadence = CompiledTime(cfl=cfl)

    assert cadence.cfl == cfl
    assert type(cadence.cfl) is type(cfl)


@pytest.mark.parametrize("cfl", [True, float("nan"), float("inf")])
def test_compiled_time_rejects_non_real_or_non_finite_cfl(cfl):
    with pytest.raises((TypeError, ValueError)):
        CompiledTime(cfl=cfl)


@pytest.mark.parametrize("cfl", [0, -1, Fraction(-1, 2)])
def test_compiled_time_rejects_non_positive_cfl(cfl):
    with pytest.raises(ValueError, match="must be > 0"):
        CompiledTime(cfl=cfl)


def test_compiled_time_refuses_to_erase_unit_metadata():
    cfl = ScalarLiteral("rational", (1, 2), unit="s")

    with pytest.raises(TypeError, match="cannot erase a scalar unit"):
        CompiledTime(cfl=cfl)


def test_uniform_runtime_converts_exact_cfl_only_at_install_boundary():
    class FakeSystem:
        def set_program_cadence(self, substeps, stride):
            self.native_cadence = (substeps, stride)

    system = FakeSystem()
    cadence = CompiledTime(substeps=2, stride=3, cfl=Fraction(1, 3))

    _SystemUnifiedInstall._install_cadence(system, cadence)

    assert system.native_cadence == (2, 3)
    assert system._program_cadence_cfl == float(Fraction(1, 3))
    schema = BindSchema(())
    compiled = type("Compiled", (), {
        "semantic_identity": make_identity("semantic", {"model": "time-test"}),
        "artifact_identity": make_identity("artifact", {"binary": "time-test"}),
    })()
    engine = type("Engine", (), {"_output_policies": (), "_diagnostic_measures": ()})()
    manifest = build_uniform_snapshot(
        engine, compiled, {}, {}, {}, cadence, {},
        schema.resolve_bind({}, compile_values={})).to_dict()
    assert manifest["cadence"] == {
        "kind": "compiled-time", "substeps": 2, "stride": 3,
        "cfl": {"rational": [1, 3]},
    }
