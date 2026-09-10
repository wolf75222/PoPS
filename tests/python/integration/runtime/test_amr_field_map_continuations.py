"""Repeated public moment/field/pullback barriers on independent composite hierarchies."""
from __future__ import annotations

import json

import numpy as np
import pops
import pytest

from tests.python.integration.runtime.test_interstage_physical_maps import resolve_interstage_maps
from tests.python.integration.runtime.test_amr_physical_maps import _internal_map_image


def _history_image(instance):
    native = instance._executor.executor_for_block("integral")._s
    return tuple((name, level, native.history_initialized(name, level),
                  native.history_fill_count(name, level),
                  tuple((float(native.history_slot_dt(name, level, slot)).hex(),
                         np.asarray(native.history_global(name, level, slot)).tobytes())
                        for slot in range(native.history_depth(name))))
                 for name in sorted(native.history_names())
                 for level in native.history_levels(name))


@pytest.mark.parametrize("reverse", (False, True))
def test_public_amr_maps_and_repeated_field_solves_share_one_region_driver(tmp_path, reverse):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_slicing import slice_program
    from pops.codegen.program_emit_field_problem import hierarchy_field_solve
    from pops.time.references import canonical_handle

    plan = resolve_interstage_maps(tmp_path, adaptive=True, with_fields=True, reverse=reverse)
    program = slice_program(plan.time, ("integral",))
    graph = ProgramModelGraph.from_resolved_blocks(tuple(block for block in plan.blocks
                                                        if block.name == "integral"))
    source = emit_cpp_program(program, model_graph=graph, target="amr_system")
    solves = tuple(value for value in program._values if value.op == "solve_linear")
    assert len(solves) == 2 and solves[0].point != solves[1].point
    assert tuple(value.id for value in solves[0].inputs[1].inputs) != \
        tuple(value.id for value in solves[1].inputs[1].inputs)
    assert all("hierarchy_field_identity" in value.attrs for value in solves)
    for solve in solves:
        coefficients = solve.inputs[0].attrs["apply_result"].inputs[2]
        assert hierarchy_field_solve(coefficients) is solve
        assert "hierarchy_field_coefficients" not in solve.attrs
    history_names = {"half driven moment history", "late driven moment history"}
    assert program._histories == dict.fromkeys(history_names, 1)
    assert {(row["name"], row["depth"], row["ring_slots"])
            for row in program.temporal_manifest()["histories"]} == {
                (name, 1, 2) for name in history_names}
    for name in history_names:
        assert source.count("ctx.store_history(%s," % json.dumps(name)) == 1
    assert source.count("ctx.rotate_histories(") == 1
    assert all(canonical_handle(state).block_ref.local_id == "integral"
               for state in program._history_state_refs.values())
    for other in ("population", "extended"):
        assert not slice_program(plan.time, (other,))._histories
    broken = plan.time._rebuild(lambda _value: True)
    del broken._history_state_refs["half driven moment history"]
    with pytest.raises(ValueError, match="histories require exact qualified state ownership"):
        slice_program(broken, ("integral",))
    assert source.count("ctx.suspend_hierarchy_barrier(") == 4
    assert source.count("ctx.solve_hierarchy_field(") == 2
    assert source.count("if (ctx.level() == 0) ctx.begin_staged_field_publications();") == 2
    assert source.count("ctx.publish_staged_field_components();") == 2
    assert source.count("ctx.suspend_map(") == 3
    assert "ctx.advance_mapping_hierarchy(dt" in source
    assert "ctx.advance_synchronized_hierarchy(" not in source
    assert "ctx.advance_hierarchy(" not in source
    # An lvalue wrapper for a context-owned field must not copy its MultiFab into a closure.
    for value in program._values:
        if value.op in ("field_problem_load", "field_problem_coefficients", "field_component"):
            assert "&program_field_%d" % value.id in source


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("reverse", (False, True))
def test_native_repeated_amr_field_maps_consume_current_stage_and_restart(tmp_path, reverse):
    from tests.python.support.native_execution_context import artifact_execution_context

    artifact = pops.compile(resolve_interstage_maps(tmp_path, adaptive=True, with_fields=True,
                                                    reverse=reverse))
    instance = pops.bind(artifact, resources={"execution_context": artifact_execution_context(artifact)})
    assert len(_internal_map_image(instance)) == 6
    pops.run(instance, t_end=0.01, max_steps=1)
    base = np.array((2., 3.))[:, None, None]
    # Each constant mode solves (diag(1,2) - Laplacian) phi = current moment.
    # The real published SourceTerm contributes dt*phi before the next map/commit.
    field_update = 1. + 0.01 / np.array((1., 2.))[:, None, None]
    factors = {"population": 10., "integral": 20. * field_update,
               "extended": 12. * field_update}
    for (name, _), actual in _internal_map_image(instance).items():
        np.testing.assert_allclose(actual, np.broadcast_to(base * factors[name], actual.shape),
                                   rtol=2e-11, atol=2e-12)
    assert sorted(instance._executor.mapping_report().values()) == [1, 2]
    first_histories = _history_image(instance)
    assert len(first_histories) == 4
    # depth=1 declares the maximum lag: each physical ring contains two slots, and
    # fill_count counts accepted stores, saturating at that two-slot capacity.
    assert all(row[2] and row[3] == 1 and len(row[4]) == 2 for row in first_histories)
    checkpoint = instance.checkpoint(tmp_path / "field-map-continuation")
    pops.run(instance, t_end=0.02, max_steps=1)
    uninterrupted = _internal_map_image(instance)
    uninterrupted_histories = _history_image(instance)
    assert len(uninterrupted_histories) == 4
    assert all(row[2] and row[3] == 2 and len(row[4]) == 2 for row in uninterrupted_histories)
    instance.restart(checkpoint)
    assert _history_image(instance) == first_histories
    pops.run(instance, t_end=0.02, max_steps=1)
    for key, actual in _internal_map_image(instance).items():
        np.testing.assert_array_equal(actual, uninterrupted[key])
    assert _history_image(instance) == uninterrupted_histories
    assert sorted(instance._executor.mapping_report().values()) == [2, 4]


@pytest.mark.compiler
@pytest.mark.native_loader
def test_native_failed_field_map_attempt_rolls_back_then_reuses_context(tmp_path, record_property):
    from tests.python.support.native_execution_context import artifact_execution_context

    artifact = pops.compile(resolve_interstage_maps(tmp_path, adaptive=True, with_fields=True,
                                                    retry_by_dt=True))
    instance = pops.bind(artifact, resources={"execution_context": artifact_execution_context(artifact)})
    before = _internal_map_image(instance)
    before_histories = _history_image(instance)
    # The first solve has already published into provisional storage when the pullback
    # rejects its non-finite source. Serial native invalid_argument becomes ValueError;
    # MPI reports the same source-packing failure collectively as RuntimeError.
    with pytest.raises((ValueError, RuntimeError), match=(
            "AMR active physical source is non-finite|"
            "AMR transfer source packing failed on a lane rank")) as failed:
        pops.run(instance, t_end=0.01, max_steps=1)
    report = instance._executor._last_step_transaction_report
    assert report.status == "failed" and report.phase == "solve"
    assert report.action == "fail_run" and report.attempts == 1
    assert report.rolled_back_effects == report.staged_effects
    record_property("native_map_failure", str(failed.value))
    record_property("failed_transaction", json.dumps(report.to_data()))
    assert instance.time() == 0. and instance.macro_step() == 0
    for key, actual in _internal_map_image(instance).items():
        np.testing.assert_array_equal(actual, before[key])
    assert _history_image(instance) == before_histories
    assert set(instance._executor.mapping_report().values()) == {0}
    step = 1e-5
    pops.run(instance, t_end=step, max_steps=1)
    assert instance.time() == step and instance.macro_step() == 1
    scale = (200 * step) * 1e307
    field_update = 1. + step / np.array((1., 2.))[:, None, None]
    factors = {"population": 10., "integral": 20. * field_update,
               "extended": 12. * scale * field_update}
    for (name, _), actual in _internal_map_image(instance).items():
        expected = np.array((2., 3.))[:, None, None] * factors[name]
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape),
                                   rtol=2e-10, atol=0.)
    assert sorted(instance._executor.mapping_report().values()) == [1, 2]
    assert len(_history_image(instance)) == 4
    assert all(row[2] and row[3] == 1 and len(row[4]) == 2 for row in _history_image(instance))
