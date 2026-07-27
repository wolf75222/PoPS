"""ADC-702: one source implementation for topology-independent Program operations."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[3]
PROGRAM_DIR = ROOT / "include" / "pops" / "runtime" / "program"
SHARED = PROGRAM_DIR / "program_execution_services.hpp"
UNIFORM = PROGRAM_DIR / "program_context.hpp"
AMR = PROGRAM_DIR / "amr_program_context.hpp"

SHARED_SIGNATURES = (
    "struct FieldStageOverride",
    "struct CouplingStateOverride",
    "void set_stage_time(",
    "void configure_primary_clock(",
    "void declare_clock_relation(",
    "bool schedule_domain_occurs(",
    "bool schedule_is_due(",
    "bool schedule_at_start(",
    "bool schedule_decision(",
    "ClockScheduleState::SubcycleScope subcycle_scope(",
    "void synchronize_sample_and_hold(",
    "int sys_block(",
    "int n_blocks(",
    "Real physical_time(",
    "void record_scalar(",
    "RuntimeParams program_params(",
    "void set_field_logical_timepoint(",
    "void set_field_boundary_parameters(",
    "void set_field_boundary_kernel(",
    "Profiler& profiler(",
    "ProfileScope profile_node(",
    "void profile_record(",
    "void count_kernel(",
    "void count_scratch(",
    "int macro_step(",
    "[[noreturn]] void scheduler_error(",
)


def _read(path):
    return path.read_text(encoding="utf-8")


def test_uniform_and_amr_inherit_the_same_execution_service():
    uniform = _read(UNIFORM)
    amr = _read(AMR)
    assert re.search(
        r"class\s+ProgramContext\s*:\s*public\s+"
        r"ProgramExecutionServices<ProgramContext>",
        uniform,
    )
    assert re.search(
        r"class\s+AmrProgramContext\s*:\s*public\s+"
        r"ProgramExecutionServices<AmrProgramContext>",
        amr,
    )


def test_extracted_operations_have_one_source_definition():
    shared = _read(SHARED)
    context_headers = tuple(PROGRAM_DIR.glob("*program_context.hpp"))
    assert set(context_headers) == {UNIFORM, AMR}
    for signature in SHARED_SIGNATURES:
        assert shared.count(signature) == 1, (
            "%r must have one implementation in program_execution_services.hpp" % signature
        )
        offenders = [
            path.relative_to(ROOT).as_posix()
            for path in context_headers
            if signature in _read(path)
        ]
        assert not offenders, "%r was reimplemented in a topology context: %s" % (
            signature,
            ", ".join(offenders),
        )


def test_clock_state_is_owned_only_by_the_shared_service():
    shared = _read(SHARED)
    declarations = (
        "mutable ClockScheduleState clock_schedule_;",
        "mutable std::string primary_clock_;",
        "mutable amr::Rational stage_time_{0, 1};",
    )
    for declaration in declarations:
        assert shared.count(declaration) == 1
        assert declaration not in _read(UNIFORM)
        assert declaration not in _read(AMR)


def test_contexts_expose_only_topology_provider_hooks_for_the_shared_surface():
    for path, context in ((UNIFORM, "ProgramContext"), (AMR, "AmrProgramContext")):
        source = _read(path)
        assert "friend class ProgramExecutionServices<%s>;" % context in source
        for hook in (
            "program_execution_block_map_",
            "program_execution_block_count_",
            "program_execution_physical_time_",
            "program_execution_record_scalar_",
            "program_execution_params_",
            "program_execution_set_field_timepoint_",
            "program_execution_set_field_parameters_",
            "program_execution_set_field_kernel_",
            "program_execution_profiler_",
            "program_execution_macro_step_",
            "program_execution_active_level_",
        ):
            assert source.count(hook) == 1, (
                "%s must provide exactly one storage/topology hook %s" % (context, hook)
            )


def test_error_schedule_is_shared_not_an_amr_capability_deferral():
    amr = _read(AMR)
    support = _read(ROOT / "python" / "pops" / "runtime" / "amr_program_support.py")
    assert 'deferred_op("scheduler_error"' not in amr
    scheduler_group = support.split('"scheduler": {', 1)[1].split("},\n}", 1)[0]
    assert '"scheduler_error"' not in scheduler_group
