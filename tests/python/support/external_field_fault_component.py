"""Test DSO faults at real FieldSolver@2 dispatch; adapter contract @4."""

from tests.python.integration.native_loader.test_external_field_solver_runtime import _solver_source


def fault_source(manifest):
    source = _solver_source(
        manifest,
        extra_includes="#include <stdexcept>",
        solve_observer_statement="++observed_callbacks;",
    )
    source = source.replace(
        "struct State {",
        "int fault_mode = 0;\nint observed_callbacks = 0;\nint observed_writes = 0;\nstruct State {",
    )
    source = source.replace(
        "if (mask[point] == 1) solution[index] = 7.0;",
        "if (mask[point] == 1) { solution[index] = 7.0; ++observed_writes; }",
    )
    source = source.replace(
        "  report->status = POPS_SOLVE_SOLVED_V2;",
        '  if (fault_mode == 1) throw std::runtime_error("test FieldSolver throw after actual writes");\n  report->status = POPS_SOLVE_SOLVED_V2;',
    )
    source = source.replace("const PopsFieldSolverApiV2 table", "PopsFieldSolverApiV2 table")
    source += """
extern "C" int pops_test_field_fault_arm(int mode) {
  if (mode < 0 || mode > 2) return 1;
  fault_mode = mode;
  table.header.struct_size = mode == 2 ? sizeof(PopsComponentTableHeaderV1) : sizeof(PopsFieldSolverApiV2);
  return 0;
}
extern "C" std::size_t pops_test_field_table_size() { return table.header.struct_size; }
extern "C" int pops_test_field_callback_count() { return observed_callbacks; }
extern "C" int pops_test_field_write_count() { return observed_writes; }
"""
    return source


def json_evidence(value):
    """Lossless JSON framing for CBOR identity bytes in actual component metadata."""
    if type(value) is bytes:
        return {"encoding": "hex", "bytes": value.hex()}
    if type(value) in (dict,):
        return {key: json_evidence(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [json_evidence(item) for item in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError("unsupported evidence value")
