"""Generated Program code volume (B03 family F7, gap C49): what each emission change keeps.

The step body may continue in nested native functions (D9), the continuation schemas and field
callbacks are defined once per translation unit (D1, D3), and the terminal exits of distributed
map outcomes share one routine per distinct statement (D5). None of these changes an operation:
the tests below rebuild the inline form from the shared one and compare it byte for byte.
"""
import re

import pytest

import pops.codegen.program_emit_control as control
import pops.codegen.program_shared_definitions as shared
from pops.codegen.program_codegen import emit_cpp_program


def strip_chunks(lines):
    """Remove the scaffold that _chunk_step_body adds; fail if it is not exactly that scaffold."""
    output, opened, index = [], 0, 0
    width = len(control._STEP_CHUNK_OPEN)
    while index < len(lines):
        if tuple(lines[index:index + width]) == control._STEP_CHUNK_OPEN:
            opened += 1
            index += width
            continue
        output.append(lines[index])
        index += 1
    assert output[len(output) - opened:] == ["}();"] * opened
    return output[:len(output) - opened], opened


def split_points(lines, chunked):
    """Indices of the original body where a nested function starts."""
    points, original, index = [], 0, 0
    width = len(control._STEP_CHUNK_OPEN)
    while index < len(chunked) and original < len(lines):
        if tuple(chunked[index:index + width]) == control._STEP_CHUNK_OPEN:
            points.append(original)
            index += width
            continue
        assert chunked[index] == lines[original]
        index += 1
        original += 1
    return points


def test_a_small_body_stays_one_function():
    lines = ["const int a = 1;", "use(a);"]
    assert control._chunk_step_body(list(lines), [0, 1, 2]) == lines


def test_splits_only_at_recorded_boundaries_outside_every_brace(monkeypatch):
    monkeypatch.setattr(control, "_STEP_CHUNK_CHARACTERS", 8)
    lines = ["const int a = 1;", "if (a && flag) {", "  use(a);", "}", "const int b = a + 1;", "use(b);"]
    # 2 is recorded and past the threshold, but it sits inside the braces of the `if`.
    chunked = control._chunk_step_body(list(lines), [0, 1, 2, 4, 5, 6])
    body, opened = strip_chunks(chunked)
    assert body == lines and opened == 3
    assert split_points(lines, chunked) == [1, 4, 5]


def test_literals_comments_and_alternative_branches_have_no_braces(monkeypatch):
    monkeypatch.setattr(control, "_STEP_CHUNK_CHARACTERS", 1)
    lines = ['report("{");', "// }", "#if defined(__GNUC__)", "[&]() {", "#else", "[&]() {",
             "#endif", "  work();", "}();", "/* { */ done();"]
    depths = control._brace_depths(lines)
    assert depths[0] == 0 and depths[1] == 0 and depths[2] == 0
    assert depths[3] is None and depths[5] is None  # inside a conditional
    assert depths[7] == 1 and depths[8] == 1 and depths[9] == 0 and depths[10] == 0
    # A comment, a preprocessor line or a bare scaffold has no statement characters, so the
    # boundary after the comment is not reached; 7 sits inside the nested function.
    chunked = control._chunk_step_body(list(lines), [0, 1, 2, 7, 9, 10])
    assert split_points(lines, chunked) == [1, 9]


def test_a_body_that_cannot_be_scanned_stays_whole(monkeypatch):
    monkeypatch.setattr(control, "_STEP_CHUNK_CHARACTERS", 1)
    unterminated = ['const char* s = "{;', "use(s);"]
    assert control._brace_depths(unterminated) is None
    assert control._chunk_step_body(list(unterminated), [0, 1, 2]) == unterminated
    unbalanced = ["{", "use();"]
    assert control._chunk_step_body(list(unbalanced), [0, 1, 2]) == unbalanced


def test_the_size_measure_ignores_outlining_scaffold_and_indentation():
    entry = "const auto x = f(a,\n    b);"
    outlined = "\n".join(["// pops-outlined-capture begin 7", "#if defined(__GNUC__) || defined(__clang__)",
                          "[&]() __attribute__((noinline)) {", "#else", "[&]() {", "#endif",
                          "  const auto x = f(a,", "        b);", "}();", "// pops-outlined-capture end 7"])
    assert control._statement_characters(entry) == control._statement_characters(outlined) > 0


def test_schema_accessors_are_keyed_by_construction_text():
    construction = "pops::runtime::program::ContinuationType(pops::runtime::program::ContinuationKind::Product)"
    assert shared.schema_reference(construction) == construction  # outside a translation unit

    @shared.shared_scope
    def unit():
        first = shared.schema_reference(construction)
        again = shared.schema_reference(construction)
        other = shared.schema_reference(construction.replace("Product", "Variant"))
        return first, again, other, shared.shared_definitions()

    first, again, other, definitions = unit()
    assert first == again != other
    assert definitions.count("static const pops::runtime::program::ContinuationType schema = ") == 2
    assert ("schema = %s;" % construction) in definitions


def test_nothing_registers_after_the_definitions_are_rendered():
    @shared.shared_scope
    def unit():
        shared.shared_definitions()
        with pytest.raises(RuntimeError, match="already rendered"):
            shared.schema_reference("pops::runtime::program::ContinuationType()")
        with pytest.raises(RuntimeError, match="already rendered"):
            shared.field_callbacks()
        with pytest.raises(RuntimeError, match="already rendered"):
            shared.terminal_failure_routine(["  throw 1;"])

    unit()


def test_terminal_routines_are_keyed_by_statement_text():
    assert shared.terminal_failure_routine(["  throw 1;"]) is None  # outside a translation unit

    @shared.shared_scope
    def unit():
        names = [shared.terminal_failure_routine([body]) for body in ("  a();", "  a();", "  b();")]
        return names, shared.shared_definitions()

    (first, again, other), definitions = unit()
    assert first == again != other
    assert definitions.count("[[noreturn]] void pops_terminal_failure_") == 2


def _distributed_program(kind):
    from tests.python.integration.runtime.test_api_v02_distributed_kokkos_native import _build
    return _build(kind, 1).case._time


def _inline_source(program, monkeypatch):
    """The same Program with every shared form spelled inline, as the codegen did before F7."""
    with monkeypatch.context() as context:
        context.setattr(shared, "field_callbacks", lambda: (shared._FIELD_COPY, shared._FIELD_PREFLIGHT))
        context.setattr(shared, "schema_reference", lambda construction: construction)
        context.setattr(shared, "terminal_failure_routine", lambda lines: None)
        return emit_cpp_program(program)


_ACCESSOR = re.compile(
    r"static pops::runtime::program::ContinuationType (pops_continuation_schema_[0-9a-f]{24})\(\) \{\n"
    r"  static const pops::runtime::program::ContinuationType schema = (.*?);\n  return schema;\n\}\n", re.S)
_ROUTINE = re.compile(
    r"template <class Context>\n\[\[noreturn\]\] void (pops_terminal_failure_[0-9a-f]{24})\(Context& ctx, "
    r"const pops::runtime::program::distributed::Outcome<pops::kNativeDimension>& pops_outcome\) \{\n(.*?)\n\}\n",
    re.S)


def _expand(source):
    """Spell each shared form inline again and drop the shared definitions."""
    schemas = dict(_ACCESSOR.findall(source))
    routines = dict(_ROUTINE.findall(source))
    markers = [source.index(m) for m in ("// Continuation field callbacks:",
                                         "static pops::runtime::program::ContinuationType pops_continuation_schema_",
                                         "// Terminal failures shared by")
               if m in source]
    if markers:
        first = min(markers)
        last = first
        for pattern in (_ACCESSOR, _ROUTINE):
            for match in pattern.finditer(source, first):
                last = max(last, match.end())
        if "// Continuation field callbacks:" in source:
            last = max(last, source.index("}  // namespace\n", source.index("// Continuation field callbacks:"))
                       + len("}  // namespace\n"))
        if routines:
            last = max(last, source.index("}  // namespace\n", source.index("// Terminal failures shared by"))
                       + len("}  // namespace\n"))
        source = source[:first] + source[last:]
    calls = re.findall(r"^( *)(pops_terminal_failure_[0-9a-f]{24})\(ctx, ([A-Za-z0-9_]+)\);$", source, re.M)
    for indent, name, token in calls:
        statement = "\n".join(indent + line[2:] for line in routines[name].split("\n"))
        statement = re.sub(r"\bpops_outcome\b", token, statement)
        source = source.replace("%s%s(ctx, %s);" % (indent, name, token), statement, 1)
    source = source.replace("pops_continuation_field_copy{ctx}", shared._FIELD_COPY)
    source = source.replace("pops_continuation_field_preflight{ctx}", shared._FIELD_PREFLIGHT)
    for name, construction in schemas.items():
        source = source.replace(name + "()", construction)
    return source, len(calls), len(routines), len(schemas)


@pytest.mark.parametrize("kind", ["nested", "bounds"])
def test_shared_forms_expand_to_the_inline_program(kind, monkeypatch):
    program = _distributed_program(kind)
    after = emit_cpp_program(program)
    before = _inline_source(program, monkeypatch)
    expanded, calls, routines, schemas = _expand(after)
    assert routines == 1 and calls == {"nested": 2, "bounds": 1}[kind] and schemas
    assert "pops_continuation_field_copy" not in before and "pops_terminal_failure_" not in before
    assert expanded == before
