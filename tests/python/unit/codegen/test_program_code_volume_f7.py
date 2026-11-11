"""Generated Program code volume (B03 family F7, gap C49): what each emission change keeps.

The step body may continue in nested native functions (D9), the continuation schemas and field
callbacks are defined once per translation unit (D1, D3), the terminal exits of distributed map
outcomes share one routine per distinct statement (D5), and the captured Arrays of device maps
share one staging routine per distinct staging text (D10). None of these changes an operation:
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


def test_a_split_follows_only_a_line_that_ends_a_statement(monkeypatch):
    monkeypatch.setattr(control, "_STEP_CHUNK_CHARACTERS", 1)
    outlined = "\n".join(["// pops-outlined-capture begin 7", "#if defined(__GNUC__) || defined(__clang__)",
                          "[&]() __attribute__((noinline)) {", "#else", "[&]() {", "#endif",
                          "  const auto x = g(a);", "  use(x);", "}();", "// pops-outlined-capture end 7"])
    lines = ["const int a = f();", "if (a)", "  use(a);", outlined, "done(a);"]
    # 2 sits after `if (a)`: a split there would make the `if` govern the rest of the step. 4
    # follows an outlined capture, whose closing comment is passed over to its `}();`.
    chunked = control._chunk_step_body(list(lines), [1, 2, 3, 4, 5])
    body, opened = strip_chunks(chunked)
    assert body == lines and opened == 3
    assert split_points(lines, chunked) == [1, 3, 4]
    assert not control._ends_statement(lines, 2) and not control._ends_statement(lines, 0)
    assert control._ends_statement(["work(); // why", "// note", ""], 3) is False
    # A preprocessor line does not qualify: after `#endif`, the `if` would govern the rest of the
    # step whenever X is undefined.
    guarded = ["const int a = f();", "if (a)", "#if X", "  use(a);", "#endif", "done(a);"]
    assert split_points(guarded, control._chunk_step_body(list(guarded), [1, 2, 5])) == [1]
    assert control._ends_statement(["use(a);", "#endif", "// note"], 3) is False


def test_raw_string_literals_have_no_braces():
    lines = ['const char* s = R"pops(" { ")pops";', "use(s);", 'const char* t = u8R"(', "#if 0", "{",
             ')";', 'const wchar_t* u = LR"({)" L"}";', "use(t, u);"]
    assert control._brace_depths(lines) == [0, 0, 0, None, None, None, 0, 0, 0]
    # An identifier that ends with R before a quote is no raw literal prefix.
    assert control._brace_depths(['use(ABR"{", 1);', "done();"]) == [0, 0, 0]
    # The scan resumes right after the closing quote, where a brace may follow.
    assert control._brace_depths(['const S s{R"(x)"};', "use(s);"]) == [0, 0, 0]
    # A literal whose delimiter is malformed or never ends leaves the body unscannable.
    assert control._brace_depths(['const char* s = R"' + "d" * 17 + '(x)' + "d" * 17 + '";']) is None
    assert control._brace_depths(['const char* s = R"x(', "{"])[-1] is None


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
        with pytest.raises(RuntimeError, match="already rendered"):
            shared.staging_routine(["int pops_a"], ["  use(pops_a);"])

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


def test_staging_routines_are_keyed_by_signature_and_statements():
    assert shared.staging_routine(["int pops_a"], ["  use(pops_a);"]) is None  # outside a unit

    @shared.shared_scope
    def unit():
        names = [shared.staging_routine(parameters, ["  use(pops_a);"])
                 for parameters in (["int pops_a"], ["int pops_a"], ["long pops_a"])]
        return names, shared.shared_definitions()

    (first, again, other), definitions = unit()
    assert first == again != other
    assert definitions.count("__attribute__((noinline))\n#endif\nvoid pops_stage_array_") == 2


def _distributed_program(kind):
    from tests.python.integration.runtime.test_api_v02_distributed_kokkos_native import _build
    return _build(kind, 1).case._time


def _inline_source(program, monkeypatch):
    """The same Program with every shared form spelled inline, as the codegen did before F7."""
    with monkeypatch.context() as context:
        context.setattr(shared, "field_callbacks", lambda: (shared._FIELD_COPY, shared._FIELD_PREFLIGHT))
        context.setattr(shared, "schema_reference", lambda construction: construction)
        context.setattr(shared, "terminal_failure_routine", lambda lines: None)
        context.setattr(shared, "staging_routine", lambda parameters, lines: None)
        return emit_cpp_program(program)


_ACCESSOR = re.compile(
    r"static pops::runtime::program::ContinuationType (pops_continuation_schema_[0-9a-f]{24})\(\) \{\n"
    r"  static const pops::runtime::program::ContinuationType schema = (.*?);\n  return schema;\n\}\n", re.S)
_ROUTINE = re.compile(
    r"template <class Context>\n\[\[noreturn\]\] void (pops_terminal_failure_[0-9a-f]{24})\(Context& ctx, "
    r"const pops::runtime::program::distributed::Outcome<pops::kNativeDimension>& pops_outcome\) \{\n(.*?)\n\}\n",
    re.S)


_STAGING = re.compile(
    r"#if defined\(__GNUC__\) \|\| defined\(__clang__\)\n__attribute__\(\(noinline\)\)\n#endif\n"
    r"void (pops_stage_array_[0-9a-f]{24})\((.*?)\) \{\n(.*?)\n\}\n", re.S)
_STAGING_CALL = re.compile(r"^( *)(pops_stage_array_[0-9a-f]{24})\((.*)\);$", re.M)
_NS = "pops::runtime::program::finite_kokkos::"


def _arguments(text):
    """Split call arguments at top-level commas (the generated arguments hold no string comma)."""
    parts, depth, start = [], 0, 0
    for index, character in enumerate(text.replace("->", "-.")):
        depth += character in "(<[{"
        depth -= character in ")>]}"
        if character == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    return parts + [text[start:]]


def _expand_staging(source, routines):
    """Spell each staging call inline again: declarations where the inline form writes first."""
    count = 0
    while True:
        match = _STAGING_CALL.search(source)
        if match is None:
            return source, count
        indent, name, text = match.groups()
        signature, body = routines[name]
        parameters = [parameter.split()[-1] for parameter in signature.split(", ")]
        mapping = dict(zip(parameters, _arguments(text), strict=True))
        columns = [mapping[p] for p in parameters if re.fullmatch(r"pops_column_\d+", p)]
        descriptor = [mapping["pops_" + member] for member in ("identity", "size", "begin", "end")]
        mapping["pops_index"] = re.sub(r"_\d+$", "", columns[0]) + "_i"
        # The declarations the call site placed before the call, removed with it.
        declared = ["std::int64_t %s;" % ", ".join("%s = 0" % d for d in descriptor)]
        types = {}
        start = match.start()
        lines = source[:start].split("\n")[:-1]
        for column in reversed(columns):
            host = lines.pop()
            view = lines.pop()
            assert host == indent + "decltype(%s)::host_mirror_type %s_host;" % (column, column)
            prefix = indent + _NS + "View<"
            assert view.startswith(prefix) and view.endswith("> %s;" % column)
            types[column] = view[len(prefix):-len("> %s;" % column)]
        assert lines.pop() == indent + declared[0]
        def rename(match, table=mapping):
            return table.get(match.group(0), match.group(0))

        expanded = []
        for line in body.split("\n"):
            line = indent + re.sub(r"\bpops_[A-Za-z0-9_]+\b", rename, line[2:])
            if any(line.startswith(indent + d + " = ") for d in descriptor):
                line = indent + "const std::int64_t " + line[len(indent):]
            allocated = [c for c in columns if (" %s = %sallocate<" % (c, _NS)) in line]
            if allocated:
                expanded.append(indent + "%sView<%s> %s;" % (_NS, types[allocated[0]], allocated[0]))
                expanded.append(line)
                line = indent + "decltype(%s)::host_mirror_type %s_host;" % (allocated[0], allocated[0])
            expanded.append(line)
        head = "\n".join(lines) + "\n"
        source = head + "\n".join(expanded) + source[match.end():]
        count += 1


def _expand(source):
    """Spell each shared form inline again and drop the shared definitions."""
    schemas = dict(_ACCESSOR.findall(source))
    routines = dict(_ROUTINE.findall(source))
    staging = {name: (signature, body) for name, signature, body in _STAGING.findall(source)}
    markers = [source.index(m) for m in ("// Continuation field callbacks:",
                                         "static pops::runtime::program::ContinuationType pops_continuation_schema_",
                                         "// Terminal failures shared by",
                                         "// Captured Array staging shared by")
               if m in source]
    if markers:
        first = min(markers)
        last = first
        for pattern in (_ACCESSOR, _ROUTINE, _STAGING):
            for match in pattern.finditer(source, first):
                last = max(last, match.end())
        if "// Continuation field callbacks:" in source:
            last = max(last, source.index("}  // namespace\n", source.index("// Continuation field callbacks:"))
                       + len("}  // namespace\n"))
        if routines:
            last = max(last, source.index("}  // namespace\n", source.index("// Terminal failures shared by"))
                       + len("}  // namespace\n"))
        if staging:
            last = max(last, source.index("}  // namespace\n", source.index("// Captured Array staging shared by"))
                       + len("}  // namespace\n"))
        source = source[:first] + source[last:]
    source, staged = _expand_staging(source, staging)
    calls = re.findall(r"^( *)(pops_terminal_failure_[0-9a-f]{24})\(ctx, ([A-Za-z0-9_]+)\);$", source, re.M)
    for indent, name, token in calls:
        statement = "\n".join(indent + line[2:] for line in routines[name].split("\n"))
        statement = re.sub(r"\bpops_outcome\b", token, statement)
        source = source.replace("%s%s(ctx, %s);" % (indent, name, token), statement, 1)
    source = source.replace("pops_continuation_field_copy{ctx}", shared._FIELD_COPY)
    source = source.replace("pops_continuation_field_preflight{ctx}", shared._FIELD_PREFLIGHT)
    for name, construction in schemas.items():
        source = source.replace(name + "()", construction)
    return source, len(calls), len(routines), len(schemas), staged, len(staging)


@pytest.mark.parametrize("kind", ["nested", "bounds", "import"])
def test_shared_forms_expand_to_the_inline_program(kind, monkeypatch):
    program = _distributed_program(kind)
    after = emit_cpp_program(program)
    before = _inline_source(program, monkeypatch)
    expanded, calls, routines, schemas, staged, staging = _expand(after)
    assert routines == 1 and calls == {"nested": 2, "bounds": 1, "import": 2}[kind] and schemas
    assert staged == staging == 1
    for shared_form in ("pops_continuation_field_copy", "pops_terminal_failure_", "pops_stage_array_"):
        assert shared_form not in before
    assert expanded == before
