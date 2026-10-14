from __future__ import annotations

import copy

import pytest

from pops.model.ownership import OwnerPath
from pops.provenance import ProvenanceRecord, SourceSpan


def test_source_span_and_provenance_are_exact_immutable_content_addressed_data():
    span = SourceSpan(__file__, 12, 3, 12, 9)
    record = ProvenanceRecord(
        primary=span,
        owner=OwnerPath.consumer("time-step"),
        authoring_api="pops.time.Program.rhs",
        origins=(span,),
        transformation="direct",
    )

    assert SourceSpan.from_data(span.to_data()).to_data() == span.to_data()
    assert ProvenanceRecord.from_data(record.to_data()).to_data() == record.to_data()
    assert record.id.startswith("sha256:") and len(record.id) == 71
    with pytest.raises(AttributeError, match="immutable"):
        record.phase = "lowering"
    with pytest.raises(TypeError):
        record.owner_data["nodes"][0]["name"] = "changed"

    extra = copy.deepcopy(record.to_data())
    extra["legacy"] = True
    with pytest.raises(TypeError, match="exactly"):
        ProvenanceRecord.from_data(extra)
    forged = copy.deepcopy(record.to_data())
    forged["id"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="canonical"):
        ProvenanceRecord.from_data(forged)


def test_derive_preserves_ordered_origins_and_parents():
    owner = OwnerPath.consumer("time-step")
    first_span = SourceSpan(__file__, 31)
    second_span = SourceSpan(__file__, 32)
    first = ProvenanceRecord(
        primary=first_span, owner=owner, authoring_api="manual.first",
    )
    second = ProvenanceRecord(
        primary=second_span, owner=owner, authoring_api="manual.second",
    )

    derived = ProvenanceRecord.derive((first, second), transformation="cse")
    assert derived.primary == first_span
    assert derived.origins == (first_span, second_span)
    assert derived.parents == (first.id, second.id)
    assert derived.phase == "transform"
    assert derived.transformation == "cse"


def test_operator_source_rejects_strings_and_manifest_boundary_requires_provenance():
    from pops.model import Operator, Rate, Signature, StateSpace

    state = StateSpace("U", ("rho",))
    with pytest.raises(TypeError, match="ProvenanceRecord"):
        Operator("rhs", "local_rate", Signature((state,), Rate(state)), source="module")

    operator = Operator("rhs", "local_rate", Signature((state,), Rate(state)))
    with pytest.raises(TypeError, match="ProvenanceRecord"):
        operator.freeze()


def test_operator_manifest_round_trips_documentary_provenance():
    from pops.model import Operator, OperatorHandle, OperatorManifestEntry, Rate, Signature, StateSpace

    owner = OwnerPath.model("transport")
    state = StateSpace("U", ("rho",))
    signature = Signature((state,), Rate(state))
    provenance = ProvenanceRecord(
        primary=SourceSpan(__file__, 70), owner=owner,
        authoring_api="pops.model.Module.operator",
    )
    operator = Operator("rhs", "local_rate", signature, source=provenance)
    handle = OperatorHandle(
        "rhs", kind="local_rate", owner=owner, signature=signature,
        registered_operator_name="rhs",
    )
    entry = OperatorManifestEntry(operator, 0, handle)

    data = entry.to_dict()
    assert data["provenance"] == provenance.to_data()
    assert OperatorManifestEntry.from_dict(data, owner=owner).to_dict() == data


def test_program_serializes_provenance_but_excludes_it_from_ir_hash():
    from pops.identity.semantic import program_semantic_data
    from pops.time import Program

    first = Program("identity")
    first._scalar_binop(1, 2, "add")
    second = Program("identity")
    second._scalar_binop(1, 2, "add")

    assert first._serialize()["nodes"][0]["provenance"] != second._serialize()["nodes"][0]["provenance"]
    assert "provenance" not in first._serialize(include_provenance=False)["nodes"][0]
    assert first._ir_hash() == second._ir_hash()
    assert program_semantic_data(first) == program_semantic_data(second)
    assert first._values[0].provenance.transformation == "direct"


def test_cse_merges_duplicate_lineage_and_records_transform():
    from pops.time import Program

    program = Program("cse")
    program._scalar_binop(1, 2, "add")
    program._scalar_binop(1, 2, "add")
    original_ids = tuple(value.provenance.id for value in program._values)

    optimized = program.eliminate_common_subexpressions()
    assert len(optimized._values) == 1
    provenance = optimized._values[0].provenance
    assert provenance.transformation == "cse"
    assert provenance.parents == original_ids
    assert optimized._ir_hash() != program._ir_hash()


def test_lowering_projection_appends_lower_without_mutating_program():
    from pops.codegen.compile_provenance import lowering_provenance_data
    from pops.time import Program

    program = Program("lower")
    value = program._scalar_binop(1, 2, "add")
    original = value.provenance

    [row] = lowering_provenance_data(program)
    lowered = ProvenanceRecord.from_data(row["provenance"])
    assert row["node_id"] == value.id
    assert lowered.transformation == "lower"
    assert lowered.phase == "lowering"
    assert lowered.parents == (original.id,)
    assert value.provenance is original


@pytest.mark.parametrize('depth,expected_line', [(0, 2), (1, 4), (2, 6), (False, 2), (True, 4)])
def test_source_span_preserves_exact_depth_filename_and_line(tmp_path, depth, expected_line):
    from pops.provenance import source_span

    filename = str(tmp_path / 'nested_capture.py')
    code = ('def inner():\n'
            '    return capture(skip_package=False, depth=depth)\n'
            'def middle():\n'
            '    return inner()\n'
            'def outer():\n'
            '    return middle()\n')
    namespace = {'capture': source_span, 'depth': depth, '__name__': __name__}
    exec(compile(code, filename, 'exec'), namespace)
    span = namespace['outer']()
    assert span.file == filename
    assert span.line == expected_line
    assert span.column == 0 and span.end_line is None and span.end_column is None


@pytest.mark.parametrize('skip_package', [False, True])
def test_source_span_skips_only_exact_package_descendants(skip_package):
    import os
    import pops.provenance as provenance

    root = os.path.dirname(provenance.__file__)
    inside = os.path.join(root, '_source_span_fixture.py')
    # The sibling has the package prefix, but is not inside the package directory.
    outside = os.path.join(root + '-external', 'caller.py')
    inner = {'capture': provenance.source_span, 'skip': skip_package, '__name__': __name__}
    exec(compile('def inner():\n    return capture(skip_package=skip, depth=0)\n',
                 inside, 'exec'), inner)
    outer = {'inner': inner['inner'], '__name__': __name__}
    exec(compile('def outer():\n    return inner()\n', outside, 'exec'), outer)
    span = outer['outer']()
    assert span.file == os.path.abspath(outside if skip_package else inside)
    assert span.line == 2


@pytest.mark.parametrize('filename_kind', ['bytecode', 'pseudo'])
def test_source_span_preserves_inspect_filename_resolution(tmp_path, filename_kind):
    import os
    from pops.provenance import source_span

    code = 'def caller():\n    return capture(skip_package=False, depth=0)\n'
    if filename_kind == 'bytecode':
        source = tmp_path / 'authored.py'
        source.write_text(code, encoding='utf-8')
        filename, expected = str(source.with_suffix('.pyc')), str(source)
    else:
        filename = '<authored-source-span>'
        expected = os.path.abspath(filename)
    namespace = {'capture': source_span, '__name__': __name__}
    exec(compile(code, filename, 'exec'), namespace)
    span = namespace['caller']()
    assert (span.file, span.line) == (expected, 2)


def test_source_span_excessive_depth_returns_unknown():
    from pops.provenance import source_span

    assert source_span(skip_package=False, depth=10**6) == SourceSpan('<unknown>', 0)


@pytest.mark.parametrize('depth', [-2, -3])
def test_source_span_negative_depth_uses_outer_stack_slice(depth):
    import inspect
    import os
    from pops.provenance import source_span

    def caller():
        # The extra source_span frame is at the beginning; these negative indices
        # therefore select the same suspended outer frame in both stacks.
        frames = inspect.stack(context=0)
        try:
            selected = frames[depth + 1]
            expected = (os.path.abspath(selected.filename), selected.lineno)
        finally:
            del selected
            del frames
        actual = source_span(skip_package=False, depth=depth)
        return actual, expected

    actual, expected = caller()
    assert (actual.file, actual.line) == expected


@pytest.mark.parametrize('depth', [-1, -(10**6)])
def test_source_span_slice_start_at_current_frame_stays_inside_source_function(depth):
    import inspect
    import os
    import pops.provenance as provenance

    lines, first = inspect.getsourcelines(provenance.source_span)
    span = provenance.source_span(skip_package=False, depth=depth)
    assert span.file == os.path.abspath(provenance.__file__)
    assert first <= span.line < first + len(lines)


def test_source_span_preserves_slice_index_protocol_and_rejects_float_depth(tmp_path):
    from pops.provenance import source_span

    class SliceIndex:
        def __index__(self):
            return 2

    class Depth:
        def __add__(self, other):
            assert other == 1
            return SliceIndex()

    filename = str(tmp_path / 'index_depth.py')
    namespace = {'capture': source_span, 'depth': Depth(), '__name__': __name__}
    exec(compile('def inner():\n    return capture(skip_package=False, depth=depth)\n'
                 'def outer():\n    return inner()\n', filename, 'exec'), namespace)
    actual = namespace['outer']()
    assert (actual.file, actual.line) == (filename, 4)
    with pytest.raises(TypeError):
        source_span(skip_package=False, depth=0.5)


def test_source_span_does_not_retain_caller_frame_without_cycle_collection():
    import gc
    import weakref
    from pops.provenance import source_span

    class Payload:
        pass

    def caller():
        payload = Payload()
        reference = weakref.ref(payload)
        span = source_span(skip_package=False, depth=0)
        return span, reference

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        span, reference = caller()
        assert isinstance(span, SourceSpan)
        # No gc.collect(): a leaked frame would retain its local payload here.
        assert reference() is None
    finally:
        if was_enabled:
            gc.enable()
