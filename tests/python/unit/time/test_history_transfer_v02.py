"""Closed algebra/identity tests; no native execution claim."""
import copy
import struct
import pytest
import pops
from pops.time import HistoryTransfer
from pops.time._history.transfer import validate_transfer_data
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.runtime._history_sample_identity import validate_identity_bytes
from examples.api_v02.method_history_transfer import quadratic_transfer, build_history_transfer


def test_history_transfer_closes_algebra_and_changes_actual_cpp():
    sources=[]
    for linear,reject in ((False,False),(True,False),(False,True)):
        spec=build_history_transfer(linear_control=linear,reject_parent=reject)
        resolved=pops.resolve(pops.validate(spec.case),layout=spec.layout)
        source=emit_cpp_program(resolved.time,target="amr_system",
            model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
        assert source.count("ctx.register_history_transfer(")==1
        assert "source.times[0]" in source and "source.times[2]" in source
        assert "ctx.history_storage_key(" in source
        assert source.count("result.rows.push_back(")==3
        # Validity executes before interpolation divisions, in a separate scope.
        reject=source.index('HistoryTransfer rejected its temporal validity domain')
        assert reject<source.index('result.rows.push_back(')
        assert "HistorySampleKind::Reconstruction" in source
        assert "history_interval_snapshot_" in source
        sources.append(source)
    assert sources[0]!=sources[1]
    assert quadratic_transfer().to_data()!=quadratic_transfer(linear_control=True).to_data()


def test_history_transfer_validation_does_not_emit_target_code(monkeypatch):
    import pops.codegen.cpp_writer as writer
    def forbidden_emission(*args, **kwargs):
        raise AssertionError("authoring validation invoked target emission")
    monkeypatch.setattr(writer, "_cse_emit", forbidden_emission)
    data = quadratic_transfer().to_data()
    assert validate_transfer_data(data)
    # The common graph validator still rejects invalid structure before lowering.
    data["nodes"][0] = ["add", 0, 0]
    with pytest.raises(ValueError, match="earlier nodes"):
        validate_transfer_data(data)


def test_transfer_refuses_a_same_name_foreign_scalar_symbol():
    from pops._ir.expr import Var
    foreign = Var("history_transfer_input_0", "history_transfer")
    with pytest.raises(TypeError, match="closed input bindings"):
        HistoryTransfer(slots=2, body=lambda samples, time, ratio: (
            time >= 0, ((time, foreign, (1, 0)), (samples[1].time, 1, (0, 1)))))


def test_interval_authority_cannot_escape_into_post_synchronization():
    from pops.intervals import validate_intervals
    program=pops.Program("foreign_phase")
    window=program.interval()
    def after_sync(builder):
        builder._new("scalar","add",(window.begin,window.duration),{},
                     "captured_interval",None)
    program.after_synchronization(after_sync)
    with pytest.raises(NotImplementedError,match="capture.*collective branches"):
        validate_intervals(program)


@pytest.mark.parametrize("mutation",[
    lambda data:data.update(schema_version=True),
    lambda data:data.update(input_count=True),
    lambda data:data.update(nodes=[]),
    lambda data:data.update(nodes=[[]]),
    lambda data:data.update(roots=None),
    lambda data:data.update(slots=2),
    lambda data:data.update(extra="hidden_callback"),
])
def test_transfer_rejects_forged_closed_graph(mutation):
    data = copy.deepcopy(quadratic_transfer().to_data())
    mutation(data)
    with pytest.raises((ValueError,TypeError)):
        validate_transfer_data(data)


def test_transfer_rejects_foreign_program_capture_and_incomplete_rows():
    foreign=pops.Program("foreign")
    h=foreign.requested_dt()
    with pytest.raises((ValueError,TypeError)):
        HistoryTransfer(slots=2,body=lambda s,t,r:(True,((t,h,(1.,0.)),(s[1].time,1.,(0.,1.)))))
    with pytest.raises(ValueError,match="every target"):
        HistoryTransfer(slots=2,body=lambda s,t,r:(True,((t,1.,(1.,0.)),)))


def _identity(kind,digest,magic=b"POPSHID2"):
    header=magic+struct.pack("<Q",1)+b"q"+struct.pack("<qQ",1,1)
    def bits(value):
        return struct.unpack("<Q", struct.pack("<d", value))[0]
    words=(kind,bits(.125),bits(.0625),1)
    if magic == b"POPSHID2":
        words += digest
    return header+struct.pack("<"+"Q"*len(words),*words)


def test_reconstruction_identity_is_distinct_and_legacy_remains_readable():
    raw=_identity(3,(1,2,3,4))
    assert validate_identity_bytes(raw,"q",1,1,initialized=True)==raw
    legacy=_identity(2,(),b"POPSHID1")
    assert validate_identity_bytes(legacy,"q",1,1,initialized=True)==legacy
    with pytest.raises(ValueError,match="missing or foreign"):
        validate_identity_bytes(_identity(3,(0,0,0,0)),"q",1,1)
    with pytest.raises(ValueError,match="missing or foreign"):
        validate_identity_bytes(_identity(2,(1,2,3,4)),"q",1,1)


def test_rhs_store_transfer_survives_canonical_freeze_and_rebuild():
    from examples.api_v02.method_history_transfer import build_flux_history_transfer
    from pops.time._program.rebuild import rebuild_program
    spec=build_flux_history_transfer()
    resolved=pops.resolve(pops.validate(spec.case),layout=spec.layout)
    original=resolved.time
    manifest=original.temporal_manifest()
    row=next(row for row in manifest["histories"] if row["name"]=="retained_rhs")
    assert row["interpolation"]["kind"]=="history_transfer"
    assert row["interpolation"]["slots"]==3
    copied=rebuild_program(original,lambda value:True)
    assert copied.temporal_manifest()==manifest
    copied.freeze()
    assert copied.temporal_manifest()==manifest
    source=emit_cpp_program(copied,target="amr_system",
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert "ctx.register_history_transfer(" in source
    assert "ctx.history_storage_key(" in source
    assert "ctx.copy_with_provenance(x" in source
    assert "pops::PureFieldAlgebra::copy(x" not in source


def _rectangular_transfer():
    # Ordered sources: two immutable entry samples, two candidate samples, endpoint.
    # Different old/new intervals prevent an accidental fixed-grid interpretation.
    def equations(samples, accepted_time, ratio, macro_start, source_duration, accepted_duration, scale):
        target = macro_start + accepted_duration
        alpha = (target - samples[0].time) / (samples[1].time - samples[0].time)
        valid = ((samples[1].time > samples[0].time)
                 & (source_duration > 0) & (accepted_duration > 0)
                 & (accepted_time == target) & (scale > 0) & (ratio > 0))
        return valid, ((target, accepted_duration, (0.,0.,0.,0.,1.)),
                       (macro_start, source_duration, (1-alpha,alpha,0.,0.,0.)))
    return HistoryTransfer(source_slots=5, target_slots=2,
        parameters=("macro_start","source_duration","accepted_duration","scale"),
        body=equations)


def test_rectangular_transfer_has_explicit_source_parameter_and_target_arity():
    from pops.time._history.transfer import transfer_dimensions, require_ring_transfer
    from pops.codegen.program_emit_history_transfer import emit_history_transfer_callback
    policy = _rectangular_transfer()
    data = policy.to_data()
    assert data["schema_version"] == 2
    assert transfer_dimensions(data) == (5, 2, ("macro_start","source_duration","accepted_duration","scale"))
    assert data["input_count"] == 16
    assert len(data["roots"]) == 15  # validity + two (time, interval, five weights) rows
    cpp = "\n".join(emit_history_transfer_callback(data))
    assert "source.times.size() != 5" in cpp
    assert "source.parameters.size() != 4" in cpp
    assert "source.parameters[3]" in cpp
    assert cpp.count("result.rows.push_back(") == 2
    assert cpp.index("rejected its temporal validity domain") < cpp.index("result.rows.push_back(")
    with pytest.raises(ValueError,match="no unbound parameters"):
        require_ring_transfer(data, 2)
    # Returned JSON is detached from the immutable policy, including named bindings.
    data["parameter_names"][0] = "foreign_start"
    assert policy.to_data()["parameter_names"][0] == "macro_start"


def test_legacy_square_transfer_keeps_schema_one_identity():
    from pops.time._history.transfer import transfer_dimensions, require_ring_transfer
    data = quadratic_transfer().to_data()
    assert data["schema_version"] == 1
    assert set(data) == {"kind","schema_version","slots","input_count","roots","nodes"}
    assert transfer_dimensions(data) == (3,3,())
    require_ring_transfer(data,3)


@pytest.mark.parametrize("mutation", [
    lambda data:data.update(source_slots=True),
    lambda data:data.update(source_slots=4),
    lambda data:data.update(slots=3),
    lambda data:data.update(parameter_names=["same"]*4),
    lambda data:data.update(parameter_names=["macro-start","h","a","scale"]),
    lambda data:data.update(parameter_names=["start","h","a"]),
    lambda data:data.update(schema_version=1),
    lambda data:data.update(extra="unbound_source"),
])
def test_rectangular_transfer_rejects_forged_input_binding(mutation):
    data = _rectangular_transfer().to_data()
    mutation(data)
    with pytest.raises((ValueError,TypeError)):
        validate_transfer_data(data)


def test_rectangular_transfer_cannot_hide_parameters_or_capture_state():
    with pytest.raises(ValueError,match="cannot be combined"):
        HistoryTransfer(slots=2,source_slots=3,target_slots=2,body=lambda *args:None)
    with pytest.raises(ValueError,match="unique ASCII"):
        HistoryTransfer(source_slots=2,target_slots=2,parameters=("a","a"),body=lambda *args:None)
    with pytest.raises(ValueError,match="one weight per source"):
        HistoryTransfer(source_slots=3,target_slots=2,
            body=lambda s,t,r:(True,((t,1.,(1.,0.)),(t-1,1.,(0.,1.)))))
    external = pops.Program("foreign_rectangular").requested_dt()
    with pytest.raises((ValueError,TypeError)):
        HistoryTransfer(source_slots=3,target_slots=2,parameters=("h",),
            body=lambda s,t,r,h:(True,((t,external,(1.,0.,0.)),(t-h,h,(0.,1.,0.)))))
