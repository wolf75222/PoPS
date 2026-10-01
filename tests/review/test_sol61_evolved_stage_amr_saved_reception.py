"""SOURCE_ONLY synthetic protocol/math negatives; never native evidence."""
import importlib.util
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("independent_stage_saved",
    Path(__file__).with_name("sol61_evolved_stage_amr_saved_reception.py"))
r = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = r
_spec.loader.exec_module(r)
ROOT = Path(__file__).resolve().parents[2]


def source_only_images(n, width):
    boxes = np.array([[0,0,0,n-1,n-1], [1,n//2,n//2,3*n//2-1,3*n//2-1]], dtype=np.int64)
    modes, owners = ["partitioned"]*2, [np.array([0],dtype=np.int64)]*2
    masks = r.topology(boxes,n,2,modes,owners)
    _, first, q0, load = r.declared_load(width)
    second = first.copy()
    desired = q0+2*r.DT*load
    for _ in range(12):
        a = second[0]
        jacobian = [[1+2*a]] if width == 1 else [[1+2*a,second[1]/5], [second[1]/5,1+2*second[1]+a/5]]
        second -= np.linalg.solve(np.array(jacobian),r.q_of(second[:,None])[:,0]-desired)
    temperatures = {1:first,2:second}
    names = [f"T{i}" for i in range(width)]+(["z"] if width == 2 else [])
    result = {}
    for phase in r.PHASES:
        step = r.STEPS[phase]
        rows = []
        for level in (0,1):
            size = n*2**level
            row = {f"Q{i}":np.full(size*size,q0[i]+step*r.DT*load[i],dtype=float) for i in range(width)}
            row["forcing"] = np.stack([np.full((size,size),v) for v in [*load,1.]]).ravel()
            row["active"] = masks[level].copy()
            if step:
                for values, suffix in ((temperatures[step],""),(first,"-previous")):
                    samples = [*values]+([values[0]/4+values[1]/2] if width == 2 else [])
                    for name,value in zip(names,samples,strict=True):
                        row[name+suffix] = np.full(size*size,value,dtype=float)
                for name in names:
                    row["history_sample_identity_"+name] = np.frombuffer(history(step,level,name),dtype=np.uint8).copy()
            rows.append(row)
        result[phase] = rows
    return result,masks,boxes,modes,owners


@pytest.mark.parametrize("n,width",sorted(r.CASES))
def test_source_only_original_equations_composite_volume_and_exchange(n,width):
    images,masks,*_ = source_only_images(n,width)
    result = r.science(images,masks,n,width)
    assert set(result) == set(r.PHASES)-{"initial"}
    assert max(row["original_Q_linf"] for row in result.values()) < 1e-15
    assert "pops" not in sys.modules


@pytest.mark.parametrize("attack",("staleT","linearQ","wrongz","wrongdt","shiftinitial","staleCapture","falseMask","negativeBranch","history","replay"))
def test_source_only_artificial_equilibrium_cannot_replace_original_math(attack):
    images,masks,*_ = source_only_images(8,2)
    if attack == "staleT":
        images["continuous"][0]["T0"] = images["accepted"][0]["T0"].copy()
    elif attack == "linearQ":
        # Q(t)=t can be exactly balanced but differs from the original quadratic Q.
        for row in images["continuous"]:
            for i in range(2):
                row[f"T{i}"] = row[f"Q{i}"].copy()
    elif attack == "wrongz":
        images["accepted"][0]["z"] += .001
    elif attack == "wrongdt":
        images["accepted"][0]["Q0"] += .001
    elif attack == "shiftinitial":
        for phase in images.values():
            for row in phase:
                row["Q0"] += .01
    elif attack == "staleCapture":
        images["accepted"][0]["forcing"][-1] += .001
    elif attack == "falseMask":
        images["accepted"][0]["active"][:] = True
    elif attack == "negativeBranch":
        images["accepted"][0]["T0"][:] = -.5
    elif attack == "history":
        images["continuous"][0]["T0-previous"] += .001
    elif attack == "replay":
        images["replay"][0]["Q0"][0] = np.nextafter(images["replay"][0]["Q0"][0],np.inf)
    with pytest.raises(ValueError):
        r.science(images,masks,8,2)


@pytest.mark.parametrize("attack",("owner","duplicate","missingbase","unaligned","fullcoarse"))
def test_source_only_owned_finest_topology_refuses_injections(attack):
    _,_,boxes,modes,owners = source_only_images(8,1)
    boxes,owners = boxes.copy(),[row.copy() for row in owners]
    if attack == "owner":
        owners[1][0] = 2
    elif attack == "duplicate":
        boxes = np.concatenate((boxes,boxes[1:]))
    elif attack == "missingbase":
        boxes[0,3] -= 1
    elif attack == "unaligned":
        boxes[1,1] += 1
    else:
        boxes[1] = [1,0,0,15,15]
    with pytest.raises(ValueError):
        r.topology(boxes,8,2,modes,owners)


def test_source_only_transposed_D_is_unidentifiable_for_zero_flux():
    d = np.array([[.012,.002],[-.001,.014]])
    assert not np.array_equal(d,d.T)
    assert np.array_equal(d@np.zeros(2),d.T@np.zeros(2))
    # A reader must not advertise this homogeneous witness as a diffusion-matrix test.
    assert "every actual gradient is zero" in r.science.__doc__


def diagnostic(rank=0,ranks=2):
    names = [(b"",0x8000000000000000),(b"a\x00\xff",0x7ff8000000000061),(b"field.rel_residual",0)]
    return b"POPSDIA1"+struct.pack("<QQQQ",64,rank,ranks,len(names))+b"".join(
        struct.pack("<Q",len(name))+name+struct.pack("<Q",bits) for name,bits in names)


def test_source_only_diagnostic_opaque_names_bits_and_rank_ownership():
    result = r.diagnostic_image(diagnostic(),0,2)
    assert result[b""] == struct.pack("<Q",0x8000000000000000)
    assert result[b"a\x00\xff"] == struct.pack("<Q",0x7ff8000000000061)
    assert r.diagnostic_image(diagnostic(1),1,2) == result


@pytest.mark.parametrize("attack",("rank","ranks","width","count","trailing","truncate"))
def test_source_only_diagnostic_rank_codec_malformed(attack):
    raw = bytearray(diagnostic())
    if attack in ("rank","ranks","width","count"):
        offset,value = {"rank":(16,1),"ranks":(24,3),"width":(8,32),"count":(32,999)}[attack]
        struct.pack_into("<Q",raw,offset,value)
    elif attack == "trailing":
        raw += b"x"
    else:
        raw = raw[:-1]
    with pytest.raises(ValueError):
        r.diagnostic_image(bytes(raw),0,2)


def history(step=2,level=1,name="T0"):
    head = b"POPSHID1"+struct.pack("<Q",len(name))+name.encode()+struct.pack("<qQ",level,2)
    return head+b"".join(struct.pack("<QQQQ",2,int.from_bytes(struct.pack("<d",start),"little"),
        int.from_bytes(struct.pack("<d",r.DT),"little"),1) for start in (0.,(step-1)*r.DT))


def test_source_only_history_native_physical_slot_points_are_distinct():
    r.history_point(history(),"T0",1,2)
    raw = history()
    assert raw[-64:-32] != raw[-32:]
    with pytest.raises(ValueError):
        r.history_point(raw[:-64]+raw[-32:]+raw[-64:-32],"T0",1,2)


@pytest.mark.parametrize("attack",("point","dt","ordinal","level","unknown","trailing"))
def test_source_only_history_sample_mutations(attack):
    raw = bytearray(history())
    offset = len(raw)-32
    if attack in ("point","dt","ordinal","unknown"):
        slot,value = {"point":(1,0),"dt":(2,0),"ordinal":(3,2),"unknown":(0,0)}[attack]
        struct.pack_into("<Q",raw,offset+slot*8,value)
    elif attack == "level":
        struct.pack_into("<q",raw,18,0)
    else:
        raw += b"x"
    with pytest.raises(ValueError):
        r.history_point(bytes(raw),"T0",1,2)


@pytest.mark.parametrize("offsets",([0,4,8],[0,8,4],[1,4,8],[0,4,9],[0,4],[0,-1,8]))
def test_source_only_rank_transport_offsets(offsets):
    state = np.arange(8,dtype=np.uint8)
    if offsets == [0,4,8]:
        assert r.rank_images(state,np.array(offsets,dtype=np.int64),2,"test") == [bytes(range(4)),bytes(range(4,8))]
    else:
        with pytest.raises(ValueError):
            r.rank_images(state,np.array(offsets,dtype=np.int64),2,"test")


@pytest.mark.parametrize("version",(1,2))
def test_source_only_actual_empty_exchange_versions_remain_distinct(version):
    raw = b"POPSEX01"+struct.pack("<Q",0) if version == 1 else b"POPSEX02"+struct.pack("<QQQ",0,0,0)
    r.empty_exchange(raw)
    with pytest.raises(ValueError):
        r.empty_exchange(raw[:-1])
    with pytest.raises(ValueError):
        r.empty_exchange(raw+b"\x00")
    bad = bytearray(raw)
    bad[8] = 1
    with pytest.raises(ValueError):
        r.empty_exchange(bytes(bad))


@pytest.mark.parametrize("a,b",((True,1),(False,0),(1.,1),({"latest_slot":True},{"latest_slot":1})))
def test_source_only_typed_metadata_has_no_python_numeric_alias(a,b):
    assert not r.typed(a,b)


def test_source_only_real_source_AST_admission_without_importing_helper():
    files = [ROOT/"tests/python/support"/name for name in
             ("evolved_stage_amr.py","evolved_stage_mms.py","captured_diffusion_mms.py")]
    values = [path.read_bytes() for path in files]
    r.declared_source(*values)
    with pytest.raises(ValueError):
        r.declared_source(values[0].replace(b"previous=tuple(states[i][0]",b"previous=tuple(states[0][0]"),*values[1:])
    with pytest.raises(ValueError):
        r.declared_source(values[0],values[1].replace(b"[-.001, .014]",b"[.001, .014]"),values[2])
    assert "pops" not in sys.modules


def test_source_only_contract_produces_no_approval_or_native_qualification(tmp_path):
    pins = tmp_path/"pins.json"
    pins.write_text(json.dumps({"SOURCE_ONLY":True}))
    seal = tmp_path/"not-a-root-approval.json"
    seal.write_text(json.dumps({"schema":"SOURCE_ONLY"}))
    with pytest.raises(ValueError,match="ROOT approval scope"):
        r.receive(pins,r.digest(pins.read_bytes()),seal,r.digest(seal.read_bytes()))
    assert r.contract()["native_evidence"] is True
    assert "status" not in r.contract()


def source_only_envelope(arrays):
    def identity(domain):
        return dict(domain=domain,schema_version=1,algorithm="sha256",hexdigest="a"*64)
    manifest = dict(schema_version=1,runtime_kind="amr",semantic_identity=identity("semantic"),
        artifact_identity=identity("artifact"),bind_identity=identity("bind"),run_identity=identity("run"),
        clock=dict(time=float(arrays["t"]).hex(),macro_step=int(arrays["macro_step"])),arrays={})
    for key,a in arrays.items():
        manifest["arrays"][key] = dict(dtype=a.dtype.str,shape=list(a.shape),content_sha256=r.digest(r.wire.cbor(
            dict(protocol="pops.array-evidence.v1",dtype=a.dtype.str,shape=list(a.shape)))+a.tobytes()))
    manifest["restart_identity"] = dict(domain="restart",schema_version=1,algorithm="sha256",hexdigest=r.digest(r.wire.cbor(
        dict(protocol="pops.identity",domain="restart",schema_version=1,payload=manifest))))
    result = dict(arrays)
    result["pops_checkpoint_manifest"] = np.array(json.dumps(manifest))
    result["pops_restart_identity"] = np.array(r.wire.identity_token(manifest["restart_identity"],"restart"))
    return result


def source_only_checkpoint():
    images,masks,boxes,modes,owners = source_only_images(8,2)
    geometry = dict(schema_version=1,dimension=2,shape=[8,8],lower=[0..hex()]*2,upper=[1..hex()]*2,
        periodicity=[True,True],refinement_ratios=[[2,2]],native_layout_identity="pops.native-spatial-layout.v1:sha256:"+"a"*64)
    geometry["identity"] = "pops.checkpoint-spatial-layout.v1:sha256:"+r.digest(r.wire.cbor(
        dict(protocol="pops.identity",domain="checkpoint-spatial-layout",schema_version=1,payload=geometry)))
    temporal = dict(clock=dict(time=r.DT.hex(),macro_step=1),status="accepted",synchronized=True,
        strategy=dict(kind="fixed_dt",dt=dict(kind="binary64",value=r.DT.hex())),
        controller_state=dict(last_accepted_dt=r.DT.hex()),
        **{key:{} for key in ("clock_cursors","schedule_cursors","synchronization_cursors","history_cursors","cache_cursors")})
    images_dia = [diagnostic(i) for i in (0,1)]
    exchange = b"POPSEX02"+struct.pack("<QQQ",0,0,0)
    arrays = {key:np.asarray(value) for key,value in dict(
        pops_amr_checkpoint_version=11,t=r.DT,macro_step=1,n_ranks=2,n_levels=2,configured_n_levels=2,
        abi_key="SOURCE_ONLY",blocks=["Q0","Q1","forcing"],patch_boxes=boxes,
        pops_spatial_contract=json.dumps(geometry),temporal_restart_state=json.dumps(temporal),
        history_names=["T0","T1","z"],amr_accepted_contract=json.dumps(dict(schema_version=7,
            guarantee="bit_identical_accepted_state",program_state="compiled",
            ledger=dict(accepted_entries=0,transaction_depth=0,entries=[]),
            interface_ledger=dict(accepted_entries=0,transaction_depth=0,entries=[]),
            clocks=[["level","0","1","0","1","0.010000"],["level","1","1","0","1","0.010000"],
                    ["logical","SOURCE_ONLY-primary","1"]]))).items()}
    arrays["program_diagnostics_state"] = np.frombuffer(b"".join(images_dia),dtype=np.uint8).copy()
    arrays["program_diagnostics_offsets"] = np.array([0,len(images_dia[0]),sum(map(len,images_dia))],dtype=np.int64)
    arrays["program_exchange_state"] = np.frombuffer(exchange*2,dtype=np.uint8).copy()
    arrays["program_exchange_offsets"] = np.array([0,len(exchange),2*len(exchange)],dtype=np.int64)
    # Opaque bytes below are never presented as real native codec bodies.
    arrays["program_accepted_state"] = np.array([1],dtype=np.uint8)
    arrays["program_accepted_state_source_authority"] = np.array([2],dtype=np.uint8)
    for name in ("T0","T1","z"):
        arrays["history_depth_"+name] = np.array(2)
        arrays["history_ncomp_"+name] = np.array(1)
        arrays["history_levels_"+name] = np.array([0,1],dtype=np.int64)
        arrays["history_stored_slots_"+name] = np.array([0,1],dtype=np.int64)
    for level in (0,1):
        arrays[f"distribution_mode_{level}"] = np.array(modes[level])
        arrays[f"dmap_{level}"] = owners[level].copy()
        arrays[f"auxiliary_checkpoint_{level}"] = np.array([3],dtype=np.uint8)
        for name in ("Q0","Q1","forcing"):
            arrays[f"state_{name}_{level}"] = images["accepted"][level][name].copy()
        for name in ("T0","T1","z"):
            arrays[f"history_init_{name}_level_{level}"] = np.array(True)
            arrays[f"history_fill_count_{name}_level_{level}"] = np.array(1)
            arrays[f"history_slot_dt_{name}_level_{level}"] = np.array([r.DT,r.DT])
            arrays[f"history_sample_identity_{name}_level_{level}"] = np.frombuffer(history(1,level,name),dtype=np.uint8).copy()
            for slot,key in ((0,name+"-previous"),(1,name)):
                arrays[f"history_{name}_level_{level}_{slot}"] = images["accepted"][level][key].copy()
    identities = tuple(f"pops.{domain}.v1:sha256:"+"a"*64 for domain in ("artifact","bind","semantic"))
    return arrays,images,masks,identities


def test_source_only_checkpoint_array_protocol_is_separate_from_opaque_native_codec():
    arrays,images,masks,identities = source_only_checkpoint()
    _,actual,diagnostics = r.checkpoint(source_only_envelope(arrays),"accepted",images,8,2,2,identities,"SOURCE_ONLY")
    assert all(np.array_equal(a,b) for a,b in zip(masks,actual,strict=True))
    assert len(diagnostics) == 2
    # No ROOT approval or receive() exists for this deliberately synthetic image.


@pytest.mark.parametrize("attack",("rawbits","resealedState","resealedOrigin","resealedOwner","resealedRank","resealedClock","resealedHistory","resealedOffsets"))
def test_source_only_checkpoint_raw_and_resealed_semantic_countermodels(attack):
    arrays,images,_masks,identities = source_only_checkpoint()
    if attack == "rawbits":
        sealed = source_only_envelope(arrays)
        sealed["state_Q0_0"][0] += .01
    else:
        if attack == "resealedState":
            arrays["state_Q0_0"][0] += .01
        elif attack == "resealedOrigin":
            geometry = json.loads(str(arrays["pops_spatial_contract"]))
            geometry["lower"][0] = (-1.).hex()
            arrays["pops_spatial_contract"] = np.array(json.dumps(geometry))
        elif attack == "resealedOwner":
            arrays["dmap_1"][0] = 2
        elif attack == "resealedRank":
            arrays["program_diagnostics_state"][16] = 1
        elif attack == "resealedClock":
            arrays["t"] = np.array(.02)
        elif attack == "resealedHistory":
            arrays["history_sample_identity_T0_level_1"][-8] = 2
        else:
            arrays["program_diagnostics_offsets"][1] -= 1
        sealed = source_only_envelope(arrays)
    with pytest.raises(ValueError):
        r.checkpoint(sealed,"accepted",images,8,2,2,identities,"SOURCE_ONLY")


def test_source_only_historical_fixture_slot_inversion_is_real():
    # Historical 4c source labelled ring.at(0) as latest. No assertion on a
    # live fixture that ROOT may already have corrected, and no Git fetch.
    first,second = .16,.17
    before_store = [first,first]
    before_store[0] = second
    after_rotate = list(reversed(before_store))
    assert after_rotate == [first,second]
    assert after_rotate[0] != second


@pytest.mark.parametrize("attack",("staleLevel","fraction","logical","duplicate","pendingLedger"))
def test_source_only_native_report_clock_and_pending_publication(attack):
    arrays,*_ = source_only_checkpoint()
    contract = json.loads(str(arrays["amr_accepted_contract"]))
    if attack == "staleLevel":
        contract["clocks"][1][2] = "0"
    elif attack == "fraction":
        contract["clocks"][1][3:5] = ["1","2"]
    elif attack == "logical":
        contract["clocks"][-1][-1] = "0"
    elif attack == "duplicate":
        contract["clocks"].append(contract["clocks"][-1])
    else:
        contract["ledger"]["transaction_depth"] = 1
    with pytest.raises(ValueError):
        r.accepted_contract(contract,1)


def source_only_ir():
    attrs = dict(newton_controls={key:value if type(value) is int else dict(kind="binary64",value=value.hex()) for key,value in r.CONTROLS.items()},
        finite_difference_step=dict(kind="binary64",value=(1e-6).hex()),ncomp=3,
        right_preconditioner="pops.amr.full-residual-basis-lu@1",
        right_preconditioner_resources=dict(version=1,max_dense_bytes=dict(uint64_hex="0000000010000000"),
            scope="per_rank_dense_arrays_active_map_and_numeric_towers"),source_contract=dict(temporal_tau={},evolved_stage={}))
    return dict(version=16,nodes=[dict(op="solve_spatial_field",attrs=attrs),
        dict(op="field_evolved_state",attrs={}),dict(op="store_history",attrs={"contract":"pops.program.global-field-history-storage@1"})])


def ir_cpp(ir):
    digest = r.digest(json.dumps(ir,sort_keys=True,separators=(",",":")).encode())
    return digest, f'extern "C" const char* pops_program_hash() {{ return "{digest}"; }}\n// store_global_field_history('


def test_source_only_saved_ir_cpp_hash_link_has_no_dso_crypto_claim():
    ir = source_only_ir()
    digest,cpp = ir_cpp(ir)
    assert r.program_image(ir,cpp,16,digest,2) == digest
    with pytest.raises(ValueError,match="CPP Program hash"):
        r.program_image(ir,cpp.replace(digest,"a"*64),16,digest,2)


@pytest.mark.parametrize("field",("newton_controls","finite_difference_step","ncomp","right_preconditioner","right_preconditioner_resources","source_contract"))
def test_source_only_rehashed_ir_controls_and_realization_tamper(field):
    ir = source_only_ir()
    attrs = ir["nodes"][0]["attrs"]
    if field == "newton_controls":
        attrs[field]["restart"] = 240
    elif field == "finite_difference_step":
        attrs[field]["value"] = (1e-7).hex()
    elif field == "ncomp":
        attrs[field] = 2
    elif field == "right_preconditioner":
        attrs[field] = "SpatialBasisJacobi@1"
    elif field == "right_preconditioner_resources":
        attrs[field]["max_dense_bytes"]["uint64_hex"] = "0000000000000001"
    else:
        attrs[field]["temporal_tau"] = None
    digest,cpp = ir_cpp(ir)  # Explicit SOURCE_ONLY countermodel, not a ROOT/native seal.
    with pytest.raises(ValueError):
        r.program_image(ir,cpp,16,digest,2)
