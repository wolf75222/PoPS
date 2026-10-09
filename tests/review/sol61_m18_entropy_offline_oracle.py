"""Independent M18 saved-state reception: stdlib/NumPy, no PoPS runtime import.

Positive receipts require external owner pins. Pure mathematical predicates and
protocol tests do not constitute native execution or saved-state qualification.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "m18_independent_checkpoint_protocol", Path(__file__).with_name("sol61_integral_feedback_offline_oracle.py"))
protocol = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = protocol
_spec.loader.exec_module(protocol)
require, digest, strict_json, cbor = protocol.require, protocol.digest, protocol.strict_json, protocol.cbor
_equation_spec = importlib.util.spec_from_file_location(
    "m18_original_equation_guard", Path(__file__).with_name("sol61_m18_original_equation_guard.py"))
equation_guard = importlib.util.module_from_spec(_equation_spec)
_equation_spec.loader.exec_module(equation_guard)
DT, RESIDUAL_TOL = .01, 2.e-11
NODES = np.array([-1., -.5, 0., .5, 1.])
WEIGHTS = np.array([.1, .2, .4, .2, .1])
BASIS = np.vstack((np.ones(5), NODES, NODES**2))
PHASES = ("initial", "accepted", "continuous", "restored", "replayed", "outside_before",
          "outside_rejected_0", "outside_rejected_1", "safe_rebind_initial", "safe_rebind_accepted")
TEST_NAME = "test_twenty_interior_targets_and_outside_cone_refusal"
CONSTANTS = dict(nodes=NODES.tolist(), weights=WEIGHTS.tolist(), basis=BASIS.tolist(), cells=[5,4], dt=DT,
                 unknowns=3, max_iterations=12, original_residual_tolerance=RESIDUAL_TOL,
                 max_backtracks=16, minimum_step=2.**-16, safeguard="backtracking")


def authored_contract(source):
    tree=ast.parse(source)
    functions={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
    make=functions["make_case"]
    nodes=list(ast.walk(make))
    calls=[node for node in nodes if isinstance(node,ast.Call)]
    commits=[node for node in calls if isinstance(node.func,ast.Attribute) and node.func.attr=="commit"]
    expected=ast.parse("program.commit(dual_state.next, solved[dual_block])").body[0].value
    require(len(commits)==1 and ast.dump(commits[0])==ast.dump(expected),"M18 must publish only the dual block")
    residual=next(node for node in nodes if isinstance(node,ast.FunctionDef) and node.name=="residual")
    expected=ast.parse('return {"dual": QUADRATURE.residual(unknowns["dual"], target)}').body[0]
    require(len(residual.body)==1 and ast.dump(residual.body[0])==ast.dump(expected),"original three-moment residual altered")
    local=[node for node in calls if isinstance(node.func,ast.Name) and node.func.id=="LocalResidual"]
    require(len(local)==1 and len(local[0].args)==2
            and ast.dump(local[0].args[1])==ast.dump(ast.parse('{"dual": seed}',mode="eval").body),
            "M18 must solve three dual unknowns, not a six-unknown target product")
    require(len(local[0].keywords)==1 and local[0].keywords[0].arg=="captures"
            and ast.dump(local[0].keywords[0].value)==ast.dump(ast.parse('{"target": data_state.n}',mode="eval").body),
            "exact readonly target capture missing")
    newton=[node for node in calls if isinstance(node.func,ast.Name) and node.func.id=="LocalNewton"]
    expected=ast.parse('LocalNewton(tolerance=2.e-11,max_iterations=12,safeguard="backtracking",max_backtracks=16,minimum_step=2.**-16)',mode="eval").body
    require(len(newton)==1 and ast.dump(newton[0])==ast.dump(expected),"Newton budget/original tolerance altered")
    steps=[node for node in calls if isinstance(node.func,ast.Name) and node.func.id=="FixedDt"]
    require(len(steps)==1 and ast.dump(steps[0])==ast.dump(ast.parse("FixedDt(.01)",mode="eval").body),
            "issued duration authority changed")


def compiled_contract(ir,expected_hash):
    require(type(ir["version"]) is int and ir["version"]==5 and len(ir["commits"])==1,
            "compiled Program version/publication mismatch")
    require(digest(json.dumps(ir,sort_keys=True,separators=(",",":"),allow_nan=False).encode())==expected_hash,
            "compiled Program IR hash differs")
    nodes=[node for node in ir["nodes"] if node["op"]=="solve_coupled_implicit"]
    require(len(nodes)==1 and len(nodes[0]["inputs"])==2,"native compiled solve/capture count differs")
    attrs=nodes[0]["attrs"]
    expected=dict(method="newton",problem_kind="local_residual_product",product_version=1,
                  unknown_names=["dual"],capture_names=["target"],product_widths=[3,3],product_reads=[0,1],
                  output_count=1,tol=dict(kind="binary64",value=RESIDUAL_TOL.hex()),relative_tol=0.,step_tol=0.,
                  max_iter=12,safeguard="backtracking",max_backtracks=16,
                  minimum_step=dict(kind="binary64",value=(2.**-16).hex()))
    require(cbor({key:attrs.get(key) for key in expected if key not in {"relative_tol","step_tol"}})
            ==cbor({key:value for key,value in expected.items() if key not in {"relative_tol","step_tol"}})
            and type(attrs["relative_tol"]) is float and attrs["relative_tol"]==0.
            and type(attrs["step_tol"]) is float and attrs["step_tol"]==0.
            and len(attrs["expressions"])==3,"compiled three-unknown residual/Newton contract altered")
    commit=ir["commits"][0]
    require(commit["block"]["local_id"]=="dual" and commit["state"]["local_id"]=="multipliers",
            "compiled readonly target was committed")
    equation_guard.verify(ir)


def cone(target):
    """Convex polygon through (v,v²); independent of the author's LP/Newton."""
    target = np.asarray(target)
    require(target.shape == (3,) and np.isfinite(target).all(), "invalid cone target")
    mass, first, second = map(float, target)
    if mass < 0 or (mass == 0 and (first != 0 or second != 0)):
        return "outside"
    if mass == 0:
        return "boundary"
    mean, energy = first / mass, second / mass
    if mean < -1.-RESIDUAL_TOL or mean > 1.+RESIDUAL_TOL:
        return "outside"
    index = min(3, max(0, int(np.searchsorted(NODES, mean, side="right")) - 1))
    left, right = NODES[index:index+2]
    lower = (left+right)*mean-left*right
    distances = (mean+1., 1.-mean, energy-lower, 1.-energy)
    if min(distances) < -RESIDUAL_TOL:
        return "outside"
    return "boundary" if min(distances) <= RESIDUAL_TOL else "interior"


def entropy(population):
    population = np.asarray(population)
    require(population.shape == (5,) and np.isfinite(population).all() and np.all(population > 0),
            "population not finite and strictly positive")
    value=math.fsum(float(p) * (math.log(float(p))-math.log(float(w))-1.)
                    for p,w in zip(population, WEIGHTS, strict=True))
    require(math.isfinite(value),"population entropy overflow")
    return value


def check_solved(dual, target):
    require(dual.dtype == target.dtype == np.dtype("float64") and dual.shape == target.shape == (3,4,5)
            and np.isfinite(dual).all() and np.isfinite(target).all(), "dual/target shape/type/finitude mismatch")
    residual_max, population_min, entropy_gap_min, tangent_defect_max = 0., math.inf, math.inf, 0.
    cells=[]
    # Exact integer null vectors for this retained basis, without the author's SVD.
    tangent = 2.**-10*np.array([-1.,4.,-6.,4.,-1.]) + 2.**-11*np.array([-1.,2.,0.,-2.,1.])
    require(np.array_equal(BASIS@tangent, np.zeros(3)), "independent null vector is not exact")
    for y,x in np.ndindex(4,5):
        require(cone(target[:,y,x]) == "interior", "accepted target not moderate interior")
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            population = WEIGHTS*np.exp(BASIS.T@dual[:,y,x])
        require(np.isfinite(population).all() and np.all(population > 0), "nonfinite/nonpositive accepted population")
        moments = np.array([math.fsum(float(a*p) for a,p in zip(row,population,strict=True)) for row in BASIS])
        residual = float(np.max(np.abs(moments-target[:,y,x])))
        require(residual <= RESIDUAL_TOL, "original moment residual exceeds 2e-11")
        varied = population+tangent
        require(np.min(varied) > 0 and np.max(np.abs(BASIS@varied-moments)) <= 2.e-15,
                "entropy perturbation changed moments or positivity")
        gap = entropy(varied)-entropy(population)
        stationarity = abs(float(np.log(population/WEIGHTS)@tangent))
        require(gap > 1.e-7 and stationarity < 2.e-14, "entropy minimum/stationarity witness failed")
        residual_max=max(residual_max,residual)
        population_min=min(population_min,float(population.min()))
        entropy_gap_min=min(entropy_gap_min,gap)
        tangent_defect_max=max(tangent_defect_max,stationarity)
        cells.append(dict(index_yx=[y,x],populations=population.tolist(),moments=moments.tolist(),
                          original_target=target[:,y,x].tolist(),entropy=entropy(population),
                          cone="interior",original_residual_linf=residual,entropy_gap=gap))
    indices=np.arange(20,dtype=float).reshape(4,5)
    expected=np.stack((-.18+.018*indices,.14*np.sin(.23*indices),-.12+.01*indices))
    multiplier_error=float(np.max(np.abs(dual-expected)))
    require(multiplier_error <= 1.e-8, "moderate multiplier mismatch")
    return dict(original_moment_residual=residual_max, minimum_population=population_min,
                minimum_entropy_gap=entropy_gap_min, entropy_tangent_defect=tangent_defect_max,
                multiplier_linf_error=multiplier_error,cells=20,recomputed_cells=cells)


def pinned(base, row):
    require(type(row) is dict and set(row)=={"path","sha256"} and type(row["path"]) is str
            and re.fullmatch("[0-9a-f]{64}",row["sha256"]) is not None, "invalid external file pin")
    path=Path(row["path"])
    if not path.is_absolute():
        require(".." not in path.parts and str(path)==row["path"] and "\\" not in row["path"], "aliased pin path")
        path=base/path
    require(not path.is_symlink() and path.is_file() and path.stat().st_size<=protocol.MAX_FILE_BYTES,
            "missing/nonregular/oversized pinned file")
    raw=path.read_bytes()
    require(digest(raw)==row["sha256"],"external file SHA mismatch: "+str(path))
    return path,raw


def typed_array(array):
    array=np.ascontiguousarray(array) if array.shape else array
    header=cbor(dict(protocol="pops.array-evidence.v1",dtype=array.dtype.str,shape=list(array.shape)))
    return dict(dtype=array.dtype.str,shape=list(array.shape),content_sha256=digest(header+array.tobytes()))


def geometry_evidence(array):
    array=np.ascontiguousarray(array)
    header=array.dtype.str.encode()+b"\0"+",".join(map(str,array.shape)).encode()+b"\0"
    return dict(dtype=array.dtype.str,shape=list(array.shape),content_sha256=digest(header+array.tobytes()))


def ownership(rows, ranks):
    require(type(rows) is list and len(rows)==ranks, "rank ownership inventory differs")
    all_counts=[]
    for block in ("dual","target"):
        counts=[]
        for rank in rows:
            require(type(rank) is dict and set(rank)=={"dual","target"}, "ownership block mismatch")
            count=np.zeros((4,5),dtype=int)
            for box in rank[block]:
                require(type(box) is list and len(box)==2 and all(type(p) is list and len(p)==2 for p in box)
                        and all(type(v) is int for p in box for v in p), "invalid rank-owned box")
                (x0,y0),(x1,y1)=box
                require(0<=x0<x1<=5 and 0<=y0<y1<=4, "rank-owned box escapes domain")
                count[y0:y1,x0:x1]+=1
            require(np.all(count<=1), "rank-local boxes overlap")
            counts.append(count)
        stacked=np.stack(counts)
        distributed=np.all(stacked.sum(axis=0)==1)
        replicated=np.all(stacked==1)
        require(distributed or replicated,"neither partitioned nor fully replicated support")
        all_counts.append(stacked)
    require(np.array_equal(*all_counts), "dual/readonly target ownership differs")
    return "distributed" if np.all(all_counts[0].sum(axis=0)==1) else "replicated"


EMPTY_EXCHANGE_READER = "sol61.m18-empty-exchange-reader@2"
EMPTY_EXCHANGE_IMAGES = (b"POPSEX01" + bytes(8), b"POPSEX02" + bytes(24))


def empty_exchange_images(raw, offsets, ranks):
    """Admit only exact zero-count native01/02 images at every rank boundary."""
    require(type(ranks) is int and ranks > 0
            and raw.dtype == np.dtype("uint8") and raw.ndim == 1
            and offsets.dtype == np.dtype("int64") and offsets.ndim == 1
            and len(offsets) == ranks + 1, "rank exchange offsets mismatch")
    bounds = list(map(int, offsets))
    require(bounds[0] == 0 and bounds[-1] == len(raw)
            and all(0 <= a < b <= len(raw) for a, b in zip(bounds[:-1], bounds[1:], strict=True)),
            "M18 has a spatial exchange or invalid empty ledger")
    images = [raw[a:b].tobytes() for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
    require(all(image in EMPTY_EXCHANGE_IMAGES for image in images),
            "M18 contains fabricated physical exchanges")
    return images


def snapshot(base,pins,phase):
    rows=pins["phases"][phase]
    require(set(rows)=={"state","receipt","checkpoint"},"incomplete phase file pins")
    checked={name:pinned(base,row) for name,row in rows.items()}
    receipt=strict_json(checked["receipt"][1])
    require(receipt["phase"]==phase and receipt["artifact_identity"]==pins["artifact_identity"]
            and type(receipt["dimension"]) is int and receipt["dimension"]==2
            and type(receipt["ranks"]) is int and receipt["ranks"]==pins["ranks"]
            and receipt["storage"]=="component,y,x", "phase/artifact/rank/storage mismatch")
    time,step=receipt["time"],receipt["macro_step"]
    require(type(time) is float and math.isfinite(time) and time>=0 and type(step) is int and step>=0
            and time.hex()==receipt["time_hex"], "noncanonical phase clock")
    state=protocol.archive(checked["state"][1])
    require(set(state)=={"dual","target","cell_volumes","coverage","valid_cells"}, "state file inventory differs")
    for name in ("dual","target"):
        require(state[name].dtype==np.dtype("float64") and state[name].shape==(3,4,5)
                and np.isfinite(state[name]).all(), "native state shape/type/finitude mismatch")
    geo=receipt["geometry"]
    require(geo["layout_kind"]=="uniform" and type(geo["level"]) is int and geo["level"]==0
            and geo["cell_shape"]==[4,5] and geo["origin"]==[0..hex(),0..hex()]
            and geo["spacing"]==[.2.hex(),.25.hex()]
            and geo["coordinate_system"]=="pops://coordinates/cartesian-2d@1"
            and geo["cell_measure"]=="pops://cell-measures/cartesian-area@1"
            and geo["axis_names"]==["x","y"], "static physical geometry mismatch")
    require(state["cell_volumes"].dtype==np.dtype("float64") and state["cell_volumes"].shape==(4,5)
            and state["valid_cells"].dtype==np.dtype("bool") and state["coverage"].dtype==np.dtype("bool")
            and np.all(state["cell_volumes"]==.2*.25) and np.all(state["valid_cells"])
            and not np.any(state["coverage"]),"physical measures/coverage mismatch")
    represented=np.zeros((4,5),dtype=int)
    for box in geo["boxes"]:
        require(type(box) is list and len(box)==4 and all(type(value) is int for value in box),
                "invalid physical geometry box")
        y0,x0,y1,x1=box
        require(0<=y0<y1<=4 and 0<=x0<x1<=5,"physical geometry box escapes support")
        represented[y0:y1,x0:x1]+=1
    require(np.all(represented==1),"physical geometry boxes do not represent each cell exactly once")
    for name in ("cell_volumes","coverage","valid_cells"):
        require(state[name].shape==(4,5) and geo[name]==geometry_evidence(state[name]),"geometry typed array mismatch")
    mode=ownership(receipt["local_boxes_by_rank"],pins["ranks"])
    checkpoint=protocol.archive(checked["checkpoint"][1])
    require(receipt["checkpoint_sha256"]==digest(checked["checkpoint"][1])
            and receipt["checkpoint"]==checked["checkpoint"][0].name,"checkpoint file pin differs")
    require(str(checkpoint["abi_key"].item())==pins["abi_key"] and protocol.scalar(checkpoint["t"],"real").hex()==time.hex()
            and protocol.scalar(checkpoint["macro_step"],"int")==step,"checkpoint native/clock mismatch")
    require(list(checkpoint["blocks"])==["dual","target"],"checkpoint block order mismatch")
    for block,names in (("dual",["lambda_0","lambda_1","lambda_2"]),("target",["u_0","u_1","u_2"])):
        require(protocol.scalar(checkpoint["ncomp_"+block],"int")==3 and list(checkpoint["names_"+block])==names,
                "checkpoint component names/count mismatch")
        array=checkpoint["state_"+block]
        require(list(array.shape)==receipt["raw_state_shapes"][block] and array.dtype==state[block].dtype
                and array.tobytes()==state[block].tobytes(),"checkpoint physical state mismatch")
    temporal=strict_json(str(checkpoint["temporal_restart_state"].item()))
    require(cbor(temporal)==cbor(receipt["temporal"]) and cbor(temporal["clock"])==cbor(dict(time=time.hex(),macro_step=step)),
            "temporal checkpoint mismatch")
    manifest=strict_json(str(checkpoint["pops_checkpoint_manifest"].item()))
    require(manifest["runtime_kind"]=="uniform" and cbor(manifest["clock"])==cbor(temporal["clock"])
            and protocol.identity_token(manifest["artifact_identity"],"artifact")==pins["artifact_identity"]
            and protocol.identity_token(manifest["bind_identity"],"bind")==receipt["bind_identity"],
            "checkpoint envelope artifact/bind/clock mismatch")
    initial=phase in {"initial","outside_before","safe_rebind_initial"}
    require(type(manifest["schema_version"]) is int and manifest["schema_version"]==(2 if initial else 1),
            "initial/run checkpoint schema mismatch")
    if initial:
        require(time.hex()=="0x0.0p+0" and step==0 and manifest["run_identity"] is None
                and receipt["run_identity"] is None and cbor(manifest["origin"])==cbor(dict(schema_version=1,kind="bound_initial")),
                "invented initial run provenance")
    else:
        require(protocol.identity_token(manifest["run_identity"],"run")==receipt["run_identity"]
                and "origin" not in manifest,"run checkpoint provenance mismatch")
    require(set(checkpoint)==set(manifest["arrays"])|{"pops_checkpoint_manifest","pops_restart_identity"},
            "checkpoint member inventory differs")
    for name,array in manifest["arrays"].items():
        require(array==typed_array(checkpoint[name]),"checkpoint typed array digest mismatch")
    payload={key:value for key,value in manifest.items() if key!="restart_identity"}
    require(manifest["restart_identity"]["hexdigest"]==digest(cbor(dict(protocol="pops.identity",domain="restart",schema_version=1,payload=payload)))
            and protocol.identity_token(manifest["restart_identity"],"restart")==str(checkpoint["pops_restart_identity"].item()),
            "checkpoint envelope digest mismatch")
    raw,offsets=checkpoint["program_exchange_state"],checkpoint["program_exchange_offsets"]
    images=empty_exchange_images(raw,offsets,pins["ranks"])
    return dict(state=state,receipt=receipt,images=images,ownership=mode)


def same_states(left,right,label):
    for name in ("dual","target","cell_volumes","coverage","valid_cells"):
        require(left["state"][name].tobytes()==right["state"][name].tobytes(),label+" changed exact "+name)


def check_junit(raw,pins,rank):
    tree=ET.fromstring(raw)
    cases=[case for case in tree.iter("testcase") if case.attrib.get("name")==TEST_NAME]
    require(len(cases)==1 and not any(child.tag in {"failure","error","skipped"} for child in cases[0]),
            "M18 JUnit missing/failed/skipped witness")
    properties={row.attrib["name"]:row.attrib["value"] for row in cases[0].iter("property")}
    expected=dict(dim=2,rank=rank,mpi_ranks=pins["ranks"],artifact_identity=pins["artifact_identity"],
                  native_sha256=pins["native_sha256"],native_abi_key=pins["abi_key"],native_capability_abi=5)
    require(all(properties.get(key)==str(value) for key,value in expected.items()),"JUnit native/artifact/rank identity differs")


def receive(path,external_sha):
    require(type(external_sha) is str and re.fullmatch("[0-9a-f]{64}",external_sha),"external owner seal is required")
    raw=path.read_bytes()
    require(digest(raw)==external_sha,"external owner manifest SHA mismatch")
    pins=strict_json(raw)
    require(set(pins)=={"schema","source_commit","native_sha256","abi_key","dimension","ranks","artifact_identity",
                        "native_capability_abi","native_system_package_abi","provenance","sources","junit_by_rank","phases"}
            and pins["schema"]=="sol61.m18-owner-pins@1","owner contract mismatch")
    require(type(pins["dimension"]) is int and pins["dimension"]==2 and type(pins["ranks"]) is int and pins["ranks"] in (1,2)
            and type(pins["native_capability_abi"]) is int and pins["native_capability_abi"]==5
            and type(pins["native_system_package_abi"]) is int and pins["native_system_package_abi"]==7,
            "native dimension/rank/ABI contract mismatch")
    require(re.fullmatch("[0-9a-f]{40}",pins["source_commit"]) and re.fullmatch("[0-9a-f]{64}",pins["native_sha256"])
            and re.fullmatch("pops.artifact.v[1-9][0-9]*:sha256:[0-9a-f]{64}",pins["artifact_identity"]),"invalid source/native/artifact identity")
    require(set(pins["phases"])==set(PHASES) and set(pins["sources"])=={"fixture","example","snapshot"}
            and len(pins["junit_by_rank"])==pins["ranks"],"external phase/source/JUnit inventory mismatch")
    all_rows=[pins["provenance"],*pins["sources"].values(),*pins["junit_by_rank"],
              *(row for phase in pins["phases"].values() for row in phase.values())]
    require(len({row["path"] for row in all_rows})==len(all_rows),"external file paths overlap")
    base=path.parent
    checked_sources={key:pinned(base,row) for key,row in pins["sources"].items()}
    authored_contract(checked_sources["example"][1].decode())
    provenance=strict_json(pinned(base,pins["provenance"])[1])
    require(provenance["schema"]=="sol61.m18-native-provenance@1" and provenance["constants"]==CONSTANTS
            and provenance["artifact_identity"]==pins["artifact_identity"] and provenance["dimension"]==2
            and provenance["ranks"]==pins["ranks"] and len(provenance["native_by_rank"])==pins["ranks"],
            "native provenance/constants differ")
    require(provenance["source_sha256"]=={name:digest(data[1]) for name,data in checked_sources.items()},
            "executed fixture/example/support source SHA mismatch")
    compiled_contract(provenance["program_ir"],provenance["program_ir_hash"])
    for native in provenance["native_by_rank"]:
        require(native["native_sha256"]==pins["native_sha256"] and native["abi_key"]==pins["abi_key"]
                and native["native_dimension"]==2 and native["native_capabilities"]["abi_version"]==5
                and len(native["system_packages"])==2
                and all(row["abi_version"]==7 and re.fullmatch("[0-9a-f]{64}",row["binary_sha256"])
                        for row in native["system_packages"]),"executed native package identity differs")
    for rank,row in enumerate(pins["junit_by_rank"]):
        check_junit(pinned(base,row)[1],pins,rank)
    snapshots={phase:snapshot(base,pins,phase) for phase in PHASES}
    initial=snapshots["initial"]
    require(np.all(initial["state"]["dual"]==0.),"initial dual seed is not zero")
    target=initial["state"]["target"]
    require(all(cone(target[:,y,x])=="interior" for y,x in np.ndindex(4,5)),"initial target not interior")
    indices=np.arange(20,dtype=float).reshape(4,5)
    reference=np.stack((-.18+.018*indices,.14*np.sin(.23*indices),-.12+.01*indices))
    recipe_error=0.
    for y,x in np.ndindex(4,5):
        population=WEIGHTS*np.exp(BASIS.T@reference[:,y,x])
        moments=np.array([math.fsum(float(a*p) for a,p in zip(row,population,strict=True)) for row in BASIS])
        recipe_error=max(recipe_error,float(np.max(np.abs(moments-target[:,y,x]))))
    require(recipe_error<=2.e-15,"saved initial target differs from the specified twenty-cell input recipe")
    metrics={}
    for phase in ("accepted","continuous","restored","replayed","safe_rebind_accepted"):
        row=snapshots[phase]
        require(row["state"]["target"].tobytes()==target.tobytes(),"read-only target changed")
        metrics[phase]=check_solved(row["state"]["dual"],row["state"]["target"])
        step=2 if phase in ("continuous","replayed") else 1
        require(row["receipt"]["time"].hex()==(DT*step).hex() and row["receipt"]["macro_step"]==step,
                "accepted publication clock differs")
    same_states(snapshots["accepted"],snapshots["restored"],"restart")
    same_states(snapshots["continuous"],snapshots["replayed"],"replay")
    same_states(initial,snapshots["safe_rebind_initial"],"safe rebind initial")
    same_states(snapshots["accepted"],snapshots["safe_rebind_accepted"],"safe rebind solve")
    outside=snapshots["outside_before"]
    expected=target.copy()
    expected[:,1,2]=(1.,0.,1.1)
    require(outside["state"]["target"].tobytes()==expected.tobytes() and np.all(outside["state"]["dual"]==0.)
            and cone(expected[:,1,2])=="outside","outside-cone witness differs")
    for phase in ("outside_rejected_0","outside_rejected_1"):
        row=snapshots[phase]
        same_states(outside,row,"failed attempt")
        require(row["receipt"]["time"].hex()=="0x0.0p+0" and row["receipt"]["macro_step"]==0
                and cbor(row["receipt"]["temporal"])==cbor(outside["receipt"]["temporal"])
                and row["images"]==outside["images"],"failed attempt changed accepted clock/temporal/mailbox")
        failures=row["receipt"]["failures"]
        require(len(failures)==pins["ranks"] and all(failure and failure[0]=="RuntimeError"
                and "coupled_implicit failed:" in failure[1] for failure in failures),"native generic refusal diagnostic missing")
    require(all(row["ownership"]==initial["ownership"] and row["receipt"]["geometry"]==initial["receipt"]["geometry"]
                and row["receipt"]["local_boxes_by_rank"]==initial["receipt"]["local_boxes_by_rank"] for row in snapshots.values()),
            "support/geometry changed")
    return dict(schema="sol61.m18-offline-reception@2",exchange_reader_contract=EMPTY_EXCHANGE_READER,
                status="received",owner_sha256=external_sha,
                source_commit=pins["source_commit"],native_sha256=pins["native_sha256"],abi_key=pins["abi_key"],
                artifact_identity=pins["artifact_identity"],dimension=2,ranks=pins["ranks"],ownership=initial["ownership"],
                phases=list(PHASES),metrics=metrics,outside_refusals=2,readonly_target_bitexact=True,
                initial_target_recipe_linf_error=recipe_error,
                safe_rebind=True,accepted_retry_same_outside_runtime=False,native_execution_in_oracle=False,
                boundary_label="pure mathematical classifier only; no native boundary run",
                limitation="moderate five-node M18 closure; no near-boundary/M19/AMR/Vlasov/BGK qualification")


def contract():
    return dict(schema="sol61.m18-offline-contract@2",exchange_reader_contract=EMPTY_EXCHANGE_READER,
                empty_exchange_image_lengths={"POPSEX01":16,"POPSEX02":32},
                status="pending_external_native_receipts",
                owner_schema="sol61.m18-owner-pins@1",external_owner_sha256_required=True,
                sources=["fixture","example","snapshot"],provenance="provenance.json",
                phases={phase:["state","receipt","checkpoint"] for phase in PHASES},
                junit_by_rank="one authentic JUnit file per rank, exact named witness and native properties",
                native_capability_abi=5,native_system_package_abi=7,dimension=2,ranks=[1,2],constants=CONSTANTS)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins",type=Path)
    parser.add_argument("--owner-sha256")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=receive(args.pins,args.owner_sha256) if args.pins else contract()
    raw=json.dumps(result,indent=2,allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(raw)
    print(raw)


if __name__=="__main__":
    main()
