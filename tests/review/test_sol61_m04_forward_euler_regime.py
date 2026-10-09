"""Source/math counterprobes; no compilation, binding or native M04 run."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
EXAMPLE=ROOT/"examples/migration/scientific/api040_m04_advection_diffusion.py"
spec=importlib.util.spec_from_file_location("independent_m04_regime",Path(__file__).with_name("sol61_m04_forward_euler_regime_oracle.py"))
oracle=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=oracle
spec.loader.exec_module(oracle)


def test_original_regime_failure_and_new_prediction_do_not_replace_native_evidence():
    original=oracle.prediction("combined_bound")
    control=oracle.prediction("fixed_courant")
    assert original["native"] is control["native"] is False
    assert original["order_guard"]==control["order_guard"]==.7
    assert not original["order_guard_met"]
    assert control["order_guard_met"]
    assert original["observed_orders"]==pytest.approx((.6063534245372065,.701766241133962),abs=2.e-12)
    assert control["observed_orders"]==pytest.approx((.9786458344361255,.9884487638697017),abs=2.e-12)
    for row in control["rows"]:
        assert row["l1"]<=oracle.CAPS[row["n"]]
        assert row["courant"]==.125
        assert row["combined_frequency_product"]<=.9


@pytest.mark.parametrize("policy",("combined_bound","fixed_courant"))
@pytest.mark.parametrize("n",(32,64,128))
def test_independent_stencil_mode_mass_and_maximum_principle(policy,n):
    dt=oracle.step(n,policy)
    initial=oracle.cell_means(n,0.)
    value=oracle.stencil(initial,dt)
    np.testing.assert_allclose(value,oracle.mode(n,dt),rtol=0,atol=3.e-12)
    assert abs(value.mean()-initial.mean())<2.e-14
    assert value.min()>=initial.min()-2.e-14
    assert value.max()<=initial.max()+2.e-14
    for h in oracle.schedule(dt):
        c,r=h*n,h*.01*n*n
        assert min(c+r,1-c-2*r,r)>=0
        assert c+r+(1-c-2*r)+r==pytest.approx(1.)
    clock=0.
    for duration in oracle.schedule(dt):
        clock+=duration
    assert clock==.1


@pytest.mark.parametrize("n",(32,64,128))
def test_counter_equations_cannot_claim_forward_euler_same_dt(n):
    dt=oracle.step(n,"combined_bound")
    baseline=oracle.mode(n,dt)
    for kwargs in (dict(temporal="ssprk2"),dict(velocity=-1.),dict(diffusivity=0.)):
        assert np.max(abs(oracle.mode(n,dt,**kwargs)-baseline))>1.e-5


def authored_namespace():
    tree=ast.parse(EXAMPLE.read_text())
    names={"VELOCITY","DIFFUSIVITY","AMPLITUDE","T_END","RESOLUTIONS","SAFETY_FACTOR","FIXED_ADVECTIVE_COURANT","CRITERIA"}
    selected=[]
    for node in tree.body:
        if isinstance(node,ast.ImportFrom) and node.module.startswith("pops"):
            selected.append(node)
        elif isinstance(node,ast.Import) and any(alias.name=="pops" for alias in node.names):
            selected.append(node)
        elif isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in names:
            selected.append(node)
        elif isinstance(node,ast.FunctionDef) and node.name=="author_case":
            selected.append(node)
    namespace={}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(EXAMPLE),"exec"),namespace)
    namespace.update(TEMPORAL_METHOD="forward_euler",DIFFUSION_VARIANT="x_only",TRANSVERSE_DIFFUSIVITY=0.)
    import_spec=importlib.util.spec_from_file_location("authored_m04_math",EXAMPLE.with_name("api040_m04_oracle.py"))
    authored=importlib.util.module_from_spec(import_spec)
    import_spec.loader.exec_module(authored)
    namespace["frequencies"]=authored.frequencies
    loop=next(node for node in tree.body if isinstance(node,ast.For))
    steps=[]
    for statement in loop.body:
        if isinstance(statement,ast.Assign) and any(isinstance(target,ast.Tuple) for target in statement.targets):
            break  # Stop before author_case; only the actual declared dt preflight is evaluated here.
        steps.append(statement)
    return namespace,steps


@pytest.mark.parametrize("policy",("combined_bound","fixed_courant"))
@pytest.mark.parametrize("n",(32,64,128))
def test_actual_public_case_fixed_dt_resolves_and_emits_original_forward_euler(policy,n):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    namespace,steps=authored_namespace()
    namespace.update(n=n,STEP_POLICY=policy)
    exec(compile(ast.Module(body=steps,type_ignores=[]),str(EXAMPLE),"exec"),namespace)
    dt=namespace["dt"]
    assert dt==oracle.step(n,policy)
    assert namespace["RESOLUTIONS"]==(32,64,128)
    assert [namespace[name] for name in ("VELOCITY","DIFFUSIVITY","AMPLITUDE","T_END","SAFETY_FACTOR")]==[1.,.01,.2,.1,.9]
    assert namespace["CRITERIA"]["minimum_observed_order"]==.7
    assert namespace["CRITERIA"]["density_l1_max"]==oracle.CAPS
    case,frame=namespace["author_case"](dt)
    layout=namespace["Uniform"](namespace["CartesianGrid"](frame=frame,cells=(n,n),
                                periodic=namespace["PeriodicAxes"](frame.axes)))
    resolved=namespace["pops"].resolve(case,layout=layout)
    program=resolved.time
    assert program._step_strategy.dt==dt
    ir=program._serialize(include_provenance=False)
    assert len(ir["commits"])==1
    assert ir["name"]=="ForwardEuler"
    rates=[node for node in ir["nodes"] if node["op"]=="diffusive_rhs"]
    assert len(rates)==1  # Original ForwardEuler has one RHS evaluation, not SSPRK2's two.
    assert rates[0]["inputs"]==[0]
    assert [node["op"] for node in ir["nodes"]]==["state","diffusive_rhs","linear_combine"]
    code=emit_cpp_program(program,model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert "diffusion_prepared_1.stage_accepted_exchanges" in code
    assert "ctx.commit_many({{&u0, &u2}})" in code


def test_runner_policy_is_explicit_and_historical_receipt_pin_is_preserved():
    tree=ast.parse((ROOT/"docs/development/api_040/run_scientific_checks.py").read_text())
    cases=ast.literal_eval(next(node.value for node in tree.body if isinstance(node,ast.Assign)
                                and isinstance(node.targets[0],ast.Name) and node.targets[0].id=="CASES"))
    for name in ("m04","m04-isotropic","m04-ssprk2"):
        assert cases[name][2]["POPS_API040_M04_STEP_POLICY"]=="combined_bound"
    control=cases["m04-fe-fixed-courant"][2]
    assert control==dict(POPS_API040_M04_DIFFUSION="x_only",POPS_API040_M04_METHOD="forward_euler",
                         POPS_API040_M04_STEP_POLICY="fixed_courant")
    receipt=ROOT/"docs/development/api_040/evidence/83b2b12/m04-xonly-220b-openmp1/scientific-receipt.json"
    assert hashlib.sha256(receipt.read_bytes()).hexdigest()==oracle.HISTORICAL_RECEIPT_SHA


def test_physics_spatial_method_and_forward_euler_authoring_body_match_frozen_99d651c1():
    node=next(node for node in ast.parse(EXAMPLE.read_text()).body
              if isinstance(node,ast.FunctionDef) and node.name=="author_case")
    digest=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
    assert digest=="ceee2c39d507204fd316409b8db3a9d0fd874067821bf064d9b12e604f58f3e5"


@pytest.mark.parametrize("bad",(0,-1,float("nan"),float("inf")))
def test_invalid_duration_cannot_start_a_schedule(bad):
    with pytest.raises(ValueError):
        oracle.schedule(bad)


@pytest.mark.parametrize("method,diffusion",(("ssprk2","x_only"),("forward_euler","isotropic")))
def test_control_refuses_inadmissible_method_or_tensor_before_authoring(monkeypatch,method,diffusion):
    import os
    monkeypatch.setenv("POPS_API040_M04_STEP_POLICY","fixed_courant")
    monkeypatch.setenv("POPS_API040_M04_METHOD",method)
    monkeypatch.setenv("POPS_API040_M04_DIFFUSION",diffusion)
    tree=ast.parse(EXAMPLE.read_text())
    nodes=[]
    begun=False
    for node in tree.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=="VELOCITY":
            begun=True
        if isinstance(node,ast.FunctionDef):
            break
        if begun:
            nodes.append(node)
    with pytest.raises(ValueError,match="requires x-only ForwardEuler"):
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(EXAMPLE),"exec"),dict(os=os))


@pytest.mark.parametrize("receipt,native",((None,None),("a"*64,None),(None,"b"*64)))
def test_future_control_needs_external_pins_before_reading_any_states(tmp_path,receipt,native):
    with pytest.raises(ValueError,match="external owner"):
        oracle.receive_control(tmp_path,receipt,native)
    assert not list(tmp_path.iterdir())
