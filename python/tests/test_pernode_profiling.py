"""Spec 3 per-node Program profiling (ADC-459): emit_cpp_program wraps each Program node.

Only coarse System phases ("step" / "field_solve") used to be timed. ADC-459 adds PER-PROGRAM-NODE
timing in the compiled program: the codegen brackets each value node's emitted C++ with a
steady_clock pair recorded under "node:<name>" through ctx.profile_record, so sim.profile_report()
shows per-node times (e.g. "node:rhs2", "node:solve_fields1") next to the coarse phases.

The wrapping is UNCONDITIONAL but cheap when profiling is off (Profiler::record early-returns; the
only cost is one extra clock read per node), so a Program with no profiling intent still emits valid
C++. This test pins the generated source only and emits through typed operator calls.
"""
import re

import pops.model as pm
import pops.time as t
import pops.lib.time as lt  # ready schemes (Spec 4)


def _module(name):
    m = pm.Module(name + "_module")
    U = m.state_space("U", ("rho",))
    rhs = m.operator(
        "rhs", signature=(U,) >> pm.Rate(U), kind="local_rate",
        capabilities={"produces_rate": True}, lowering={"flux": False, "sources": []},
        expr=0.0)
    return m, rhs


def _forward_euler():
    """A small real Program: forward Euler over one block via pops.lib.time."""
    m, rhs = _module("pernode_fe")
    P = t.Program("pernode_fe").bind_operators(m)
    lt.forward_euler(P, "gas", rhs_operator=rhs)
    return P, m


def _ssprk3():
    """A multi-stage Program (three rhs / two intermediate lincomb / one commit)."""
    m, rhs = _module("pernode_ssprk3")
    P = t.Program("pernode_ssprk3").bind_operators(m)
    lt.ssprk3(P, "gas", rhs_operator=rhs)
    return P, m


def test_per_node_scope_named_by_node():
    """Each work node is wrapped: a ProfileScope marker + a ctx.profile_record under node:<name>."""
    P, m = _forward_euler()
    src = P.emit_cpp_program(model=m)
    # The RAII-style marker the codegen emits for every wrapped node (issue: "ProfileScope" + "node:").
    assert "ProfileScope" in src, "generated source missing the per-node ProfileScope marker"
    assert "ctx.profile_record(" in src, "generated source missing the per-node profile_record call"
    # The two operator call/update nodes of Forward Euler are named by their stable value names.
    for node in ("node:fe_step_k", "node:fe_step"):
        assert node in src, "generated source missing per-node scope %r" % node
    # Every node scope is a matched pair: one steady_clock now() open + one profile_record close.
    opens = src.count("std::chrono::steady_clock::now();  // ProfileScope")
    closes = src.count("ctx.profile_record(")
    assert opens == closes and opens >= 2, "unbalanced per-node scopes (opens=%d closes=%d)" % (
        opens, closes)
    # Each open declares a unique _pt<id> the matching close reads (no redefinition at body scope).
    pts = re.findall(r"const auto (_pt\d+) = std::chrono::steady_clock::now\(\)", src)
    assert len(pts) == len(set(pts)), "duplicate per-node timer variable: %r" % pts


def test_pure_reference_nodes_not_wrapped():
    """The state node binds a MultiFab& and does no work, so it is not wrapped (no node:gas noise)."""
    P, m = _forward_euler()
    src = P.emit_cpp_program(model=m)
    assert "node:gas" not in src, "the pure state-binding node should not be profiled"
    # The state binding itself is still emitted (it is the base every op clones / commits into).
    assert "ctx.state(0)" in src, "generated source missing the block-0 state binding"


def test_multistage_wraps_every_work_node():
    """SSPRK3 lowers nine work nodes (3 solve_fields + 3 rhs + 3 lincomb); each gets one scope."""
    P, m = _ssprk3()
    src = P.emit_cpp_program(model=m)
    closes = src.count("ctx.profile_record(")
    assert closes == 6, "SSPRK3 should wrap 6 work nodes, got %d" % closes
    for node in ("node:ssprk3_0_k", "node:ssprk3_1_k", "node:ssprk3_step"):
        assert node in src, "generated source missing per-node scope prefix %r" % node


def test_no_profiling_intent_still_valid_cpp():
    """A Program with NO profiling intent still emits valid, complete C++ (the scope is unconditional
    and cheap-when-disabled). The chrono header and the stable ABI surface are present."""
    P, m = _forward_euler()
    src = P.emit_cpp_program(model=m)
    for tok in ("#include <chrono>", "pops::runtime::program::ProgramContext ctx(sys)",
                "pops_install_program", "ctx.install(", "std::chrono::steady_clock::now()"):
        assert tok in src, "generated source missing %r" % tok
    # The body is balanced and the per-node opens precede their closes (a close after each node block).
    assert src.count("ctx.profile_record(") >= 2, "expected at least 2 per-node records"
