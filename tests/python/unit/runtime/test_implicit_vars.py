"""A partial implicit update is an exact typed source, never a component-name mask.

The final Program route does not accept ``implicit_vars=["rho_u", ...]``.  Such a string list is
ambiguous across StateSpaces and used to let a block policy choose time integration behind the
Program.  The model instead declares the complete stiff source operator: components that are not
implicit have an exact zero contribution.  The ordinary IMEX residual then keeps those components
equal to the stage predictor while Newton solves the selected source contribution.
"""

from __future__ import annotations

import inspect

import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.lib.time import IMEX
from pops.physics._facade import Model
from pops.problem import Case
from pops.solvers.nonlinear import LocalNewton


def _partial_momentum_program():
    model = Model("typed_partial_implicit")
    rho, momentum_x, momentum_y = model.conservative_vars(
        "rho",
        "momentum_x",
        "momentum_y",
        roles=("Density", "MomentumX", "MomentumY"),
    )
    explicit = model.rate("transport", flux=False, sources=())
    drag = model.source_term(
        "momentum_drag",
        [
            0.0 * rho,
            -50.0 * momentum_x,
            -50.0 * momentum_y,
        ],
    )
    state = next(handle for handle in model.declaration_index().records() if handle.kind == "state")
    block = Case("partial_implicit_case").block("fluid", model=model, states=(state,))
    state_instance = block[state]
    program = IMEX(
        state_instance,
        explicit_operator=explicit,
        implicit_operator=drag,
        implicit_solver=LocalNewton(
            tolerance=1.0e-12,
            max_iterations=12,
            finite_difference_step=1.0e-7,
        ),
    )
    return model, state_instance, explicit, drag, program


def test_partial_implicit_source_is_one_authenticated_program_operator():
    model, _state, _explicit, drag, program = _partial_momentum_program()

    assert drag.kind == "local_source"
    assert program.validate() is True
    nodes = program._serialize()["nodes"]
    solve = next(node for node in nodes if node["op"] == "solve_local_nonlinear")
    sources = [node for node in solve["attrs"]["residual_block"] if node["op"] == "source"]
    assert len(sources) == 1
    assert sources[0]["attrs"]["source"] == "momentum_drag"
    identity = sources[0]["attrs"]["operator_handle"]["handle"]
    assert identity["local_id"] == drag.local_id
    assert identity["registered_operator_name"] == drag.registered_operator_name

    consume = next(node for node in nodes if node["op"] == "solve_outcome")
    assert consume["attrs"]["action"]["kind"] == "fail_run"
    assert len(program.commits()) == 1

    generated = emit_cpp_program(program, model=model)
    assert "pops::detail::mat_inverse<3>(" in generated
    assert "for (int it_ = 0;" in generated


def test_imex_has_no_component_name_or_role_mask_surface():
    parameters = tuple(inspect.signature(IMEX).parameters)
    assert "implicit_vars" not in parameters
    assert "implicit_roles" not in parameters

    _model, state, explicit, drag, _program = _partial_momentum_program()
    with pytest.raises(TypeError, match="unexpected keyword"):
        IMEX(
            state,
            explicit_operator=explicit,
            implicit_operator=drag,
            implicit_solver=LocalNewton(),
            implicit_vars=("momentum_x", "momentum_y"),
        )
