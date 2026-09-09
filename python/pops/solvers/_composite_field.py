"""Prepared synchronized hierarchy solver for an authored joint field equation."""
from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from pops.descriptors import Descriptor
from pops.identity.scalar import exact_cpp_int, scalar_data, scalar_literal, scalar_cpp
from pops.native_components import PreparedNativeComponent
from .providers import (PreparedHierarchySolverProvider, PreparedHierarchySolverUsePolicy,
    PreparedHierarchySolverUseFacts, PreparedHierarchyConvergenceContract,
    PreparedHierarchyFlatExecution, PreparedHierarchySolverNativeEmission,
    register_prepared_hierarchy_solver_provider)


def _options(values, where):
    from pops.model._bind_schema_data import literal_value
    if not isinstance(values, Mapping) or set(values) != {"max_iter", "rel_tol", "abs_tol", "restart"}:
        raise TypeError("%s has an incomplete composite field solver contract" % where)
    maximum = exact_cpp_int(values["max_iter"], where=where, minimum=1)
    restart = exact_cpp_int(values["restart"], where=where, minimum=1)
    relative = literal_value(values["rel_tol"], where=where)
    absolute = literal_value(values["abs_tol"], where=where)
    if isinstance(relative, bool) or not 0 < relative < 1 or isinstance(absolute, bool) or absolute < 0:
        raise ValueError("%s requires 0 < rel_tol < 1 and abs_tol >= 0" % where)
    return {"max_iter": maximum, "restart": restart, "rel_tol": scalar_data(relative), "abs_tol": scalar_data(absolute)}


def _field_apply(operator):
    from pops.fields._program_problem import validate_field_apply
    block = operator.attrs.get("apply_block", ())
    rows = tuple(row for row in block if row.op == "field_problem_apply")
    if len(rows) != 1 or any(row.op not in {"apply_in", "apply_out", "field_problem_apply"} for row in block):
        raise ValueError("composite field solver requires one exact field equation apply")
    apply = rows[0]
    validate_field_apply(apply)
    if apply.inputs[1].id != operator.attrs["apply_in"].id or apply.inputs[0].id != operator.attrs["apply_out"].id:
        raise ValueError("composite field apply changed its exact operator input/output")
    return apply


def _validate(facts, operator, where):
    if facts.target not in (None, "amr_system") or facts.scope != "hierarchy" or facts.problem_kind != "general_field_hierarchy":
        raise ValueError("%s requires a synchronized AMR field hierarchy" % where)
    apply = _field_apply(operator)
    if operator.attrs.get("scope") != "hierarchy" or facts.components != apply.attrs["ncomp"]:
        raise ValueError("%s changes field hierarchy component or scope authority" % where)
    return facts


def _author(program, problem, prepared, name, provider):
    from pops.linalg import LinearProblem
    from pops.fields._prepared_nullspace_registry import PreparedNullspaceContracts, prepared_nullspace_provider_from_identity
    from pops.time.values import _resolve_handle
    from pops.time.solve_outcome import SolveOutcome
    if not isinstance(problem, LinearProblem):
        raise TypeError("composite field solver requires the normalized LinearProblem authority")
    options = provider.authenticate_prepared(prepared)
    operator = program._canonical_value(_resolve_handle(problem.operator))
    rhs = program._canonical_value(_resolve_handle(problem.rhs))
    apply = _field_apply(operator)
    ncomp = apply.attrs["ncomp"]
    if rhs.op != "field_problem_load" or rhs.attrs.get("field_problem_identity") != apply.attrs["field_problem_identity"]:
        raise ValueError("composite field load and operator have different physical authorities")
    identity = problem.canonical_nullspace_provider()
    nullspace = problem.canonical_nullspace_contract()
    gauge = problem.canonical_gauge_contract()
    declaration = prepared_nullspace_provider_from_identity(identity)
    declaration.validate_use(contracts=PreparedNullspaceContracts(nullspace["contract"], gauge), components=ncomp,
        operator_properties=problem.properties.canonical_data(), where="composite field kernel")
    facts = PreparedHierarchySolverUseFacts(target=None, scope="hierarchy", problem_kind="general_field_hierarchy",
        domain=operator.attrs["domain"], range=operator.attrs["range"], components=ncomp,
        singular_nullspace=declaration.singular, extensions={})
    provider.use_policy.validate(facts, operator=operator, where="composite field solver")
    guess = None if problem.initial_guess is None else program._canonical_value(_resolve_handle(problem.initial_guess))
    if guess is not None and (guess.vtype != "scalar_field" or guess.attrs.get("ncomp") != ncomp):
        raise ValueError("composite field initial guess changes the physical tuple")
    relative, absolute, maximum = provider.convergence.values(options, where="composite field solver")
    attrs = {"tol": scalar_literal(relative), "abs_tol": scalar_literal(absolute), "max_iter": maximum,
        "has_guess": guess is not None, "ncomp": ncomp, "scope": "hierarchy",
        "operator_properties": problem.properties.canonical_data(),
        "nullspace_provider": identity, "nullspace_contract": nullspace, "gauge_contract": gauge,
        "hierarchy_solver_provider": provider.authority(), "hierarchy_solver_options": deepcopy(options),
        "hierarchy_solver_identity": prepared.identity.token, "solver_identity": prepared.identity.token,
        "hierarchy_block_index": -1, "hierarchy_field_identity": apply.attrs["field_problem_identity"],
        "hierarchy_field_coefficients": apply.inputs[2].id,
        "problem_kind": "general_field_hierarchy", "hierarchy_use_facts": {}}
    token = program._new("scalar_field", "solve_linear", (operator, rhs) if guess is None else (operator, rhs, guess),
        attrs, name, None, space=rhs.space, point=rhs.point, inherit_state_ref=False)
    def project(outcome):
        return program._new("scalar_field", "solve_outcome_component", (outcome,), {"index": 0, "ncomp": ncomp},
            name or token.name, None, point=token.point, inherit_state_ref=False)
    return SolveOutcome(program, token, project, name or token.name)


def _native_options(node, options):
    from pops.fields._program_expression import decode_field_literal
    apply = _field_apply(node.inputs[0])
    n = apply.attrs["ncomp"]
    pairs = [('coefficient_components', 'std::uint64_t{%d}' % apply.inputs[2].attrs["ncomp"]),
             ('restart', 'std::uint64_t{%d}' % options["restart"])]
    pairs.extend(('reaction.%d.%d' % (i,j), 'static_cast<double>(%s)' % scalar_cpp(decode_field_literal(apply.attrs["reaction"][i*n+j])))
                 for i in range(n) for j in range(n))
    basis = node.attrs["nullspace_contract"]["contract"]
    gauge = node.attrs["gauge_contract"]
    modes = basis.get("modes", ())
    coordinates = gauge.get("coordinates", ())
    if basis.get("basis") == "shared-constant-vector":
        modes = (tuple(scalar_data(1) for _ in range(n)),)
        coordinates = ("static_cast<double>(%s) / double(%d)" % (scalar_cpp(gauge["value"]), n),)
    else:
        coordinates = tuple('static_cast<double>(%s)' % scalar_cpp(x) for x in coordinates)
    pairs.append(('modes.count', 'std::uint64_t{%d}' % len(modes)))
    for k, row in enumerate(modes):
        pairs.extend(('modes.%d.%d' % (k,i), 'static_cast<double>(%s)' % scalar_cpp(decode_field_literal(x))) for i,x in enumerate(row))
        pairs.append(('gauge.%d' % k, coordinates[k]))
    physical = "periodic" if apply.attrs["physical_boundary"] == "periodic" else "neumann"
    entries = ', '.join('{%s, %s}' % (json.dumps(key), value) for key,value in pairs)
    return ('[&]() { pops::PreparedProviderOptions options{"pops.hierarchy.general-field@1", {%s}}; '
            'for (int axis=0; axis<pops::kNativeDimension; ++axis) for (int side=0; side<2; ++side) '
            'options.values.emplace("physical." + std::to_string(axis) + "." + std::to_string(side), std::string(%s)); return options; }()'
            % (entries, json.dumps(physical)))


def _emit(request, provider, options):
    node = request.node
    configured = ('ctx.configure_hierarchy_field_solver(%d, %d, %s, %s, %s, %s, '
                  '{"pops.general-field.coefficients", "pops.general-field.rhs"}, "pops.general-field.solution", %s);'
                  % (node.id, request.components, json.dumps(node.attrs["hierarchy_field_identity"]),
                     json.dumps(provider.provider_id), json.dumps(node.attrs["hierarchy_solver_identity"]),
                     json.dumps("pops.operator.general-field.hierarchy@1"), _native_options(node, options)))
    return PreparedHierarchySolverNativeEmission(configure=(
        "ctx.register_hierarchy_tensor_solver_provider(std::make_shared<pops::elliptic::nd::CompositeGeneralFieldProvider<pops::kNativeDimension>>());", configured),
        solve=("pops::SolveOutcome %s = ctx.solve_hierarchy_field(%d, %s, %s, %d);" %
               (request.report_name, node.id, request.relative_tolerance_cpp, request.absolute_tolerance_cpp, request.max_iterations),))


_PROVIDER = register_prepared_hierarchy_solver_provider(PreparedHierarchySolverProvider(
    provider_id="pops.hierarchy.general-field.gmres@1", interface_version=1,
    emitter_id="pops.codegen.hierarchy.general-field@1", option_schema="pops.hierarchy.general-field.controls@1",
    capabilities=frozenset({"pops.hierarchy.general-field.component-matrix@1", "pops.hierarchy.field-owned-storage@1"}),
    use_policy=PreparedHierarchySolverUsePolicy("pops.use-policy.general-field-hierarchy", 1,
        frozenset({"pops.target.amr-system@1", "pops.operator.general-field.hierarchy@1"}), _validate),
    convergence=PreparedHierarchyConvergenceContract("rel_tol", "abs_tol", "max_iter"),
    flat_execution=PreparedHierarchyFlatExecution.direct_provider(),
    native_component=PreparedNativeComponent.pops_builtin("pops.hierarchy.general-field",
        entry_headers=("pops/numerics/elliptic/nd/prepared_composite_general_field.hpp",)),
    option_validator=_options, author=_author, emitter=_emit))


class CompositeFieldGMRES(Descriptor):
    """Solve one joint field on a synchronized AMR hierarchy using composite GMRES.

    Spatial coarse/fine flux replacement and composite volume weights belong to
    this numerical realization; physical equations and normalization remain in FieldProblem.
    """
    category = "linear_solver"
    __pops_ir_immutable__ = True

    def __init__(self, *, max_iter: int, rel_tol: Any = 1e-10, abs_tol: Any = 0, restart: int = 30):
        self._options = _options({"max_iter": max_iter, "rel_tol": scalar_data(rel_tol),
                                 "abs_tol": scalar_data(abs_tol), "restart": restart}, "CompositeFieldGMRES")

    def field_problem_scope(self):
        from .scopes import Hierarchy
        return Hierarchy()

    def to_data(self):
        from ._composite_field import _PROVIDER
        return _PROVIDER.instance_data(self._options)

    def prepare_program_solve(self):
        from ._composite_field import _PROVIDER
        return _PROVIDER.prepare(self._options)
