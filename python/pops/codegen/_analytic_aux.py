"""Native geometry-only auxiliary launchers, regenerated at every AMR publication."""
from __future__ import annotations

import json

from ._analytic_expression_lowering import lower_analytic_components


def emit_analytic_aux_launcher(identity, producer):
    """Lower the authenticated analytic VM vocabulary to a straight-line device expression."""
    ((operations, literals),) = lower_analytic_components(
        (producer.expression.to_data(),), frame_id=producer.frame.canonical_id)
    stack = []
    statements = []
    binary = {"add": "+", "sub": "-", "mul": "*", "div": "/",
              "eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">=",
              "and": "&&", "or": "||"}
    unary = {"sqrt": "sqrt", "abs": "abs", "sin": "sin", "cos": "cos",
             "exp": "exp", "log": "log", "erf": "erf", "erfc": "erfc"}
    pairs = {"pow": "pow", "atan2": "atan2", "hypot": "hypot",
             "minimum": "fmin", "maximum": "fmax"}
    for ordinal, (operation, literal) in enumerate(zip(operations, literals, strict=True)):
        if operation == "constant":
            expression = "pops::Real(%s)" % float(literal).hex()
        elif operation in ("x", "y", "z"):
            axis = ("x", "y", "z").index(operation)
            expression = "geometry.cell_coordinate(%d, index[%d])" % (axis, axis)
        elif operation == "input":
            slot = int(literal)
            if slot == 2 * len(producer.frame.axes):
                expression = " * ".join("geometry.spacing(%d)" % a
                                        for a in range(len(producer.frame.axes)))
            else:
                axis, upper = divmod(slot, 2)
                expression = "geometry.face_coordinate(%d, index[%d] + %d)" % (axis, axis, upper)
        elif operation in binary:
            right, left = stack.pop(), stack.pop()
            expression = "(%s %s %s)" % (left, binary[operation], right)
        elif operation in pairs:
            right, left = stack.pop(), stack.pop()
            expression = "Kokkos::%s(%s, %s)" % (pairs[operation], left, right)
        elif operation in unary:
            expression = "Kokkos::%s(%s)" % (unary[operation], stack.pop())
        elif operation in ("neg", "not"):
            expression = "(%s%s)" % ("-" if operation == "neg" else "!", stack.pop())
        elif operation == "where":
            false, true, condition = stack.pop(), stack.pop(), stack.pop()
            expression = "(%s ? %s : %s)" % (condition, true, false)
        elif operation == "between":
            upper, lower, value = stack.pop(), stack.pop(), stack.pop()
            expression = "(%s <= %s && %s <= %s)" % (lower, value, value, upper)
        else:
            raise ValueError("unsupported analytic auxiliary opcode %r" % operation)
        variable = "analytic_%d" % ordinal
        statements.append("                const pops::Real %s = %s;" % (variable, expression))
        stack.append(variable)
    if len(stack) != 1:
        raise ValueError("analytic auxiliary expression has an invalid stack")
    dimension = len(producer.frame.axes)
    checks = []
    for axis, (lower, upper) in enumerate(zip(producer.frame.lower, producer.frame.upper, strict=True)):
        checks.append(
            "            if (geometry.lower()[%d] != pops::Real(%s) || geometry.upper()[%d] != pops::Real(%s)) "
            'throw std::invalid_argument("analytic auxiliary frame differs from publication geometry");'
            % (axis, float(lower).hex(), axis, float(upper).hex()))
    return "\n".join([
        "      std::vector<Dependency>{},",
        "      Provider::launcher_type::trusted_extension(",
        "          pops::PreparedProviderIdentity{%s, 1}, %s," % (
            json.dumps("pops.analytic-aux." + identity), json.dumps(identity)),
        "          [](const pops::runtime::system::AuxiliaryKernelLaunchContext<pops::kNativeDimension>& context) {",
        '            static_assert(pops::kNativeDimension == %d, "analytic auxiliary dimension mismatch");' % dimension,
        '            if (context.outputs.size() != 1 || !context.dependencies.empty() || context.storage.geometry == nullptr) '
        'throw std::logic_error("analytic auxiliary requires one output, no dependencies and exact level geometry");',
        "            const auto geometry = *context.storage.geometry;",
        *checks,
        "            auto* const candidate = context.storage.candidate;",
        '            if (candidate == nullptr) throw std::logic_error("analytic auxiliary has no candidate storage");',
        "            auto* const output_group = candidate->find(context.outputs[0].address.group);",
        '            if (output_group == nullptr) throw std::logic_error("analytic auxiliary output group is absent");',
        "            const auto output_component = context.outputs[0].address.component;",
        "            for (std::size_t local_fab = 0; local_fab < output_group->local_size(); ++local_fab) {",
        "              const auto output = output_group->fab(local_fab).view();",
        "              std::size_t cells = 1;",
        "              for (int axis = 0; axis < pops::kNativeDimension; ++axis) cells *= static_cast<std::size_t>(output.extents[axis]);",
        "              int invalid = 0;",
        '              Kokkos::parallel_reduce("pops_analytic_aux", Kokkos::RangePolicy<>(0, cells), KOKKOS_LAMBDA(const std::size_t linear, int& bad) {',
        "                std::size_t remainder = linear;",
        "                pops::Index<pops::kNativeDimension> index{};",
        "                for (int axis = 0; axis < pops::kNativeDimension; ++axis) {",
        "                  index[axis] = output.origin[axis] + static_cast<int>(remainder % static_cast<std::size_t>(output.extents[axis]));",
        "                  remainder /= static_cast<std::size_t>(output.extents[axis]);",
        "                }",
        *statements,
        "                output(index, output_component) = %s;" % stack[0],
        "                if (!Kokkos::isfinite(%s)) ++bad;" % stack[0],
        "              }, invalid);",
        '              if (invalid) throw std::runtime_error("analytic auxiliary produced a non-finite value");',
        "            }",
        "          })",
    ])
