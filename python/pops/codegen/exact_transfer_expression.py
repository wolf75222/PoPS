"""Emit a lazy rational scalar DAG directly as C++ temporal metadata algebra.

Arithmetic remains exact until a row is converted to Real. Transcendentals and
explicit rounded nodes are binary64 boundaries; their finite results enter the
remaining rational algebra as the exact represented dyadic. There is no opcode
interpreter or Python evaluation during a native transfer.
"""
from fractions import Fraction
import json


def emit_exact_transfer_dag(nodes, inputs, *, prefix="transfer_exact"):
    lines = []
    names = tuple(f"{prefix}_{i}" for i in range(len(nodes)))
    budget = "source.exact_coefficient_bytes"
    exact = "pops::ExactWeight"

    def call(index):
        return names[index]+"()"

    def scalar(value):
        return f"{exact}({value}, 1, {budget})"

    def predicate(value):
        return f"{exact}(({value}) ? 1 : 0, 1, {budget})"

    for index, node in enumerate(nodes):
        operation = node[0]
        body = []
        if operation == "input":
            expression = f"{exact}::from_double({inputs[node[1]][node[2]]}, {budget})"
        elif operation == "literal":
            literal = node[1]
            if literal.target is not None or literal.cpp is not None:
                expression = f"{exact}::from_double({literal.to_cpp()}, {budget})"
            else:
                value = Fraction(literal.to_python())
                expression = f"{exact}::from_hex({json.dumps(hex(value.numerator))}, " \
                             f"{json.dumps(hex(value.denominator))}, {budget})"
        elif operation in ("add", "sub", "mul", "div"):
            token = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[operation]
            expression = f"({call(node[1])} {token} {call(node[2])})"
        elif operation == "neg":
            expression = "(-"+call(node[1])+")"
        elif operation == "where":
            expression = f"(!{call(node[1])}.zero() ? {call(node[2])} : {call(node[3])})"
        elif operation in ("and", "or"):
            token = "&&" if operation == "and" else "||"
            expression = predicate(f"!{call(node[1])}.zero() {token} !{call(node[2])}.zero()")
        elif operation == "not":
            expression = predicate(call(node[1])+".zero()")
        elif operation == "compare":
            left, right = call(node[2]), call(node[3])
            relation = {"eq": "==", "ne": "!=", "lt": "<", "gt": ">",
                        "le": "<=", "ge": ">="}.get(node[1], node[1])
            if relation in ("==", "!="):
                condition = f"({left} == {right})"
                if relation == "!=":
                    condition = "!"+condition
            elif relation == "<":
                condition = f"({left}-{right}).negative()"
            elif relation == ">":
                condition = f"({right}-{left}).negative()"
            elif relation == "<=":
                condition = f"!({right}-{left}).negative()"
            elif relation == ">=":
                condition = f"!({left}-{right}).negative()"
            else:
                raise ValueError("unknown exact temporal comparison")
            expression = predicate(condition)
        elif operation in ("minimum", "maximum"):
            left, right = call(node[1]), call(node[2])
            choice = f"({left}-{right}).negative()"
            yes, no = (left, right) if operation == "minimum" else (right, left)
            expression = f"({choice} ? {yes} : {no})"
        elif operation == "abs":
            value = call(node[1])
            expression = f"({value}.negative() ? -{value} : {value})"
        elif operation == "sign":
            value = call(node[1])
            expression = f"({value}.zero() ? {scalar(0)} : ({value}.negative() ? {scalar(-1)} : {scalar(1)}))"
        elif operation == "pow" and nodes[node[2]][0] == "literal" and \
                Fraction(nodes[node[2]][1].to_python()).denominator == 1:
            # A literal integer power stays rational. Binary exponentiation
            # has no fixed exponent-width restriction; the ExactWeight budget
            # rejects growing products before an output is published.
            exponent = Fraction(nodes[node[2]][1].to_python()).numerator
            body += [f"auto power = {scalar(1)};",
                     f"for (const char bit : std::string_view({json.dumps(bin(abs(exponent))[2:])})) {{",
                     "  power = power*power;",
                     f"  if (bit == '1') power = power*{call(node[1])};", "}"]
            expression = f"{scalar(1)}/power" if exponent < 0 else "power"
        elif operation in ("sqrt", "exp", "sin", "cos", "log", "expm1", "pow", "rounded"):
            if operation == "rounded":
                value = call(node[1])+".value()"
            else:
                args = ", ".join(call(child)+".value()" for child in node[1:])
                value = f"std::{operation}({args})"
            expression = f"{exact}::from_double({value}, {budget})"
        elif operation in ("finite_linear_v1", "finite_projection_v1"):
            # This operation already has an explicit finite native scalar
            # realization. It is a binary64 boundary, never a hidden rational
            # interpretation of that separately declared implementation.
            from pops.codegen.program_emit_expressions import checked_expression_dag
            code, roots, bad = checked_expression_dag((index,), nodes[:index+1], inputs)
            body += code
            body += [f"if ({bad}) throw std::invalid_argument(\"nonfinite temporal finite-map result\");"]
            expression = f"{exact}::from_double({roots[0]}, {budget})"
        else:
            raise ValueError("unsupported exact temporal algebra node: "+str(operation))
        cache = f"{prefix}_cache_{index}"
        lines += [f"std::optional<{exact}> {cache};",
                  f"[[maybe_unused]] const auto {names[index]} = [&]() -> const {exact}& {{",
                  f"  if (!{cache}) {{", *("    "+line for line in body),
                  f"    {cache}.emplace({expression});", "  }", f"  return *{cache};", "};"]
    return lines, names
