"""Native residual evaluation through the existing prepared local solve contract."""
from __future__ import annotations


def evaluate_residual_expressions(expressions, *, indent="      "):
    from .cpp_writer import _cse_emit
    lines, values, statuses = _cse_emit(expressions, "pops::Real", indent,
                                       return_native_statuses=True)
    if statuses:
        lines += [indent + "int evaluation_status_ = 0;",
                  indent + "unsigned evaluation_reason_ = 0;"]
        for name in statuses:
            lines += [indent + "{",
                indent + "  const int raw_ = static_cast<int>(%s.status);" % name,
                indent + "  const int status_ = (raw_ >= 0 && raw_ <= 3) ? raw_ : 4;",
                indent + "  if (status_ > evaluation_status_) { evaluation_status_ = status_; evaluation_reason_ = %s.reason; }" % name,
                indent + "  else if (status_ == evaluation_status_ && %s.reason > evaluation_reason_) evaluation_reason_ = %s.reason;" % (name, name),
                indent + "}"]
        for status, factory in ((1, "retry"), (2, "reject"), (3, "failed"), (4, "invalid")):
            lines.append(indent + "if (evaluation_status_ == %d) return pops::LocalNonlinearEvaluationResult::%s(evaluation_reason_);" % (status, factory))
    return lines, values


def coupled_jacobian_expressions(components, sources, total, *, route):
    from pops._ir.expr import Const
    from pops._ir.lowering import diff, _s_add
    derivatives = []
    for rows in components.values():
        for expression in rows:
            for column in range(total):
                result = Const(0)
                for coordinate, source in sources.items():
                    if source == ("unknown", column):
                        result = _s_add(result, diff(expression, coordinate,
                            native_derivative_route=route))
                derivatives.append(result)
    return derivatives
