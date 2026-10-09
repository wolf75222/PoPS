"""Lower explicit normalized polynomial arithmetic; no closure is selected here."""
from __future__ import annotations

import math


class _ArithmeticEmitter:
    def __init__(self, order, indices, nodes, capacity, *, path=False, endpoint_integral=False):
        self.order, self.indices, self.path, self.nodes = order, indices, path, nodes
        self.capacity = capacity
        self.endpoint_integral = endpoint_integral
        self.lines, self.memo = [], {}

    def emit(self, key):
        if type(key) is not int or not 0 <= key < len(self.nodes):
            raise ValueError("arithmetic reference is outside its exact graph")
        if key in self.memo:
            return self.memo[key]
        node = self.nodes[key]
        if type(node) is not dict or set(node) != {"op", "args", "degree", "polynomial"}:
            raise ValueError("arithmetic expression requires the exact contract @1 fields")
        op, args = node["op"], node["args"]
        if type(op) is not str:
            raise TypeError("arithmetic operation must be an exact string")
        if self.endpoint_integral and op in ("normalized", "raw", "density", "second_moment_norm"):
            raise ValueError("analytic integral requires explicitly qualified left/right endpoint inputs")
        if type(args) is not tuple or type(node["degree"]) is not int or type(node["polynomial"]) is not bool:
            raise TypeError("malformed arithmetic expression types")
        child_count = {"add": 2, "multiply": 2, "derivative": 1, "fma": 3,
                       "sqrt": 1, "abs": 1, "power": 1, "divide_literal": 1}.get(op, len(args) if op == "compensated_sum" else 0)
        if any(type(ref) is not int or not 0 <= ref < key for ref in args[:child_count]):
            raise ValueError("arithmetic graph must be acyclic and refer only to prior nodes")
        poly, degree = False, 0
        if op == "literal":
            if len(args) != 1 or type(args[0]) not in (int, float) or not math.isfinite(args[0]):
                raise ValueError("arithmetic literal must be finite")
            expression = f"Real({args[0]!r})"
        elif op in ("normalized", "raw", "direction", "left_raw", "right_raw"):
            limit = 2 if op == "direction" else len(self.indices)
            if len(args) != 1 or type(args[0]) is not int or not 0 <= args[0] < limit:
                raise ValueError("arithmetic input is outside its exact support")
            slot = args[0]
            if op == "direction":
                expression = f"g[{slot}]"
            elif op in ("left_raw", "right_raw"):
                if not self.endpoint_integral:
                    raise ValueError("endpoint-pair coordinates require the analytic integral context")
                expression = f"raw_{'left' if op == 'left_raw' else 'right'}[{slot}]"
            elif op == "raw":
                if self.path:
                    raise ValueError("raw endpoint input is unavailable along the normalized path")
                expression = f"raw[{slot}]"
            else:
                p, q = self.indices[slot]
                index = q * (self.order + 1) - q * (q - 1) // 2 + p
                if self.path:
                    poly, degree = True, 1
                    expression = f"affine(left.normalized[{index}], right.normalized[{index}])"
                else:
                    expression = f"state.normalized[{index}]"
        elif op in ("density", "second_moment_norm"):
            if args or self.path:
                raise ValueError("endpoint density/norm input is unavailable along the polynomial path")
            expression = ("state.density" if op == "density" else
                          "pops::moments::raw_second_moment_norm(state, g[0], g[1])")
        else:
            arity = {"add": 2, "multiply": 2, "derivative": 1, "fma": 3, "sqrt": 1, "abs": 1}
            if op == "divide_literal":
                if len(args) != 2 or type(args[1]) not in (int, float) or not math.isfinite(args[1]) or args[1] == 0:
                    raise ValueError("division requires a finite nonzero literal")
                a, poly, degree = self.emit(args[0])
                expression = (f"scale({a}, Real(1) / Real({args[1]!r}))" if poly else
                              f"({a} / Real({args[1]!r}))")
            elif op == "power":
                if len(args) != 2 or type(args[1]) is not int or not 0 <= args[1] <= 2**31 - 1:
                    raise ValueError("power requires a nonnegative exact exponent")
                a, poly, d = self.emit(args[0]); degree = d * args[1]
                if poly:
                    expression = f"power({a}, {args[1]})"
                else:
                    expression = f"pops::moments::power({a}, {args[1]})"
            else:
                if op not in (*arity, "compensated_sum") or (op in arity and len(args) != arity[op]) or not args:
                    raise ValueError("unknown arithmetic operation or arity")
                values = [self.emit(arg) for arg in args]
                poly = any(value[1] for value in values)
                degree = (sum(value[2] for value in values) if op == "multiply" else max(value[2] for value in values))
                refs = [value[0] for value in values]
                if op in ("add", "multiply"):
                    if poly:
                        if op == "multiply" and values[0][1] != values[1][1]:
                            scalar, polynomial = (refs[1], refs[0]) if values[0][1] else (refs[0], refs[1])
                            expression = f"scale({polynomial}, {scalar})"
                        else:
                            refs = [ref if value[1] else f"Poly({ref})" for ref, value in zip(refs, values)]
                            expression = f"{op}({refs[0]}, {refs[1]})"
                    else:
                        expression = f"({refs[0]} {'+' if op == 'add' else '*'} {refs[1]})"
                elif op == "derivative":
                    if not poly:
                        raise ValueError("derivative requires polynomial arithmetic")
                    degree = max(0, degree - 1)
                    expression = f"derivative({refs[0]})"
                else:
                    if poly:
                        raise ValueError("scalar operation received polynomial arithmetic")
                    if op == "compensated_sum":
                        name = f"a{len(self.memo)}"
                        self.lines += [f"    Real {name} = Real(0), {name}_comp = Real(0);"]
                        for k, ref in enumerate(refs):
                            self.lines += [f"    const Real {name}_t{k} = {ref} - {name}_comp;",
                                           f"    const Real {name}_n{k} = {name} + {name}_t{k};",
                                           f"    {name}_comp = ({name}_n{k} - {name}) - {name}_t{k};",
                                           f"    {name} = {name}_n{k};"]
                        result = name, False, 0
                        self.memo[key] = result
                        if node["degree"] != 0 or node["polynomial"]:
                            raise ValueError("arithmetic metadata disagrees with its operations")
                        return result
                    expression = f"std::{op}({', '.join(refs)})"
        if degree != node["degree"] or poly != node["polynomial"] or degree > self.capacity:
            raise ValueError("arithmetic degree/type disagrees with its operations or exceeds capacity")
        name = f"a{len(self.memo)}"
        self.lines.append(f"    const {'Poly' if poly else 'Real'} {name} = {expression};")
        result = name, poly, degree
        self.memo[key] = result
        return result


def emit_moment_path_kernel(plan: dict, name: str) -> list[str]:
    """Compile the complete authored arithmetic graph, with exact support checks."""
    if type(plan) is not dict or set(plan) != {"schema_version", "order", "polynomial_degree", "indices", "nodes", "flux", "integrands", "factors", "speed", "integral_mode", "integral"}:
        raise ValueError("normalized polynomial path requires complete contract @1")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise ValueError("unsupported normalized polynomial path contract")
    order, indices = plan["order"], plan["indices"]
    if type(order) is not int or order < 2 or type(indices) is not tuple:
        raise ValueError("normalized path requires a complete Cartesian covariance basis")
    if (not indices or any(type(index) is not tuple or len(index) != 2
            or any(type(i) is not int or i < 0 for i in index) for index in indices)
            or max(map(sum, indices)) != order or len(set(indices)) != len(indices)
            or len(indices) != math.comb(order + 2, 2)):
        raise ValueError("normalized path basis must cover each exact monomial once")
    canonical = tuple((p, q) for q in range(order + 1) for p in range(order + 1 - q))
    n = len(indices)
    if any(type(plan[key]) is not tuple or len(plan[key]) != n for key in ("flux", "integrands", "factors")):
        raise ValueError("normalized path outputs must cover every basis component")
    mode = plan["integral_mode"]
    if mode not in ("density_weighted_polynomial", "analytic_endpoint"):
        raise ValueError("unknown normalized path integral realization")
    if ((mode == "density_weighted_polynomial" and plan["integral"] is not None)
            or mode == "analytic_endpoint" and (type(plan["integral"]) is not tuple or len(plan["integral"]) != n)):
        raise ValueError("integral realization requires its exact authored support")
    if any(type(factor) not in (int, float) or not math.isfinite(factor) for factor in plan["factors"]):
        raise ValueError("normalized path factors must be finite literals")
    if not name.isidentifier():
        raise ValueError("kernel name must be an identifier")
    if type(plan["nodes"]) is not tuple:
        raise TypeError("arithmetic graph must be an immutable node tuple")
    capacity = plan["polynomial_degree"]
    if type(capacity) is not int or not 0 <= capacity < 2**31 - 1:
        raise ValueError("polynomial capacity must fit the native coefficient index type")
    if any(type(node) is not dict or type(node.get("degree")) is not int
           or node["degree"] < 0 or type(node.get("polynomial")) is not bool for node in plan["nodes"]):
        raise ValueError("malformed polynomial degree metadata")
    if capacity != max((node["degree"] for node in plan["nodes"] if node["polynomial"]), default=0):
        raise ValueError("polynomial capacity must equal the degree inferred from the complete graph")
    endpoint = _ArithmeticEmitter(order, indices, plan["nodes"], capacity)
    speed, poly, _ = endpoint.emit(plan["speed"])
    if poly:
        raise ValueError("speed must be scalar")
    speed_lines = list(endpoint.lines)
    used_nodes = set(endpoint.memo)
    endpoint = _ArithmeticEmitter(order, indices, plan["nodes"], capacity)
    flux = [endpoint.emit(value) for value in plan["flux"]]
    if any(value[1] for value in flux):
        raise ValueError("endpoint flux must be scalar")
    path = _ArithmeticEmitter(order, indices, plan["nodes"], capacity, path=True)
    integrands = [path.emit(value) for value in plan["integrands"]]
    used_nodes.update(endpoint.memo)
    used_nodes.update(path.memo)
    lines = [f"struct {name} {{", "  using Real = pops::Real;", "  using Direction = std::array<Real, 2>;",
             f"  using Recovered = pops::moments::NormalizedRawMoments<{order}>;",
             f"  using Poly = pops::moments::Polynomial<{capacity}>;", "  template<class State>",
             "  POPS_HD static Real canonical_component(const State& input, int k) {",
             "    constexpr int slots[] = {" + ", ".join(str(indices.index(index)) for index in canonical) + "};",
             "    return input[slots[k]];", "  }", "  template<class State>",
             "  POPS_HD static pops::PathStatus recover(const State& input, Recovered& state) {", f"    Real raw[{n}];"]
    lines += [f"    raw[{k}] = input[{indices.index(index)}];" for k, index in enumerate(canonical)]
    lines += [f"    return pops::moments::recover_raw_moments<{order}>(raw, state);", "  }", "  template<class State>",
              "  POPS_HD static pops::PathStatus admissibility(const State& input) {", "    Recovered state; return recover(input, state);", "  }",
              "  template<class State>", "  POPS_HD static Real speed_bound(const State& raw, const Recovered& state, const Direction& g) {", "    (void)raw; (void)state; (void)g;", *speed_lines, f"    return {speed};", "  }",
              "  template<class State>", f"  POPS_HD pops::PathFluxResult<{n}> path_directional_flux(const State& raw, const Direction& g) const {{",
              f"    pops::PathFluxResult<{n}> result;", "    Recovered state; result.status = recover(raw, state);",
              "    if (!result.succeeded()) return result;", "    for (auto value : g) if (!std::isfinite(value)) {",
              "      result.status = pops::PathStatus::NonFiniteDirection; return result;", "    }",
              "    if (g[0] == Real(0) && g[1] == Real(0)) return result;", *endpoint.lines]
    for value, _, _ in flux:
        lines.append(f"    if (!std::isfinite({value})) {{ result.status = pops::PathStatus::NonFiniteResult; return result; }}")
    for slot, (value, _, _) in enumerate(flux):
        lines.append(f"    result.flux.values[{slot}] = {value};")
    lines += ["    return result;", "  }", "  POPS_HD static void path_polynomials(const Recovered& left, const Recovered& right, const Direction& g,",
              f"      Poly (&integrands)[{n}], Real (&factors)[{n}]) {{", "    using namespace pops::moments;",
              "    (void)left; (void)right; (void)g;"]
    if capacity > 0:
        lines.append("    const auto affine = [](Real first, Real second) { Poly p; p.c[0] = first; p.c[1] = second - first; return p; };")
    lines += path.lines
    for slot, ((value, poly, _), factor) in enumerate(zip(integrands, plan["factors"], strict=True)):
        lines += [f"    integrands[{slot}] = {value if poly else f'Poly({value})'};", f"    factors[{slot}] = Real({factor!r});"]
    lines += ["  }", "  template<class State>", f"  POPS_HD pops::PathIntegralResult<{n}> path_integral(const State& left, const State& right, const Direction& g) const {{"]
    if mode == "density_weighted_polynomial":
        lines += [f"    return pops::moments::integrate_normalized_moment_path<{order}{", " + str(capacity) if capacity != order else ""}>(left, right, g, *this);"]
    else:
        analytic = _ArithmeticEmitter(order, indices, plan["nodes"], capacity, endpoint_integral=True)
        values = [analytic.emit(value) for value in plan["integral"]]
        used_nodes.update(analytic.memo)
        if any(value[1] for value in values):
            raise ValueError("analytic endpoint integral must be scalar")
        lines += [f"    pops::PathIntegralResult<{n}> result;", "    Recovered first, second;",
                  "    result.status = recover(left, first); if (!result.succeeded()) { result.input_side = 0; return result; }",
                  "    result.status = recover(right, second); if (!result.succeeded()) { result.input_side = 1; return result; }",
                  "    for (auto value : g) if (!std::isfinite(value)) { result.status = pops::PathStatus::NonFiniteDirection; return result; }",
                  "    const Real speed_first = speed_bound(left, first, g), speed_second = speed_bound(right, second, g);",
                  "    if (!std::isfinite(speed_first) || !std::isfinite(speed_second) || speed_first < Real(0) || speed_second < Real(0)) {",
                  "      result.status = pops::PathStatus::NonFiniteResult; return result; }",
                  "    result.speed_bound = speed_first > speed_second ? speed_first : speed_second;",
                  "    bool identical = true, reverse = false;",
                  f"    for (int k = 0; k < {n}; ++k) if (canonical_component(left, k) != canonical_component(right, k)) {{",
                  "      identical = false; reverse = canonical_component(left, k) < canonical_component(right, k); break; }",
                  "    if (identical || (g[0] == Real(0) && g[1] == Real(0))) return result;",
                  "    const auto& raw_left = reverse ? right : left;", "    const auto& raw_right = reverse ? left : right;", *analytic.lines]
        for value, _, _ in values:
            lines.append(f"    if (!std::isfinite({value})) {{ result.status = pops::PathStatus::NonFiniteResult; return result; }}")
        for slot, (value, _, _) in enumerate(values):
            lines.append(f"    result.integral[{slot}] = reverse ? -{value} : {value};")
        lines.append("    return result;")
    lines += ["  }", "};"]
    if used_nodes != set(range(len(plan["nodes"]))):
        raise ValueError("arithmetic graph contains unreachable operations")
    return lines
