/// @file
/// @brief Typed analytic-expression IR and compact data-only device interpreter.
///
/// Analytic expressions are authored as a tree on the host, validated once, and lowered to a
/// postfix program stored in Kokkos SharedSpace.  A kernel captures only AnalyticProgramView: two
/// device-valid pointers and small scalar metadata.  There is no callback, allocation, virtual
/// dispatch, or Python interaction in the per-cell evaluator.

#pragma once

#include <pops/core/foundation/allocator.hpp>
#include <pops/core/foundation/types.hpp>

#include <Kokkos_MathematicalFunctions.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::analytic {

/// Stable operations understood by the analytic-expression VM.
enum class AnalyticOp : std::uint8_t {
  Constant = 0,
  X = 1,
  Y = 2,
  Add = 3,
  Sub = 4,
  Mul = 5,
  Div = 6,
  Pow = 7,
  Neg = 8,
  Sqrt = 9,
  Abs = 10,
  Sin = 11,
  Cos = 12,
  Exp = 13,
  Log = 14,
  Atan2 = 15,
  Hypot = 16,
  Min = 17,
  Max = 18,
  Lt = 19,
  Le = 20,
  Gt = 21,
  Ge = 22,
  Eq = 23,
  Ne = 24,
  And = 25,
  Or = 26,
  Not = 27,
  Select = 28,
  Between = 29,
  Input = 30,
};

enum class AnalyticValueType : std::uint8_t { Scalar = 0, Predicate = 1 };

inline constexpr std::size_t kAnalyticMaxNodes = 4096;
inline constexpr std::size_t kAnalyticMaxDepth = 64;
inline constexpr std::size_t kAnalyticMaxStack = 64;

/// Limits are part of validation, not hints.  Values above the device VM hard limits are rejected.
struct AnalyticLimits {
  std::size_t max_nodes = kAnalyticMaxNodes;
  std::size_t max_depth = kAnalyticMaxDepth;
  std::size_t max_stack = kAnalyticMaxStack;
};

/// Canonical host tree: an operation, an optional finite literal, and ordered arguments.
struct AnalyticNode {
  AnalyticOp op = AnalyticOp::Constant;
  Real literal = Real(0);
  std::vector<AnalyticNode> args;

  static AnalyticNode constant(Real value) {
    return AnalyticNode{AnalyticOp::Constant, value, {}};
  }
  static AnalyticNode x() { return AnalyticNode{AnalyticOp::X, Real(0), {}}; }
  static AnalyticNode y() { return AnalyticNode{AnalyticOp::Y, Real(0), {}}; }
  static AnalyticNode apply(AnalyticOp operation, std::vector<AnalyticNode> arguments) {
    return AnalyticNode{operation, Real(0), std::move(arguments)};
  }
};

/// Public host-side postfix token.  This is the direct lowering seam for bindings that already
/// traverse the canonical data tree.  literal must be finite for Constant and exactly zero otherwise.
struct AnalyticToken {
  AnalyticOp op = AnalyticOp::Constant;
  Real literal = Real(0);
};

/// Compact device instruction.  Only Constant reads operand (an index in the literal table).
struct AnalyticInstruction {
  std::uint32_t operand = 0;
  AnalyticOp op = AnalyticOp::Constant;
};

static_assert(std::is_trivially_copyable_v<AnalyticInstruction>);
static_assert(sizeof(AnalyticInstruction) <= 8,
              "analytic instruction must remain compact (opcode plus literal index)");

inline const char* op_name(AnalyticOp op) {
  switch (op) {
    case AnalyticOp::Constant:
      return "constant";
    case AnalyticOp::X:
      return "x";
    case AnalyticOp::Y:
      return "y";
    case AnalyticOp::Add:
      return "add";
    case AnalyticOp::Sub:
      return "sub";
    case AnalyticOp::Mul:
      return "mul";
    case AnalyticOp::Div:
      return "div";
    case AnalyticOp::Pow:
      return "pow";
    case AnalyticOp::Neg:
      return "neg";
    case AnalyticOp::Sqrt:
      return "sqrt";
    case AnalyticOp::Abs:
      return "abs";
    case AnalyticOp::Sin:
      return "sin";
    case AnalyticOp::Cos:
      return "cos";
    case AnalyticOp::Exp:
      return "exp";
    case AnalyticOp::Log:
      return "log";
    case AnalyticOp::Atan2:
      return "atan2";
    case AnalyticOp::Hypot:
      return "hypot";
    case AnalyticOp::Min:
      return "minimum";
    case AnalyticOp::Max:
      return "maximum";
    case AnalyticOp::Lt:
      return "lt";
    case AnalyticOp::Le:
      return "le";
    case AnalyticOp::Gt:
      return "gt";
    case AnalyticOp::Ge:
      return "ge";
    case AnalyticOp::Eq:
      return "eq";
    case AnalyticOp::Ne:
      return "ne";
    case AnalyticOp::And:
      return "and";
    case AnalyticOp::Or:
      return "or";
    case AnalyticOp::Not:
      return "not";
    case AnalyticOp::Select:
      return "where";
    case AnalyticOp::Between:
      return "between";
    case AnalyticOp::Input:
      return "input";
  }
  return "unknown";
}

/// Strict inverse for canonical schema operation names.  A coordinate node is resolved by the
/// binding layer from its typed axis to X or Y before this function is called.
inline AnalyticOp analytic_op_from_name(std::string_view name) {
  if (name == "constant")
    return AnalyticOp::Constant;
  if (name == "x")
    return AnalyticOp::X;
  if (name == "y")
    return AnalyticOp::Y;
  if (name == "add")
    return AnalyticOp::Add;
  if (name == "sub")
    return AnalyticOp::Sub;
  if (name == "mul")
    return AnalyticOp::Mul;
  if (name == "div")
    return AnalyticOp::Div;
  if (name == "pow")
    return AnalyticOp::Pow;
  if (name == "neg")
    return AnalyticOp::Neg;
  if (name == "sqrt")
    return AnalyticOp::Sqrt;
  if (name == "abs")
    return AnalyticOp::Abs;
  if (name == "sin")
    return AnalyticOp::Sin;
  if (name == "cos")
    return AnalyticOp::Cos;
  if (name == "exp")
    return AnalyticOp::Exp;
  if (name == "log")
    return AnalyticOp::Log;
  if (name == "atan2")
    return AnalyticOp::Atan2;
  if (name == "hypot")
    return AnalyticOp::Hypot;
  if (name == "minimum")
    return AnalyticOp::Min;
  if (name == "maximum")
    return AnalyticOp::Max;
  if (name == "lt")
    return AnalyticOp::Lt;
  if (name == "le")
    return AnalyticOp::Le;
  if (name == "gt")
    return AnalyticOp::Gt;
  if (name == "ge")
    return AnalyticOp::Ge;
  if (name == "eq")
    return AnalyticOp::Eq;
  if (name == "ne")
    return AnalyticOp::Ne;
  if (name == "and")
    return AnalyticOp::And;
  if (name == "or")
    return AnalyticOp::Or;
  if (name == "not")
    return AnalyticOp::Not;
  if (name == "where")
    return AnalyticOp::Select;
  if (name == "between")
    return AnalyticOp::Between;
  if (name == "input")
    return AnalyticOp::Input;
  throw std::invalid_argument("analytic expression: unknown operation '" + std::string(name) +
                              "'");
}

inline bool is_known(AnalyticOp op) {
  return static_cast<std::uint8_t>(op) <= static_cast<std::uint8_t>(AnalyticOp::Input);
}

inline int arity(AnalyticOp op) {
  switch (op) {
    case AnalyticOp::Constant:
    case AnalyticOp::X:
    case AnalyticOp::Y:
    case AnalyticOp::Input:
      return 0;
    case AnalyticOp::Neg:
    case AnalyticOp::Sqrt:
    case AnalyticOp::Abs:
    case AnalyticOp::Sin:
    case AnalyticOp::Cos:
    case AnalyticOp::Exp:
    case AnalyticOp::Log:
    case AnalyticOp::Not:
      return 1;
    case AnalyticOp::Add:
    case AnalyticOp::Sub:
    case AnalyticOp::Mul:
    case AnalyticOp::Div:
    case AnalyticOp::Pow:
    case AnalyticOp::Atan2:
    case AnalyticOp::Hypot:
    case AnalyticOp::Min:
    case AnalyticOp::Max:
    case AnalyticOp::Lt:
    case AnalyticOp::Le:
    case AnalyticOp::Gt:
    case AnalyticOp::Ge:
    case AnalyticOp::Eq:
    case AnalyticOp::Ne:
    case AnalyticOp::And:
    case AnalyticOp::Or:
      return 2;
    case AnalyticOp::Select:
    case AnalyticOp::Between:
      return 3;
  }
  return -1;
}

/// One value produced by the device VM together with its mathematical-domain validity.
///
/// Keeping validity separate from the floating-point payload prevents operations such as fmin,
/// comparisons and selection from accidentally turning an invalid intermediate into an apparently
/// finite result.  Select deliberately observes only the chosen branch, so guarded expressions such
/// as where(x > 0, log(x), 0) retain their expected piecewise semantics.
struct AnalyticEvaluation {
  Real value = Real(0);
  bool valid = false;
};

static_assert(std::is_trivially_copyable_v<AnalyticEvaluation>);

/// Non-owning POD captured by value in Kokkos Serial/OpenMP/Cuda kernels.  The pointers refer to
/// Kokkos SharedSpace allocations owned by the corresponding AnalyticProgram.
struct AnalyticProgramView {
  const AnalyticInstruction* instructions = nullptr;
  const Real* literals = nullptr;
  std::uint32_t instruction_count = 0;
  std::uint8_t required_stack = 0;
  AnalyticValueType result_type = AnalyticValueType::Scalar;

  /// Evaluate at Cartesian coordinates (x,y), preserving validity of every intermediate.
  /// The program structure has already passed host validation.
  POPS_HD AnalyticEvaluation eval_checked(Real x, Real y, const Real* inputs = nullptr,
                                          std::uint8_t input_count = 0) const {
    // Keep the per-thread VM stack compact on accelerators. AnalyticEvaluation is naturally padded
    // to 16 bytes (Real + bool), whereas these two uninitialized arrays need only 9 bytes per slot.
    // The validated postfix program writes every active slot before reading it.
    Real values[kAnalyticMaxStack];
    std::uint8_t validity[kAnalyticMaxStack];
    std::size_t sp = 0;
    for (std::uint32_t pc = 0; pc < instruction_count; ++pc) {
      const AnalyticInstruction instruction = instructions[pc];
      switch (instruction.op) {
        case AnalyticOp::Constant:
          assert(sp < kAnalyticMaxStack);
          values[sp] = literals[instruction.operand];
          validity[sp++] = std::uint8_t{1};
          break;
        case AnalyticOp::X:
          assert(sp < kAnalyticMaxStack);
          values[sp] = x;
          validity[sp++] = Kokkos::isfinite(x) ? std::uint8_t{1} : std::uint8_t{0};
          break;
        case AnalyticOp::Y:
          assert(sp < kAnalyticMaxStack);
          values[sp] = y;
          validity[sp++] = Kokkos::isfinite(y) ? std::uint8_t{1} : std::uint8_t{0};
          break;
        case AnalyticOp::Input: {
          assert(sp < kAnalyticMaxStack);
          const std::uint32_t slot = instruction.operand;
          const bool ok = inputs != nullptr && slot < input_count;
          values[sp] = ok ? inputs[slot] : std::numeric_limits<Real>::quiet_NaN();
          validity[sp++] = ok && Kokkos::isfinite(values[sp - 1]) ? std::uint8_t{1}
                                                                  : std::uint8_t{0};
        } break;
        case AnalyticOp::Add: {
          const std::size_t right = --sp;
          const std::size_t left = sp - 1;
          values[left] += values[right];
          validity[left] = validity[left] && validity[right] && Kokkos::isfinite(values[left]);
        } break;
        case AnalyticOp::Sub: {
          const std::size_t right = --sp;
          const std::size_t left = sp - 1;
          values[left] -= values[right];
          validity[left] = validity[left] && validity[right] && Kokkos::isfinite(values[left]);
        } break;
        case AnalyticOp::Mul: {
          const std::size_t right = --sp;
          const std::size_t left = sp - 1;
          values[left] *= values[right];
          validity[left] = validity[left] && validity[right] && Kokkos::isfinite(values[left]);
        } break;
        case AnalyticOp::Div: {
          const std::size_t right = --sp;
          const std::size_t left = sp - 1;
          values[left] /= values[right];
          validity[left] = validity[left] && validity[right] && Kokkos::isfinite(values[left]);
        } break;
        case AnalyticOp::Pow: {
          const std::size_t right = --sp;
          const std::size_t left = sp - 1;
          values[left] = Kokkos::pow(values[left], values[right]);
          validity[left] = validity[left] && validity[right] && Kokkos::isfinite(values[left]);
        } break;
        case AnalyticOp::Neg: {
          const std::size_t value = sp - 1;
          values[value] = -values[value];
          validity[value] = validity[value] && Kokkos::isfinite(values[value]);
        } break;
        case AnalyticOp::Sqrt: {
          const std::size_t value = sp - 1;
          values[value] = Kokkos::sqrt(values[value]);
          validity[value] = validity[value] && Kokkos::isfinite(values[value]);
        } break;
        case AnalyticOp::Abs: {
          const std::size_t value = sp - 1;
          values[value] = Kokkos::fabs(values[value]);
          validity[value] = validity[value] && Kokkos::isfinite(values[value]);
        } break;
        case AnalyticOp::Sin: {
          const std::size_t value = sp - 1;
          values[value] = Kokkos::sin(values[value]);
          validity[value] = validity[value] && Kokkos::isfinite(values[value]);
        } break;
        case AnalyticOp::Cos: {
          const std::size_t value = sp - 1;
          values[value] = Kokkos::cos(values[value]);
          validity[value] = validity[value] && Kokkos::isfinite(values[value]);
        } break;
        case AnalyticOp::Exp: {
          const std::size_t value = sp - 1;
          values[value] = Kokkos::exp(values[value]);
          validity[value] = validity[value] && Kokkos::isfinite(values[value]);
        } break;
        case AnalyticOp::Log: {
          const std::size_t value = sp - 1;
          values[value] = Kokkos::log(values[value]);
          validity[value] = validity[value] && Kokkos::isfinite(values[value]);
        } break;
        case AnalyticOp::Atan2: {
          const std::size_t right = --sp;
          const std::size_t left = sp - 1;
          values[left] = Kokkos::atan2(values[left], values[right]);
          validity[left] = validity[left] && validity[right] && Kokkos::isfinite(values[left]);
        } break;
        case AnalyticOp::Hypot: {
          const std::size_t right = --sp;
          const std::size_t left = sp - 1;
          values[left] = Kokkos::hypot(values[left], values[right]);
          validity[left] = validity[left] && validity[right] && Kokkos::isfinite(values[left]);
        } break;
        case AnalyticOp::Min: {
          const std::size_t right = --sp;
          const std::size_t left = sp - 1;
          values[left] = Kokkos::fmin(values[left], values[right]);
          validity[left] = validity[left] && validity[right] && Kokkos::isfinite(values[left]);
        } break;
        case AnalyticOp::Max: {
          const std::size_t right = --sp;
          const std::size_t left = sp - 1;
          values[left] = Kokkos::fmax(values[left], values[right]);
          validity[left] = validity[left] && validity[right] && Kokkos::isfinite(values[left]);
        } break;
        case AnalyticOp::Lt: {
          const std::size_t right = --sp;
          const std::size_t left = sp - 1;
          values[left] = values[left] < values[right] ? Real(1) : Real(0);
          validity[left] = validity[left] && validity[right];
        } break;
        case AnalyticOp::Le: {
          const std::size_t right = --sp;
          const std::size_t left = sp - 1;
          values[left] = values[left] <= values[right] ? Real(1) : Real(0);
          validity[left] = validity[left] && validity[right];
        } break;
        case AnalyticOp::Gt: {
          const std::size_t right = --sp;
          const std::size_t left = sp - 1;
          values[left] = values[left] > values[right] ? Real(1) : Real(0);
          validity[left] = validity[left] && validity[right];
        } break;
        case AnalyticOp::Ge: {
          const std::size_t right = --sp;
          const std::size_t left = sp - 1;
          values[left] = values[left] >= values[right] ? Real(1) : Real(0);
          validity[left] = validity[left] && validity[right];
        } break;
        case AnalyticOp::Eq: {
          const std::size_t right = --sp;
          const std::size_t left = sp - 1;
          values[left] = values[left] == values[right] ? Real(1) : Real(0);
          validity[left] = validity[left] && validity[right];
        } break;
        case AnalyticOp::Ne: {
          const std::size_t right = --sp;
          const std::size_t left = sp - 1;
          values[left] = values[left] != values[right] ? Real(1) : Real(0);
          validity[left] = validity[left] && validity[right];
        } break;
        case AnalyticOp::And: {
          const std::size_t right = --sp;
          const std::size_t left = sp - 1;
          values[left] = values[left] != Real(0) && values[right] != Real(0) ? Real(1) : Real(0);
          validity[left] = validity[left] && validity[right];
        } break;
        case AnalyticOp::Or: {
          const std::size_t right = --sp;
          const std::size_t left = sp - 1;
          values[left] = values[left] != Real(0) || values[right] != Real(0) ? Real(1) : Real(0);
          validity[left] = validity[left] && validity[right];
        } break;
        case AnalyticOp::Not: {
          const std::size_t value = sp - 1;
          values[value] = values[value] == Real(0) ? Real(1) : Real(0);
        } break;
        case AnalyticOp::Select: {
          const std::size_t otherwise = --sp;
          const std::size_t selected = --sp;
          const std::size_t condition = --sp;
          const std::size_t chosen = values[condition] != Real(0) ? selected : otherwise;
          const Real chosen_value = values[chosen];
          const std::uint8_t chosen_validity = validity[chosen];
          const std::uint8_t condition_validity = validity[condition];
          values[sp] = chosen_value;
          validity[sp++] = condition_validity && chosen_validity;
        } break;
        case AnalyticOp::Between: {
          const std::size_t upper = --sp;
          const std::size_t lower = --sp;
          const std::size_t value = --sp;
          const Real result =
              values[value] >= values[lower] && values[value] <= values[upper] ? Real(1) : Real(0);
          const std::uint8_t result_validity =
              validity[value] && validity[lower] && validity[upper];
          values[sp] = result;
          validity[sp++] = result_validity;
        } break;
      }
    }
    assert(sp == 1);
    return {values[0], validity[0] != std::uint8_t{0}};
  }

  /// Scalar convenience seam. Invalid evaluation is represented by NaN so every existing finite
  /// preflight remains fail-closed even when an operation could otherwise mask the invalid payload.
  POPS_HD Real eval(Real x, Real y, const Real* inputs = nullptr,
                    std::uint8_t input_count = 0) const {
    const AnalyticEvaluation result = eval_checked(x, y, inputs, input_count);
    return result.valid ? result.value : std::numeric_limits<Real>::quiet_NaN();
  }

  POPS_HD bool eval_predicate(Real x, Real y) const {
    assert(result_type == AnalyticValueType::Predicate);
    const AnalyticEvaluation result = eval_checked(x, y);
    assert(result.valid);
    return result.valid && result.value != Real(0);
  }

};

static_assert(std::is_trivially_copyable_v<AnalyticProgramView>);

/// Immutable owning program.  Its vectors use the project's Kokkos SharedSpace allocator, so the
/// raw pointers exported by view() are valid in native CPU and GPU kernels.
class AnalyticProgram {
 public:
  using InstructionStorage = std::vector<AnalyticInstruction, fab_allocator<AnalyticInstruction>>;
  using LiteralStorage = std::vector<Real, fab_allocator<Real>>;

  AnalyticProgram() = default;

  [[nodiscard]] bool empty() const noexcept { return instructions_.empty(); }
  [[nodiscard]] std::size_t instruction_count() const noexcept { return instructions_.size(); }
  [[nodiscard]] std::size_t literal_count() const noexcept { return literals_.size(); }
  [[nodiscard]] std::size_t required_stack() const noexcept { return required_stack_; }
  [[nodiscard]] AnalyticValueType result_type() const noexcept { return result_type_; }

  [[nodiscard]] AnalyticProgramView view() const noexcept {
    return AnalyticProgramView{instructions_.data(), literals_.data(),
                               static_cast<std::uint32_t>(instructions_.size()),
                               static_cast<std::uint8_t>(required_stack_), result_type_};
  }

  [[nodiscard]] Real evaluate(Real x, Real y) const {
    if (empty())
      throw std::logic_error("analytic expression: cannot evaluate an empty program");
    return view().eval(x, y);
  }

 private:
  AnalyticProgram(InstructionStorage instructions, LiteralStorage literals,
                  std::size_t required_stack, AnalyticValueType result_type)
      : instructions_(std::move(instructions)),
        literals_(std::move(literals)),
        required_stack_(required_stack),
        result_type_(result_type) {}

  InstructionStorage instructions_;
  LiteralStorage literals_;
  std::size_t required_stack_ = 0;
  AnalyticValueType result_type_ = AnalyticValueType::Scalar;

  friend AnalyticProgram compile_analytic_postfix(const std::vector<AnalyticToken>&,
                                                  AnalyticLimits);
};

namespace detail {

inline void reject(const std::string& message) {
  throw std::invalid_argument("analytic expression: " + message);
}

inline void validate_limits(const AnalyticLimits& limits) {
  if (limits.max_nodes == 0 || limits.max_nodes > kAnalyticMaxNodes)
    reject("max_nodes must be in [1," + std::to_string(kAnalyticMaxNodes) + "]");
  if (limits.max_depth == 0 || limits.max_depth > kAnalyticMaxDepth)
    reject("max_depth must be in [1," + std::to_string(kAnalyticMaxDepth) + "]");
  if (limits.max_stack == 0 || limits.max_stack > kAnalyticMaxStack)
    reject("max_stack must be in [1," + std::to_string(kAnalyticMaxStack) + "]");
}

inline bool comparison(AnalyticOp op) {
  return op >= AnalyticOp::Lt && op <= AnalyticOp::Ne;
}

inline void require_type(AnalyticValueType actual, AnalyticValueType expected, AnalyticOp op,
                         std::size_t token_index) {
  if (actual != expected)
    reject("operator '" + std::string(op_name(op)) + "' at token " +
           std::to_string(token_index) + " requires " +
           (expected == AnalyticValueType::Scalar ? "scalar" : "predicate") + " arguments");
}

struct TypedStackEntry {
  AnalyticValueType type = AnalyticValueType::Scalar;
  std::size_t depth = 0;
  bool literal_constant = false;
  Real literal_value = Real(0);
};

inline void flatten_node(const AnalyticNode& node, const AnalyticLimits& limits, std::size_t depth,
                         std::size_t& node_count, std::vector<AnalyticToken>& tokens) {
  if (depth > limits.max_depth)
    reject("tree depth exceeds max_depth=" + std::to_string(limits.max_depth));
  if (++node_count > limits.max_nodes)
    reject("tree node count exceeds max_nodes=" + std::to_string(limits.max_nodes));
  if (!is_known(node.op))
    reject("unknown opcode " + std::to_string(static_cast<unsigned>(node.op)));
  const int expected = arity(node.op);
  if (node.args.size() != static_cast<std::size_t>(expected))
    reject("operator '" + std::string(op_name(node.op)) + "' requires " +
           std::to_string(expected) + " arguments, got " + std::to_string(node.args.size()));
  for (const AnalyticNode& argument : node.args)
    flatten_node(argument, limits, depth + 1, node_count, tokens);
  tokens.push_back(AnalyticToken{node.op, node.literal});
}

}  // namespace detail

/// Validate and compile an already flattened postfix expression.
inline AnalyticProgram compile_analytic_postfix(const std::vector<AnalyticToken>& tokens,
                                                AnalyticLimits limits = {}) {
  detail::validate_limits(limits);
  if (tokens.empty())
    detail::reject("postfix program is empty");
  if (tokens.size() > limits.max_nodes)
    detail::reject("node count " + std::to_string(tokens.size()) + " exceeds max_nodes=" +
                   std::to_string(limits.max_nodes));

  std::array<detail::TypedStackEntry, kAnalyticMaxStack> stack{};
  std::size_t sp = 0;
  std::size_t maximum_stack = 0;

  AnalyticProgram::InstructionStorage instructions(tokens.size());
  std::size_t literal_count = 0;
  for (const AnalyticToken& token : tokens)
    if (token.op == AnalyticOp::Constant || token.op == AnalyticOp::Input)
      ++literal_count;
  AnalyticProgram::LiteralStorage literals(literal_count);
  std::size_t next_literal = 0;

  for (std::size_t index = 0; index < tokens.size(); ++index) {
    const AnalyticToken token = tokens[index];
    if (!is_known(token.op))
      detail::reject("unknown opcode " + std::to_string(static_cast<unsigned>(token.op)) +
                     " at token " + std::to_string(index));
    if (token.op == AnalyticOp::Constant) {
      if (!std::isfinite(token.literal))
        detail::reject("constant at token " + std::to_string(index) + " must be finite");
    } else if (token.op == AnalyticOp::Input) {
      if (!std::isfinite(token.literal) || token.literal < Real(0) ||
          token.literal != std::floor(token.literal) ||
          token.literal > static_cast<Real>(std::numeric_limits<std::uint32_t>::max()))
        detail::reject("input at token " + std::to_string(index) +
                       " must carry a non-negative integer slot");
    } else if (token.literal != Real(0)) {
      detail::reject("non-constant token " + std::to_string(index) + " ('" + op_name(token.op) +
                     "') must not carry a literal");
    }

    const int needed = arity(token.op);
    if (sp < static_cast<std::size_t>(needed))
      detail::reject("postfix stack underflow at token " + std::to_string(index) + " ('" +
                     op_name(token.op) + "')");

    AnalyticValueType result = AnalyticValueType::Scalar;
    std::size_t result_depth = 1;
    bool result_is_literal = token.op == AnalyticOp::Constant;
    Real result_literal = result_is_literal ? token.literal : Real(0);
    if (needed == 0) {
      result = AnalyticValueType::Scalar;
    } else if (needed == 1) {
      const detail::TypedStackEntry argument = stack[--sp];
      result_depth = argument.depth + 1;
      if (token.op == AnalyticOp::Not) {
        detail::require_type(argument.type, AnalyticValueType::Predicate, token.op, index);
        result = AnalyticValueType::Predicate;
      } else {
        detail::require_type(argument.type, AnalyticValueType::Scalar, token.op, index);
        result = AnalyticValueType::Scalar;
      }
    } else if (needed == 2) {
      const detail::TypedStackEntry right = stack[--sp];
      const detail::TypedStackEntry left = stack[--sp];
      result_depth = std::max(left.depth, right.depth) + 1;
      if (token.op == AnalyticOp::And || token.op == AnalyticOp::Or) {
        detail::require_type(left.type, AnalyticValueType::Predicate, token.op, index);
        detail::require_type(right.type, AnalyticValueType::Predicate, token.op, index);
        result = AnalyticValueType::Predicate;
      } else {
        detail::require_type(left.type, AnalyticValueType::Scalar, token.op, index);
        detail::require_type(right.type, AnalyticValueType::Scalar, token.op, index);
        if (token.op == AnalyticOp::Pow && !right.literal_constant)
          detail::reject("operator 'pow' at token " + std::to_string(index) +
                         " requires a literal exponent");
        result = detail::comparison(token.op) ? AnalyticValueType::Predicate
                                              : AnalyticValueType::Scalar;
      }
    } else if (token.op == AnalyticOp::Select) {
      const detail::TypedStackEntry otherwise = stack[--sp];
      const detail::TypedStackEntry selected = stack[--sp];
      const detail::TypedStackEntry condition = stack[--sp];
      result_depth = std::max(condition.depth, std::max(selected.depth, otherwise.depth)) + 1;
      detail::require_type(condition.type, AnalyticValueType::Predicate, token.op, index);
      if (selected.type != otherwise.type)
        detail::reject("select at token " + std::to_string(index) +
                       " requires branches with the same type");
      result = selected.type;
    } else {
      const detail::TypedStackEntry upper = stack[--sp];
      const detail::TypedStackEntry lower = stack[--sp];
      const detail::TypedStackEntry value = stack[--sp];
      result_depth = std::max(value.depth, std::max(lower.depth, upper.depth)) + 1;
      detail::require_type(value.type, AnalyticValueType::Scalar, token.op, index);
      detail::require_type(lower.type, AnalyticValueType::Scalar, token.op, index);
      detail::require_type(upper.type, AnalyticValueType::Scalar, token.op, index);
      if (lower.literal_constant && upper.literal_constant &&
          lower.literal_value > upper.literal_value)
        detail::reject("operator 'between' at token " + std::to_string(index) +
                       " requires lower <= upper");
      result = AnalyticValueType::Predicate;
    }

    if (result_depth > limits.max_depth)
      detail::reject("expression depth exceeds max_depth=" + std::to_string(limits.max_depth));
    if (sp >= limits.max_stack)
      detail::reject("postfix stack exceeds max_stack=" + std::to_string(limits.max_stack));
    stack[sp++] =
        detail::TypedStackEntry{result, result_depth, result_is_literal, result_literal};
    maximum_stack = std::max(maximum_stack, sp);

    AnalyticInstruction instruction{};
    instruction.op = token.op;
    if (token.op == AnalyticOp::Constant || token.op == AnalyticOp::Input) {
      instruction.operand = static_cast<std::uint32_t>(next_literal);
      literals[next_literal++] = token.literal;
      if (token.op == AnalyticOp::Input)
        instruction.operand = static_cast<std::uint32_t>(token.literal);
    }
    instructions[index] = instruction;
  }

  if (sp != 1)
    detail::reject("postfix program must leave exactly one result, got " + std::to_string(sp));

  return AnalyticProgram(std::move(instructions), std::move(literals), maximum_stack,
                         stack[0].type);
}

/// Validate and compile the canonical host tree.
inline AnalyticProgram compile_analytic_expression(const AnalyticNode& root,
                                                   AnalyticLimits limits = {}) {
  detail::validate_limits(limits);
  std::vector<AnalyticToken> tokens;
  tokens.reserve(std::min<std::size_t>(limits.max_nodes, 64));
  std::size_t node_count = 0;
  detail::flatten_node(root, limits, 1, node_count, tokens);
  return compile_analytic_postfix(tokens, limits);
}

}  // namespace pops::analytic
