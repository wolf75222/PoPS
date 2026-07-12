#pragma once

/// @file
/// @brief Fail-closed native contract for residual and index-1 DAE operators.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops::runtime::program {

struct DomainShape {
  std::vector<std::size_t> extents;

  std::size_t size() const {
    if (extents.empty()) return 0;
    std::size_t n = 1;
    for (const auto extent : extents) {
      if (extent == 0 || n > std::numeric_limits<std::size_t>::max() / extent) return 0;
      n *= extent;
    }
    return n;
  }
};

struct BlockSlot {
  std::string name;
  std::size_t offset = 0;
  std::size_t width = 0;
};

struct ResidualDomain {
  DomainShape shape;
  std::vector<BlockSlot> blocks;
};

enum class LinearizationFidelity { kExact, kJvp, kApproximate };
enum class MassKind { kIdentity, kConstant, kAlgebraic };
enum class DaeIndex { kNotDae, kIndex1, kHigherIndex };
enum class ConsistentInitializationPolicy { kValidateOnly, kRequireInitializer };

enum class SupportRefusal {
  kNone,
  kInvalidDomain,
  kUnsupportedLinearization,
  kUnsupportedMass,
  kHigherIndex,
  kInconsistentInitialState,
};

struct SupportDecision {
  SupportRefusal refusal = SupportRefusal::kNone;
  std::string detail;
  explicit operator bool() const noexcept { return refusal == SupportRefusal::kNone; }
};

enum class ResidualSolveReason {
  kConverged,
  kMaximumIterations,
  kLinearizationUnavailable,
  kUnsupportedContract,
  kNonFiniteResidual,
};

struct ResidualSolveResult {
  int nonlinear_iterations = 0;
  double residual_norm = 0.0;
  ResidualSolveReason reason = ResidualSolveReason::kUnsupportedContract;
  bool converged() const noexcept { return reason == ResidualSolveReason::kConverged; }
};

struct MassDescriptor {
  MassKind kind = MassKind::kIdentity;
  /// Constant diagonal. Algebraic variables are represented by an exact zero.
  std::vector<double> diagonal;
};

inline SupportDecision validate_domain(const ResidualDomain& domain) {
  const auto n = domain.shape.size();
  if (n == 0) return {SupportRefusal::kInvalidDomain, "shape must have non-zero finite size"};
  std::size_t next = 0;
  for (std::size_t i = 0; i < domain.blocks.size(); ++i) {
    const auto& block = domain.blocks[i];
    if (block.name.empty() || block.width == 0 || block.offset != next || block.width > n - next)
      return {SupportRefusal::kInvalidDomain,
              "block slots must be named, non-empty, contiguous, and within the domain"};
    for (std::size_t j = 0; j < i; ++j)
      if (domain.blocks[j].name == block.name)
        return {SupportRefusal::kInvalidDomain, "block slot names must be unique"};
    next += block.width;
  }
  if (domain.blocks.empty() || next != n)
    return {SupportRefusal::kInvalidDomain, "block slots must cover the domain exactly"};
  return {};
}

inline SupportDecision validate_mass(const MassDescriptor& mass, std::size_t n, DaeIndex index) {
  if (index == DaeIndex::kHigherIndex)
    return {SupportRefusal::kHigherIndex, "only explicitly classified index-1 DAEs are supported"};
  if (mass.kind == MassKind::kIdentity)
    return mass.diagonal.empty()
               ? SupportDecision{}
               : SupportDecision{SupportRefusal::kUnsupportedMass,
                                 "identity mass must not carry coefficients"};
  if (mass.diagonal.size() != n)
    return {SupportRefusal::kUnsupportedMass, "constant/algebraic mass diagonal has wrong size"};
  for (const double value : mass.diagonal)
    if (!std::isfinite(value))
      return {SupportRefusal::kUnsupportedMass, "mass coefficients must be finite"};
  if (mass.kind == MassKind::kConstant) {
    for (const double value : mass.diagonal)
      if (value == 0.0)
        return {SupportRefusal::kUnsupportedMass, "constant mass must be nonsingular"};
    if (index != DaeIndex::kNotDae)
      return {SupportRefusal::kUnsupportedMass, "nonsingular mass is not a DAE"};
    return {};
  }
  bool has_differential = false, has_algebraic = false;
  for (const double value : mass.diagonal) {
    has_algebraic |= value == 0.0;
    has_differential |= value != 0.0;
  }
  if (!has_differential || !has_algebraic || index != DaeIndex::kIndex1)
    return {SupportRefusal::kUnsupportedMass,
            "algebraic mass requires mixed differential/algebraic rows and index-1 classification"};
  return {};
}

using ResidualFunction = std::function<void(const std::vector<double>&, std::vector<double>&)>;
using JvpFunction =
    std::function<void(const std::vector<double>&, const std::vector<double>&, std::vector<double>&)>;
using ConsistentInitializationFunction =
    std::function<SupportDecision(std::vector<double>&, double)>;

class ResidualOperator {
 public:
  ResidualOperator(ResidualDomain domain, MassDescriptor mass, DaeIndex index,
                   LinearizationFidelity fidelity, ResidualFunction residual,
                   JvpFunction exact_jvp = {}, JvpFunction jvp = {},
                   ConsistentInitializationPolicy initialization_policy =
                       ConsistentInitializationPolicy::kValidateOnly,
                   ConsistentInitializationFunction consistent_initializer = {})
      : domain_(std::move(domain)), mass_(std::move(mass)), index_(index), fidelity_(fidelity),
        residual_(std::move(residual)), exact_jvp_(std::move(exact_jvp)), jvp_(std::move(jvp)),
        initialization_policy_(initialization_policy),
        consistent_initializer_(std::move(consistent_initializer)) {}

  SupportDecision support() const {
    auto decision = validate_domain(domain_);
    if (!decision) return decision;
    decision = validate_mass(mass_, domain_.shape.size(), index_);
    if (!decision) return decision;
    if (!residual_)
      return {SupportRefusal::kUnsupportedLinearization, "residual evaluator is required"};
    if (fidelity_ == LinearizationFidelity::kExact && !exact_jvp_)
      return {SupportRefusal::kUnsupportedLinearization, "exact fidelity requires an exact JVP"};
    if (fidelity_ == LinearizationFidelity::kJvp && !jvp_)
      return {SupportRefusal::kUnsupportedLinearization, "JVP fidelity requires a JVP evaluator"};
    if (index_ == DaeIndex::kIndex1 &&
        initialization_policy_ == ConsistentInitializationPolicy::kRequireInitializer &&
        !consistent_initializer_)
      return {SupportRefusal::kInconsistentInitialState,
              "index-1 initialization policy requires a native initializer"};
    return {};
  }

  std::vector<double> evaluate(const std::vector<double>& state) const {
    require_supported(state);
    std::vector<double> out(state.size());
    residual_(state, out);
    validate_output(out, state.size(), "residual");
    return out;
  }

  std::vector<double> apply_jvp(const std::vector<double>& state,
                                const std::vector<double>& direction) const {
    require_supported(state);
    if (direction.size() != state.size()) throw std::invalid_argument("JVP direction has wrong size");
    std::vector<double> out(state.size());
    if (fidelity_ == LinearizationFidelity::kExact)
      exact_jvp_(state, direction, out);
    else if (fidelity_ == LinearizationFidelity::kJvp)
      jvp_(state, direction, out);
    else
      finite_difference_jvp(state, direction, out);
    validate_output(out, state.size(), "JVP");
    return out;
  }

  SupportDecision validate_consistent_initial_state(const std::vector<double>& state,
                                                    double tolerance) const {
    auto decision = support();
    if (!decision) return decision;
    if (index_ != DaeIndex::kIndex1) return {};
    if (state.size() != domain_.shape.size() || !(tolerance >= 0.0))
      return {SupportRefusal::kInconsistentInitialState, "invalid state size or tolerance"};
    std::vector<double> residual(state.size());
    residual_(state, residual);
    if (residual.size() != state.size())
      return {SupportRefusal::kInconsistentInitialState, "residual evaluator changed output size"};
    for (std::size_t i = 0; i < residual.size(); ++i)
      if (mass_.diagonal[i] == 0.0 && (!std::isfinite(residual[i]) || std::abs(residual[i]) > tolerance))
        return {SupportRefusal::kInconsistentInitialState,
                "algebraic residual exceeds consistent-initialization tolerance"};
    return {};
  }

  SupportDecision consistent_initialize(std::vector<double>& state, double tolerance) const {
    auto decision = support();
    if (!decision) return decision;
    if (index_ != DaeIndex::kIndex1) return {};
    if (!consistent_initializer_)
      return {SupportRefusal::kInconsistentInitialState,
              "no native index-1 consistent initializer is available"};
    if (state.size() != domain_.shape.size() || !(tolerance >= 0.0))
      return {SupportRefusal::kInconsistentInitialState, "invalid state size or tolerance"};
    decision = consistent_initializer_(state, tolerance);
    if (!decision) return decision;
    if (state.size() != domain_.shape.size())
      return {SupportRefusal::kInconsistentInitialState,
              "consistent initializer changed state size"};
    for (const double value : state)
      if (!std::isfinite(value))
        return {SupportRefusal::kInconsistentInitialState,
                "consistent initializer produced non-finite state"};
    return validate_consistent_initial_state(state, tolerance);
  }

 private:
  void require_supported(const std::vector<double>& state) const {
    const auto decision = support();
    if (!decision) throw std::logic_error(decision.detail);
    if (state.size() != domain_.shape.size()) throw std::invalid_argument("residual state has wrong size");
  }

  static void validate_output(const std::vector<double>& output, std::size_t expected,
                              const char* what) {
    if (output.size() != expected)
      throw std::runtime_error(std::string(what) + " evaluator changed output size");
    for (const double value : output)
      if (!std::isfinite(value)) throw std::runtime_error(std::string(what) + " produced non-finite output");
  }

  void finite_difference_jvp(const std::vector<double>& state, const std::vector<double>& direction,
                             std::vector<double>& out) const {
    double scale = 0.0;
    for (const double value : state) scale = std::max(scale, std::abs(value));
    const double eps = std::sqrt(std::numeric_limits<double>::epsilon()) * (1.0 + scale);
    auto perturbed = state;
    for (std::size_t i = 0; i < state.size(); ++i) perturbed[i] += eps * direction[i];
    std::vector<double> base(state.size()), shifted(state.size());
    residual_(state, base);
    residual_(perturbed, shifted);
    for (std::size_t i = 0; i < state.size(); ++i) out[i] = (shifted[i] - base[i]) / eps;
  }

  ResidualDomain domain_;
  MassDescriptor mass_;
  DaeIndex index_;
  LinearizationFidelity fidelity_;
  ResidualFunction residual_;
  JvpFunction exact_jvp_;
  JvpFunction jvp_;
  ConsistentInitializationPolicy initialization_policy_;
  ConsistentInitializationFunction consistent_initializer_;
};

}  // namespace pops::runtime::program
