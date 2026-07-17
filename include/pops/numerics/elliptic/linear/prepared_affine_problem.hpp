#pragma once

/// @file
/// @brief Prepared, snapshot-authenticated affine linear problems.

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/linear/pure_field_algebra.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace pops {

/// Host-side matrix-free application. The callback must overwrite every valid output cell and owns
/// any typed halo/boundary fill required before it reads input ghosts. Device work remains inside the
/// Kokkos-backed kernels called by the function; no callback is ever copied or constructed inside an
/// iteration.
using ApplyFn = std::function<void(MultiFab& out, const MultiFab& in)>;
using PreparedResourceFn = std::function<void()>;

enum class KrylovMethod : std::uint8_t { kCg, kBicgstab, kGmres, kRichardson };

enum class LinearOperatorProperty : std::uint32_t {
  kNone = 0,
  kSymmetric = 1u << 0u,
  kPositiveDefinite = 1u << 1u,
};

constexpr std::uint32_t operator_property_bit(LinearOperatorProperty property) {
  return static_cast<std::uint32_t>(property);
}

/// Authenticated mathematical properties. Positive-definite without symmetric is incoherent and is
/// refused at construction; CG requires the complete SPD certificate and is never silently replaced.
struct LinearOperatorProperties {
  std::uint32_t bits = 0;

  static constexpr LinearOperatorProperties general() { return {}; }
  static constexpr LinearOperatorProperties symmetric() {
    return {operator_property_bit(LinearOperatorProperty::kSymmetric)};
  }
  static constexpr LinearOperatorProperties symmetric_positive_definite() {
    return {operator_property_bit(LinearOperatorProperty::kSymmetric) |
            operator_property_bit(LinearOperatorProperty::kPositiveDefinite)};
  }
  constexpr bool has(LinearOperatorProperty property) const {
    return (bits & operator_property_bit(property)) != 0;
  }
  constexpr bool certifies_spd() const {
    return has(LinearOperatorProperty::kSymmetric) &&
           has(LinearOperatorProperty::kPositiveDefinite);
  }
  constexpr bool valid() const {
    constexpr std::uint32_t known =
        operator_property_bit(LinearOperatorProperty::kSymmetric) |
        operator_property_bit(LinearOperatorProperty::kPositiveDefinite);
    return (bits & ~known) == 0 && (!has(LinearOperatorProperty::kPositiveDefinite) ||
                                    has(LinearOperatorProperty::kSymmetric));
  }
};

/// Exact memory/layout requirement of one prepared solve route.
struct KrylovFootprint {
  int components = 1;
  int input_ghosts = 0;
  int restart = 0;
  bool preconditioned = false;

  friend bool operator==(const KrylovFootprint&, const KrylovFootprint&) = default;
};

/// Allocation-free identity of one operator evaluation. The 256-bit authority is emitted from the
/// canonical Program/IR identity. The remaining fields authenticate the actual runtime evaluation
/// point and topology. Binary64 values travel as exact bit patterns, never rounded text.
using OperatorFingerprint = std::array<std::uint64_t, 4>;

struct OperatorEvaluationSnapshot {
  OperatorFingerprint authority{};
  std::uint64_t revision = 0;
  std::int64_t macro_step = 0;
  std::int64_t stage_numerator = 0;
  std::int64_t stage_denominator = 1;
  std::uint64_t dt_bits = 0;
  std::uint64_t physical_time_bits = 0;
  std::uint64_t topology_revision = 0;
  OperatorFingerprint topology{};
  OperatorFingerprint resources{};

  bool valid() const {
    const bool nonzero_authority =
        authority[0] != 0 || authority[1] != 0 || authority[2] != 0 || authority[3] != 0;
    const auto any_nonzero = [](const OperatorFingerprint& fingerprint) {
      return fingerprint[0] != 0 || fingerprint[1] != 0 || fingerprint[2] != 0 ||
             fingerprint[3] != 0;
    };
    const double dt = std::bit_cast<double>(dt_bits);
    const double time = std::bit_cast<double>(physical_time_bits);
    return nonzero_authority && revision != 0 && macro_step >= 0 && stage_numerator >= 0 &&
           stage_denominator > 0 && stage_numerator <= stage_denominator && std::isfinite(dt) &&
           dt >= 0.0 && std::isfinite(time) && topology_revision != 0 && any_nonzero(topology) &&
           any_nonzero(resources);
  }
  friend bool operator==(const OperatorEvaluationSnapshot&,
                         const OperatorEvaluationSnapshot&) = default;
};

using OperatorSnapshotProbe = std::function<OperatorEvaluationSnapshot()>;

/// Trust boundary for the native operator callback. Generated PoPS applies are compiled from a
/// restricted, side-effect-free Program region and need no second snapshot probe after each matvec.
/// Extension callbacks default to verified mode, which detects a mutation performed by the callback
/// itself before the result can be consumed.
enum class OperatorApplyPurity : std::uint8_t {
  kVerifyAfterApply,
  kAuthenticatedProgram,
};

namespace detail {

struct PreparedProblemAccess;

inline std::uint64_t fingerprint_mix_word(std::uint64_t hash, std::uint64_t value) {
  constexpr std::uint64_t prime = 1099511628211ull;
  for (unsigned byte = 0; byte < 8; ++byte) {
    hash ^= (value >> (byte * 8u)) & 0xffu;
    hash *= prime;
  }
  return hash;
}

inline OperatorFingerprint fingerprint_seed() {
  return {1469598103934665603ull, 1099511628211ull, 7809847782465536322ull, 9650029242287828579ull};
}

inline void fingerprint_mix(OperatorFingerprint& hash, std::uint64_t value) {
  constexpr std::array<std::uint64_t, 4> domain_separators{
      0x9e3779b97f4a7c15ull, 0xbf58476d1ce4e5b9ull, 0x94d049bb133111ebull, 0xd6e8feb86659fd93ull};
  for (std::size_t lane = 0; lane < hash.size(); ++lane)
    hash[lane] = fingerprint_mix_word(hash[lane], value ^ domain_separators[lane]);
}

inline void fingerprint_mix(OperatorFingerprint& hash, std::string_view value) {
  fingerprint_mix(hash, static_cast<std::uint64_t>(value.size()));
  for (const unsigned char byte : value)
    fingerprint_mix(hash, static_cast<std::uint64_t>(byte));
}

inline OperatorFingerprint layout_fingerprint(const MultiFab& value) {
  OperatorFingerprint hash = fingerprint_seed();
  fingerprint_mix(hash, static_cast<std::uint64_t>(value.ncomp()));
  fingerprint_mix(hash, static_cast<std::uint64_t>(value.n_grow()));
  const auto& boxes = value.box_array().boxes();
  const auto& ranks = value.dmap().ranks();
  fingerprint_mix(hash, static_cast<std::uint64_t>(boxes.size()));
  for (std::size_t index = 0; index < boxes.size(); ++index) {
    const Box2D& box = boxes[index];
    for (int axis = 0; axis < 2; ++axis) {
      fingerprint_mix(hash, static_cast<std::uint64_t>(static_cast<std::int64_t>(box.lo[axis])));
      fingerprint_mix(hash, static_cast<std::uint64_t>(static_cast<std::int64_t>(box.hi[axis])));
    }
    fingerprint_mix(hash, static_cast<std::uint64_t>(static_cast<std::int64_t>(ranks[index])));
  }
  return hash;
}

inline void fingerprint_geometry(OperatorFingerprint& hash, const Geometry& geometry) {
  fingerprint_mix(hash, "cartesian");
  for (int axis = 0; axis < 2; ++axis) {
    fingerprint_mix(
        hash, static_cast<std::uint64_t>(static_cast<std::int64_t>(geometry.domain.lo[axis])));
    fingerprint_mix(
        hash, static_cast<std::uint64_t>(static_cast<std::int64_t>(geometry.domain.hi[axis])));
  }
  fingerprint_mix(hash, std::bit_cast<std::uint64_t>(geometry.xlo));
  fingerprint_mix(hash, std::bit_cast<std::uint64_t>(geometry.xhi));
  fingerprint_mix(hash, std::bit_cast<std::uint64_t>(geometry.ylo));
  fingerprint_mix(hash, std::bit_cast<std::uint64_t>(geometry.yhi));
}

inline void fingerprint_geometry(OperatorFingerprint& hash, const PolarGeometry& geometry) {
  fingerprint_mix(hash, "polar");
  for (int axis = 0; axis < 2; ++axis) {
    fingerprint_mix(
        hash, static_cast<std::uint64_t>(static_cast<std::int64_t>(geometry.domain.lo[axis])));
    fingerprint_mix(
        hash, static_cast<std::uint64_t>(static_cast<std::int64_t>(geometry.domain.hi[axis])));
  }
  fingerprint_mix(hash, std::bit_cast<std::uint64_t>(geometry.r_min));
  fingerprint_mix(hash, std::bit_cast<std::uint64_t>(geometry.r_max));
}

inline void fingerprint_boundary(OperatorFingerprint& hash, const BCRec& boundary) {
  fingerprint_mix(hash, static_cast<std::uint64_t>(boundary.xlo));
  fingerprint_mix(hash, static_cast<std::uint64_t>(boundary.xhi));
  fingerprint_mix(hash, static_cast<std::uint64_t>(boundary.ylo));
  fingerprint_mix(hash, static_cast<std::uint64_t>(boundary.yhi));
  const std::array<Real, 14> values{boundary.xlo_val,   boundary.xhi_val,   boundary.ylo_val,
                                    boundary.yhi_val,   boundary.xlo_alpha, boundary.xlo_beta,
                                    boundary.xhi_alpha, boundary.xhi_beta,  boundary.ylo_alpha,
                                    boundary.ylo_beta,  boundary.yhi_alpha, boundary.yhi_beta,
                                    boundary.dx,        boundary.dy};
  for (const Real value : values)
    fingerprint_mix(hash, std::bit_cast<std::uint64_t>(value));
}

}  // namespace detail

/// A fixed preconditioner prepared separately from the affine operator. Its raw callback may contain
/// an affine physical-boundary response; prepare() captures the exact zero response and apply()
/// subtracts it. The preparation hook must allocate/build all native state before the first apply;
/// apply() refuses an unbound or changed snapshot, so no lazy initialization can hide inside a
/// Krylov iteration.
class PreparedLinearPreconditioner {
 public:
  static PreparedLinearPreconditioner identity() { return PreparedLinearPreconditioner(); }

  PreparedLinearPreconditioner() = default;
  explicit PreparedLinearPreconditioner(const MultiFab& prototype, ApplyFn raw_apply,
                                        PreparedResourceFn prepare = {})
      : raw_apply_(std::move(raw_apply)),
        prepare_(std::move(prepare)),
        zero_(prototype.box_array(), prototype.dmap(), prototype.ncomp(), prototype.n_grow()),
        constant_(prototype.box_array(), prototype.dmap(), prototype.ncomp(), prototype.n_grow()),
        layout_(detail::layout_fingerprint(prototype)) {
    if (!raw_apply_)
      throw std::invalid_argument("PreparedLinearPreconditioner requires a non-empty apply");
    zero_.share_halo_cache_from(prototype);
    constant_.share_halo_cache_from(prototype);
  }

  bool is_identity() const { return !static_cast<bool>(raw_apply_); }
  bool compatible_with(const MultiFab& prototype) const {
    return is_identity() || layout_ == detail::layout_fingerprint(prototype);
  }

  void prepare(const OperatorEvaluationSnapshot& snapshot) {
    snapshot_.reset();
    if (!snapshot.valid())
      throw std::invalid_argument("PreparedLinearPreconditioner received an invalid snapshot");
    if (prepare_)
      prepare_();
    if (!is_identity()) {
      // A physical-BC preconditioner can itself be affine. Evaluate its exact zero response once
      // after all resources are materialized, then subtract it from every search-direction apply.
      // This is the preconditioner analogue of A_lin(v) = A(v) - A(0), and it also warms every
      // callback/halo resource before the iteration begins.
      detail::PreparedFieldAlgebra::zero(zero_);
      detail::PreparedFieldAlgebra::zero(constant_);
      raw_apply_(constant_, zero_);
    }
    snapshot_ = snapshot;
  }

  void apply(MultiFab& out, const MultiFab& in, const OperatorEvaluationSnapshot& snapshot) const {
    if (!snapshot_ || *snapshot_ != snapshot)
      throw std::logic_error("prepared preconditioner snapshot changed without preparation");
    if (out.shares_storage_with(in))
      throw std::invalid_argument("prepared preconditioner output must not alias its input");
    if (is_identity())
      detail::PreparedFieldAlgebra::copy(out, in);
    else {
      raw_apply_(out, in);
      detail::PreparedFieldAlgebra::axpy(out, Real(-1), constant_);
    }
  }

 private:
  ApplyFn raw_apply_{};
  PreparedResourceFn prepare_{};
  MultiFab zero_{};
  MultiFab constant_{};
  OperatorFingerprint layout_{};
  std::optional<OperatorEvaluationSnapshot> snapshot_{};
};

/// Owns one affine operator evaluation A(u), its exact constant c=A(0), and the frozen resources
/// captured for that evaluation. Fields/callbacks are created by the constructor; prepare() is the
/// explicit resource-materialization boundary where frozen coefficients, halo/MPI capacities and
/// preconditioners may be warmed before it evaluates A(0). No lazy work may escape into iteration.
class PreparedAffineLinearProblem {
 public:
  PreparedAffineLinearProblem(
      const MultiFab& prototype, ApplyFn raw_apply, PreparedLinearPreconditioner preconditioner,
      LinearOperatorProperties properties, KrylovFootprint footprint,
      OperatorSnapshotProbe snapshot_probe, PreparedResourceFn freeze_resources = {},
      OperatorApplyPurity apply_purity = OperatorApplyPurity::kVerifyAfterApply)
      : raw_apply_(std::move(raw_apply)),
        preconditioner_(std::move(preconditioner)),
        properties_(properties),
        footprint_(footprint),
        snapshot_probe_(std::move(snapshot_probe)),
        freeze_resources_(std::move(freeze_resources)),
        apply_purity_(apply_purity),
        zero_(prototype.box_array(), prototype.dmap(), prototype.ncomp(), footprint.input_ghosts),
        constant_(prototype.box_array(), prototype.dmap(), prototype.ncomp(),
                  footprint.input_ghosts),
        layout_(detail::layout_fingerprint(prototype)) {
    zero_.share_halo_cache_from(prototype);
    constant_.share_halo_cache_from(prototype);
    if (!raw_apply_)
      throw std::invalid_argument("PreparedAffineLinearProblem requires a raw affine apply");
    if (!properties_.valid())
      throw std::invalid_argument("PreparedAffineLinearProblem received incoherent properties");
    if (footprint_.components != prototype.ncomp() || footprint_.components < 1 ||
        footprint_.input_ghosts < 0 || footprint_.input_ghosts != prototype.n_grow() ||
        footprint_.restart < 0)
      throw std::invalid_argument("PreparedAffineLinearProblem footprint disagrees with prototype");
    if (footprint_.preconditioned != !preconditioner_.is_identity())
      throw std::invalid_argument(
          "PreparedAffineLinearProblem footprint disagrees with preconditioner presence");
    if (!preconditioner_.compatible_with(prototype))
      throw std::invalid_argument(
          "PreparedAffineLinearProblem preconditioner layout disagrees with prototype");
    if (!snapshot_probe_)
      throw std::invalid_argument("PreparedAffineLinearProblem requires a snapshot probe");
  }

  void prepare(const OperatorEvaluationSnapshot& snapshot) {
    snapshot_.reset();
    if (!snapshot.valid())
      throw std::invalid_argument("PreparedAffineLinearProblem received an invalid snapshot");
    if (snapshot_probe_() != snapshot)
      throw std::logic_error("operator snapshot changed before preparation");
    if (freeze_resources_)
      freeze_resources_();
    preconditioner_.prepare(snapshot);
    detail::PreparedFieldAlgebra::zero(zero_);
    raw_apply_(constant_, zero_);  // exact c = A(0) after every resource is materialized
    if (snapshot_probe_() != snapshot)
      throw std::logic_error("operator snapshot changed during preparation");
    snapshot_ = snapshot;
  }

  bool prepared() const { return snapshot_.has_value(); }
  const OperatorEvaluationSnapshot& snapshot() const {
    require_current();
    return *snapshot_;
  }
  const OperatorFingerprint& layout_fingerprint() const { return layout_; }
  const LinearOperatorProperties& properties() const { return properties_; }
  const KrylovFootprint& footprint() const { return footprint_; }
  const MultiFab& constant_term() const {
    require_current();
    return constant_;
  }
  bool has_preconditioner() const { return !preconditioner_.is_identity(); }

  void effective_rhs(MultiFab& out, const MultiFab& rhs) const {
    require_vector_field(rhs, "PreparedAffineLinearProblem::effective_rhs(rhs)");
    require_operator_field(out, "PreparedAffineLinearProblem::effective_rhs(out)");
    require_distinct_storage(out, rhs, "PreparedAffineLinearProblem::effective_rhs");
    effective_rhs_prepared_(out, rhs);
  }

  /// A_lin(v) = A(v) - A(0), valid for search directions even when boundaries/sources make A affine.
  void apply_linear(MultiFab& out, const MultiFab& direction) const {
    require_operator_field(direction, "PreparedAffineLinearProblem::apply_linear(direction)");
    require_operator_field(out, "PreparedAffineLinearProblem::apply_linear(out)");
    require_distinct_storage(out, direction, "PreparedAffineLinearProblem::apply_linear");
    apply_linear_prepared_(out, direction);
  }

  /// Scientific residual R(u) = b - A(u), never a preconditioned or Arnoldi estimate.
  void true_residual(MultiFab& out, const MultiFab& rhs, const MultiFab& iterate) const {
    require_vector_field(rhs, "PreparedAffineLinearProblem::true_residual(rhs)");
    require_operator_field(iterate, "PreparedAffineLinearProblem::true_residual(iterate)");
    require_operator_field(out, "PreparedAffineLinearProblem::true_residual(out)");
    require_distinct_storage(out, rhs, "PreparedAffineLinearProblem::true_residual(out, rhs)");
    require_distinct_storage(out, iterate,
                             "PreparedAffineLinearProblem::true_residual(out, iterate)");
    true_residual_prepared_(out, rhs, iterate);
  }

  void apply_preconditioner(MultiFab& out, const MultiFab& in) const {
    require_operator_field(in, "PreparedAffineLinearProblem::apply_preconditioner(in)");
    require_operator_field(out, "PreparedAffineLinearProblem::apply_preconditioner(out)");
    require_distinct_storage(out, in, "PreparedAffineLinearProblem::apply_preconditioner");
    apply_preconditioner_prepared_(out, in);
  }

  /// The delivered prepared metric is one global L2 product over every component and rank. Keeping
  /// it on the problem (rather than inside individual algorithms) gives every method and report one
  /// authority and leaves a narrow metric-provider seam for a future weighted/composite route.
  Real inner_product(const MultiFab& left, const MultiFab& right) const {
    require_vector_field(left, "PreparedAffineLinearProblem::inner_product(left)");
    require_vector_field(right, "PreparedAffineLinearProblem::inner_product(right)");
    return inner_product_prepared_(left, right);
  }

  Real residual_norm(const MultiFab& value) const {
    require_vector_field(value, "PreparedAffineLinearProblem::residual_norm");
    return residual_norm_prepared_(value);
  }

  void require_vector_field(const MultiFab& value, const char* where) const {
    if (!PureFieldAlgebra::same_vector_space(value, zero_))
      throw std::invalid_argument(std::string(where) + ": incompatible vector space");
  }

  void require_operator_field(const MultiFab& value, const char* where) const {
    require_vector_field(value, where);
    if (value.n_grow() != footprint_.input_ghosts)
      throw std::invalid_argument(std::string(where) + ": incompatible ghost footprint");
  }

  void require_current() const {
    if (!snapshot_)
      throw std::logic_error("PreparedAffineLinearProblem is not prepared");
    if (snapshot_probe_() != *snapshot_)
      throw std::logic_error("operator snapshot mutated after preparation");
  }

  static void require_distinct_storage(const MultiFab& out, const MultiFab& in,
                                       const char* where) {
    if (out.shares_storage_with(in))
      throw std::invalid_argument(std::string(where) + ": output aliases an input field");
  }

 private:
  friend struct detail::PreparedProblemAccess;

  void verify_external_apply_() const {
    if (apply_purity_ == OperatorApplyPurity::kVerifyAfterApply)
      require_current();
  }

  void effective_rhs_prepared_(MultiFab& out, const MultiFab& rhs) const {
    require_current();
    detail::PreparedFieldAlgebra::lincomb(out, Real(1), rhs, Real(-1), constant_);
  }

  void apply_linear_prepared_(MultiFab& out, const MultiFab& direction) const {
    require_current();
    raw_apply_(out, direction);
    verify_external_apply_();
    detail::PreparedFieldAlgebra::axpy(out, Real(-1), constant_);
  }

  void true_residual_prepared_(MultiFab& out, const MultiFab& rhs, const MultiFab& iterate) const {
    require_current();
    raw_apply_(out, iterate);
    verify_external_apply_();
    detail::PreparedFieldAlgebra::lincomb(out, Real(1), rhs, Real(-1), out);
  }

  void apply_preconditioner_prepared_(MultiFab& out, const MultiFab& in) const {
    require_current();
    preconditioner_.apply(out, in, *snapshot_);
    verify_external_apply_();
  }

  Real inner_product_prepared_(const MultiFab& left, const MultiFab& right) const {
    return detail::PreparedFieldAlgebra::dot(left, right);
  }

  Real local_inner_product_prepared_(const MultiFab& left, const MultiFab& right) const {
    return detail::PreparedFieldAlgebra::local_dot(left, right);
  }

  Real residual_norm_prepared_(const MultiFab& value) const {
    return detail::PreparedFieldAlgebra::norm(value);
  }

  ApplyFn raw_apply_;
  mutable PreparedLinearPreconditioner preconditioner_;
  LinearOperatorProperties properties_{};
  KrylovFootprint footprint_{};
  OperatorSnapshotProbe snapshot_probe_;
  PreparedResourceFn freeze_resources_;
  OperatorApplyPurity apply_purity_ = OperatorApplyPurity::kVerifyAfterApply;
  MultiFab zero_;
  MultiFab constant_;
  OperatorFingerprint layout_{};
  std::optional<OperatorEvaluationSnapshot> snapshot_{};
};

namespace detail {

/// Private, allocation-free access used only after solve_prepared_affine has authenticated the
/// caller-owned iterate/RHS and KrylovWorkspace against the prepared problem. Public direct calls on
/// PreparedAffineLinearProblem retain their complete defensive layout validation.
struct PreparedProblemAccess {
  static void effective_rhs(const PreparedAffineLinearProblem& problem, MultiFab& out,
                            const MultiFab& rhs) {
    problem.effective_rhs_prepared_(out, rhs);
  }
  static void apply_linear(const PreparedAffineLinearProblem& problem, MultiFab& out,
                           const MultiFab& direction) {
    problem.apply_linear_prepared_(out, direction);
  }
  static void true_residual(const PreparedAffineLinearProblem& problem, MultiFab& out,
                            const MultiFab& rhs, const MultiFab& iterate) {
    problem.true_residual_prepared_(out, rhs, iterate);
  }
  static void apply_preconditioner(const PreparedAffineLinearProblem& problem, MultiFab& out,
                                   const MultiFab& in) {
    problem.apply_preconditioner_prepared_(out, in);
  }
  static Real inner_product(const PreparedAffineLinearProblem& problem, const MultiFab& left,
                            const MultiFab& right) {
    return problem.inner_product_prepared_(left, right);
  }
  static Real local_inner_product(const PreparedAffineLinearProblem& problem, const MultiFab& left,
                                  const MultiFab& right) {
    return problem.local_inner_product_prepared_(left, right);
  }
  static Real residual_norm(const PreparedAffineLinearProblem& problem, const MultiFab& value) {
    return problem.residual_norm_prepared_(value);
  }
};

}  // namespace detail

}  // namespace pops
