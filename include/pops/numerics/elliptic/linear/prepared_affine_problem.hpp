#pragma once

/// @file
/// @brief Prepared, snapshot-authenticated affine linear problems.

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/linear/pure_field_algebra.hpp>
#include <pops/parallel/comm.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace pops {

class PreparedAffineLinearProblem;
namespace detail {
struct PreparedProblemAccess;
}

/// Host-side matrix-free application. The callback must overwrite every valid output cell and owns
/// any typed halo/boundary fill required before it reads input ghosts. Device work remains inside the
/// Kokkos-backed kernels called by the function; no callback is ever copied or constructed inside an
/// iteration. A callback that enters MPI must execute the same collective trace on every rank and
/// must not leave rank-locally before that trace is complete. PoPS can make an exception uniform
/// after the callback returns; it cannot repair a collective trace that the callback already split.
using ApplyFn = std::function<void(MultiFab& out, const MultiFab& in)>;
using PreparedResourceFn = std::function<void()>;

enum class KrylovMethod : std::uint8_t { kCg, kBicgstab, kGmres, kRichardson };

enum class LinearOperatorProperty : std::uint32_t {
  kNone = 0,
  kSymmetric = 1u << 0u,
  kPositiveDefinite = 1u << 1u,
  kPositiveDefiniteOnNullspaceComplement = 1u << 2u,
};

constexpr std::uint32_t operator_property_bit(LinearOperatorProperty property) {
  return static_cast<std::uint32_t>(property);
}

/// Authenticated mathematical properties. Global positive definiteness and positive definiteness
/// on a declared nullspace complement are mutually exclusive, both require symmetry, and CG
/// requires the certificate coherent with the problem's explicit nullspace policy.
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
  static constexpr LinearOperatorProperties symmetric_positive_definite_on_nullspace_complement() {
    return {operator_property_bit(LinearOperatorProperty::kSymmetric) |
            operator_property_bit(LinearOperatorProperty::kPositiveDefiniteOnNullspaceComplement)};
  }
  constexpr bool has(LinearOperatorProperty property) const {
    return (bits & operator_property_bit(property)) != 0;
  }
  constexpr bool certifies_spd() const {
    return has(LinearOperatorProperty::kSymmetric) &&
           has(LinearOperatorProperty::kPositiveDefinite);
  }
  constexpr bool certifies_cg(bool declared_nullspace) const {
    return declared_nullspace
               ? has(LinearOperatorProperty::kSymmetric) &&
                     has(LinearOperatorProperty::kPositiveDefiniteOnNullspaceComplement)
               : certifies_spd();
  }
  constexpr bool valid() const {
    constexpr std::uint32_t known =
        operator_property_bit(LinearOperatorProperty::kSymmetric) |
        operator_property_bit(LinearOperatorProperty::kPositiveDefinite) |
        operator_property_bit(LinearOperatorProperty::kPositiveDefiniteOnNullspaceComplement);
    const bool positive_definite = has(LinearOperatorProperty::kPositiveDefinite);
    const bool complement_positive =
        has(LinearOperatorProperty::kPositiveDefiniteOnNullspaceComplement);
    return (bits & ~known) == 0 && !(positive_definite && complement_positive) &&
           (!(positive_definite || complement_positive) || has(LinearOperatorProperty::kSymmetric));
  }
};

/// Required nullspace contract of a prepared problem. `preserving(plan)` is an authenticated
/// install-time certificate that both A_lin and the installed preconditioner map the declared
/// gauge-fixed subspace into itself. The generic algorithms therefore need no method-specific
/// projection branch or collective in their hot loops. Preparation still validates the complete
/// FieldNullspacePlan against the concrete layout; a missing/incomplete gauge fails before solve.
class PreparedNullspacePolicy {
 public:
  static PreparedNullspacePolicy nonsingular() { return PreparedNullspacePolicy(); }

  static PreparedNullspacePolicy preserving(FieldNullspacePlan plan, int first_level = 0) {
    if (plan.empty())
      throw std::invalid_argument(
          "a preserving prepared nullspace policy requires a non-empty FieldNullspacePlan");
    if (first_level < 0)
      throw std::invalid_argument("prepared nullspace first level must be non-negative");
    PreparedNullspacePolicy result;
    result.plan_ = std::move(plan);
    result.first_level_ = first_level;
    result.singular_ = true;
    return result;
  }

  bool singular() const { return singular_; }

  /// Public/direct preparation deliberately always revalidates. It is not a collective cache
  /// boundary: callers that invoke this policy themselves may not have completed the fixed
  /// PreparedAffineLinearProblem consensus preflight, so a rank-local early return could split
  /// the Gram reduction below.
  void prepare(const MultiFab& layout) {
    prepared_ = false;
    if (!singular_) {
      moments_.clear();
      prepared_ = true;
      return;
    }
    if (plan_.identity.empty() || plan_.layout_identity.empty())
      throw std::invalid_argument(
          "prepared nullspace policy requires exact plan and layout identities");
    if (plan_.gauges.size() != plan_.bases.size())
      throw std::invalid_argument(
          "prepared nullspace policy must gauge every declared basis exactly once");
    for (const FieldGaugeConstraint& gauge : plan_.gauges)
      if (!std::isfinite(static_cast<double>(gauge.value)))
        throw std::invalid_argument("prepared nullspace gauge values must be finite");
    validate_field_nullspace_basis(std::vector<const MultiFab*>{&layout}, plan_, first_level_);
    moments_.assign(plan_.bases.size() * 2u, 0.0);
    prepared_ = true;
  }

  void require_compatible(const MultiFab& normalized_rhs) const {
    require_prepared();
    if (!singular_)
      return;
    detail::require_field_nullspace_compatible_prevalidated(normalized_rhs, plan_, first_level_,
                                                            moments_.data(), moments_.size());
  }

  void apply_gauge(MultiFab& iterate) const {
    require_prepared();
    if (!singular_)
      return;
    detail::apply_field_gauge_prevalidated(iterate, plan_, first_level_, moments_.data(),
                                           moments_.size());
  }

 private:
  friend class PreparedAffineLinearProblem;
  friend struct detail::PreparedProblemAccess;

  PreparedNullspacePolicy() = default;

  /// This route is private to PreparedAffineLinearProblem. Its caller has just established the
  /// exact fixed collective contract, including the immutable plan, layout, prepared bit, and
  /// persistent moment capacity on every rank. Only that consensus authorizes reusing the Gram
  /// certificate; public prepare() above remains deliberately uncached.
  void prepare_after_collective_preflight_(const MultiFab& layout) {
    if (!prepared_)
      prepare(layout);
  }

  /// A failed collective preparation stage may have completed the local certificate on only a
  /// subset of ranks. The owner calls this on every rank after its failure reduction so the next
  /// attempt starts from one uniform, re-preparable state.
  void invalidate_collective_certificate_() noexcept { prepared_ = false; }

  void require_prepared() const {
    if (!prepared_)
      throw std::logic_error("prepared nullspace policy was used before problem preparation");
  }

  FieldNullspacePlan plan_{};
  std::array<std::uint64_t, 4> plan_fingerprint_{};
  mutable std::vector<double> moments_{};
  int first_level_ = 0;
  bool singular_ = false;
  bool prepared_ = false;
};

/// Scale-safe physical reference used by the authored stopping criterion.  The field returned with
/// this value is divided by `reference_norm` only for the one nullspace-compatibility check; the
/// iterative equation chooses a separate scale from the actual warm-start residual.
struct PreparedEquationReference {
  Real reference_norm = Real(0);
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

/// Rank-local observation of the current evaluation identity. A probe must not issue MPI
/// collectives: PreparedAffineLinearProblem owns the identically ordered consensus operation and
/// converts a probe failure on any one rank into the same exception on every rank.
using OperatorSnapshotProbe = std::function<OperatorEvaluationSnapshot()>;

/// Trust boundary for the native operator callback. Generated PoPS applies are compiled from a
/// restricted, side-effect-free Program region and need no second snapshot probe or collective
/// after each matvec. Extension callbacks default to verified mode, whose rank-symmetric gate
/// detects a mutation performed by the callback itself before the result can be consumed.
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

inline std::array<char, 19 * sizeof(std::uint64_t)> snapshot_consensus_payload(
    const OperatorEvaluationSnapshot& snapshot) {
  std::array<std::uint64_t, 19> words{};
  std::size_t index = 0;
  const auto append_fingerprint = [&words, &index](const OperatorFingerprint& fingerprint) {
    for (const std::uint64_t word : fingerprint)
      words[index++] = word;
  };
  append_fingerprint(snapshot.authority);
  words[index++] = snapshot.revision;
  words[index++] = std::bit_cast<std::uint64_t>(snapshot.macro_step);
  words[index++] = std::bit_cast<std::uint64_t>(snapshot.stage_numerator);
  words[index++] = std::bit_cast<std::uint64_t>(snapshot.stage_denominator);
  words[index++] = snapshot.dt_bits;
  words[index++] = snapshot.physical_time_bits;
  words[index++] = snapshot.topology_revision;
  append_fingerprint(snapshot.topology);
  append_fingerprint(snapshot.resources);

  std::array<char, sizeof(words)> payload{};
  std::memcpy(payload.data(), words.data(), payload.size());
  return payload;
}

/// Fixed-capacity, allocation-free consensus payload used at public Krylov boundaries. Bytewise
/// min/max is intentional: it preserves exact binary64 bits and detects disagreement without
/// interpreting unsigned fingerprints as signed MPI integers.
struct KrylovCollectivePayload {
  static constexpr std::size_t kCapacity = 1024;
  static constexpr std::size_t kUsableCapacity = kCapacity - 1u;
  static constexpr std::size_t kFingerprintBytes = 4u * sizeof(std::uint64_t);
  static constexpr std::size_t kSnapshotBytes = 19u * sizeof(std::uint64_t);

  // The largest current sequence is generic_krylov's solve preflight: prepared-problem state
  // (including its observed snapshot), workspace state, controls, two field contracts, and the
  // alias bit. Keeping this arithmetic here makes the fixed capacity a compile-time contract
  // instead of a rank-local overflow path. Update it with any new payload append sequence.
  static constexpr std::size_t kPreparedProblemContractBytes =
      sizeof(std::uint32_t) + 3u * sizeof(int) + 4u * sizeof(std::uint8_t) + kSnapshotBytes +
      kFingerprintBytes + 2u * sizeof(std::uint8_t) + sizeof(std::uint64_t) + kFingerprintBytes +
      sizeof(std::uint8_t) + sizeof(std::uint64_t) + 2u * kFingerprintBytes + sizeof(std::uint8_t) +
      kSnapshotBytes;
  static constexpr std::size_t kPreparedProblemAccessBytes =
      kPreparedProblemContractBytes + sizeof(std::uint8_t) + kSnapshotBytes;
  static constexpr std::size_t kWorkspaceStateBytes = sizeof(std::uint8_t) + 3u * sizeof(int) +
                                                      sizeof(std::uint8_t) + kFingerprintBytes +
                                                      sizeof(std::uint8_t) + kSnapshotBytes;
  static constexpr std::size_t kControlsBytes =
      sizeof(std::uint8_t) + 3u * sizeof(std::uint64_t) + 2u * sizeof(int);
  static constexpr std::size_t kFieldContractBytes = kFingerprintBytes + 2u * sizeof(int);
  static constexpr std::size_t kMaximumKnownPayloadBytes =
      kPreparedProblemAccessBytes + kWorkspaceStateBytes + kControlsBytes +
      2u * kFieldContractBytes + sizeof(std::uint8_t);
  static_assert(kMaximumKnownPayloadBytes <= kUsableCapacity);

  std::array<char, kCapacity> bytes{};
  std::size_t used = 0;

  template <class T>
  void append(const T& value) noexcept {
    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(sizeof(T) <= kUsableCapacity);
    // The last byte is a collective overflow witness and is never payload storage.  A future
    // schema extension that forgets to update kMaximumKnownPayloadBytes therefore fails uniformly
    // after the same min/max reductions instead of terminating or writing past the fixed buffer.
    if (bytes.back() != 0 || used > kUsableCapacity - sizeof(T)) {
      bytes.back() = 1;
      return;
    }
    std::memcpy(bytes.data() + used, &value, sizeof(T));
    used += sizeof(T);
  }

  void append(const OperatorFingerprint& value) noexcept {
    for (const std::uint64_t word : value)
      append(word);
  }

  void append(const OperatorEvaluationSnapshot& value) noexcept {
    append(value.authority);
    append(value.revision);
    append(value.macro_step);
    append(value.stage_numerator);
    append(value.stage_denominator);
    append(value.dt_bits);
    append(value.physical_time_bits);
    append(value.topology_revision);
    append(value.topology);
    append(value.resources);
  }
};

inline bool collective_payload_agrees(const KrylovCollectivePayload& local) {
  auto minimum = local.bytes;
  auto maximum = local.bytes;
  all_reduce_min_inplace(minimum.data(), minimum.size());
  all_reduce_max_inplace(maximum.data(), maximum.size());
  if (maximum.back() != 0)
    throw std::logic_error(
        "prepared Krylov collective payload exceeded its fixed internal capacity");
  return minimum == maximum;
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
  friend class PreparedAffineLinearProblem;

  void invalidate_collective_preparation_() noexcept { snapshot_.reset(); }

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
      PreparedNullspacePolicy nullspace_policy, OperatorSnapshotProbe snapshot_probe,
      PreparedResourceFn freeze_resources = {},
      OperatorApplyPurity apply_purity = OperatorApplyPurity::kVerifyAfterApply)
      : raw_apply_(std::move(raw_apply)),
        preconditioner_(std::move(preconditioner)),
        properties_(properties),
        footprint_(footprint),
        nullspace_policy_(std::move(nullspace_policy)),
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
    if (nullspace_policy_.singular() && properties_.has(LinearOperatorProperty::kPositiveDefinite))
      throw std::invalid_argument(
          "a prepared singular operator cannot carry a global positive-definite certificate");
    if (!nullspace_policy_.singular() &&
        properties_.has(LinearOperatorProperty::kPositiveDefiniteOnNullspaceComplement))
      throw std::invalid_argument(
          "a nullspace-complement certificate requires a prepared nullspace policy");
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
    nullspace_policy_.plan_fingerprint_ = compute_nullspace_plan_fingerprint_();
  }

  void prepare(const OperatorEvaluationSnapshot& snapshot) {
    require_collective_prepare_contract_();
    snapshot_.reset();
    require_collective_prepare_snapshot_(snapshot);
    run_collective_prepare_stage_(PrepareStage::kFreeze, [&] {
      if (freeze_resources_)
        freeze_resources_();
    });
    require_collective_snapshot_match_(snapshot, PrepareStage::kFreeze);
    prepare_nullspace_collectively_(snapshot);
    try {
      run_collective_prepare_stage_(PrepareStage::kPreconditioner,
                                    [&] { preconditioner_.prepare(snapshot); });
    } catch (...) {
      // The stage failure reduction has already made this path uniform. A callback may nevertheless
      // have completed on only a subset of ranks, so discard every locally published snapshot before
      // exposing the common exception and keep the next prepare() retry collective-safe.
      preconditioner_.invalidate_collective_preparation_();
      throw;
    }
    require_collective_snapshot_match_(snapshot, PrepareStage::kPreconditioner);
    run_collective_prepare_stage_(PrepareStage::kOperatorConstant, [&] {
      detail::PreparedFieldAlgebra::zero(zero_);
      raw_apply_(constant_, zero_);  // exact c = A(0) after every resource is materialized
    });
    require_collective_snapshot_match_(snapshot, PrepareStage::kOperatorConstant);
    snapshot_ = snapshot;
  }

  bool prepared() const { return snapshot_.has_value(); }
  const OperatorEvaluationSnapshot& snapshot() const {
    require_prepared_local_("PreparedAffineLinearProblem::snapshot");
    return *snapshot_;
  }
  const OperatorFingerprint& layout_fingerprint() const { return layout_; }
  const LinearOperatorProperties& properties() const { return properties_; }
  const KrylovFootprint& footprint() const { return footprint_; }
  const MultiFab& constant_term() const {
    require_prepared_local_("PreparedAffineLinearProblem::constant_term");
    return constant_;
  }
  bool has_preconditioner() const { return !preconditioner_.is_identity(); }
  bool has_nullspace() const { return nullspace_policy_.singular(); }

  void require_nullspace_compatible(const MultiFab& normalized_rhs) const {
    require_current();
    require_collective_arguments_(vector_field_failure_(normalized_rhs),
                                  "PreparedAffineLinearProblem::require_nullspace_compatible");
    nullspace_policy_.require_compatible(normalized_rhs);
  }

  void apply_nullspace_gauge(MultiFab& iterate) const {
    require_current();
    require_collective_arguments_(operator_field_failure_(iterate),
                                  "PreparedAffineLinearProblem::apply_nullspace_gauge");
    nullspace_policy_.apply_gauge(iterate);
  }

  void effective_rhs(MultiFab& out, const MultiFab& rhs) const {
    require_current();
    long failure = std::max(vector_field_failure_(rhs), operator_field_failure_(out));
    failure = std::max(failure, distinct_storage_failure_(out, rhs));
    require_collective_arguments_(failure, "PreparedAffineLinearProblem::effective_rhs");
    effective_rhs_prepared_(out, rhs);
  }

  /// A_lin(v) = A(v) - A(0), valid for search directions even when boundaries/sources make A affine.
  void apply_linear(MultiFab& out, const MultiFab& direction) const {
    require_current();
    long failure = std::max(operator_field_failure_(direction), operator_field_failure_(out));
    failure = std::max(failure, distinct_storage_failure_(out, direction));
    require_collective_arguments_(failure, "PreparedAffineLinearProblem::apply_linear");
    apply_linear_prepared_(out, direction);
  }

  /// Scientific residual R(u) = b - A(u), never a preconditioned or Arnoldi estimate.
  void true_residual(MultiFab& out, const MultiFab& rhs, const MultiFab& iterate) const {
    require_current();
    long failure = std::max(vector_field_failure_(rhs), operator_field_failure_(iterate));
    failure = std::max(failure, operator_field_failure_(out));
    failure = std::max(failure, distinct_storage_failure_(out, rhs));
    failure = std::max(failure, distinct_storage_failure_(out, iterate));
    require_collective_arguments_(failure, "PreparedAffineLinearProblem::true_residual");
    true_residual_prepared_(out, rhs, iterate);
  }

  void apply_preconditioner(MultiFab& out, const MultiFab& in) const {
    require_current();
    long failure = std::max(operator_field_failure_(in), operator_field_failure_(out));
    failure = std::max(failure, distinct_storage_failure_(out, in));
    require_collective_arguments_(failure, "PreparedAffineLinearProblem::apply_preconditioner");
    apply_preconditioner_prepared_(out, in);
  }

  /// The delivered prepared metric is one global L2 product over every component and rank. Keeping
  /// it on the problem (rather than inside individual algorithms) gives every method and report one
  /// authority and leaves a narrow metric-provider seam for a future weighted/composite route.
  Real inner_product(const MultiFab& left, const MultiFab& right) const {
    require_current();
    require_collective_arguments_(
        std::max(vector_field_failure_(left), vector_field_failure_(right)),
        "PreparedAffineLinearProblem::inner_product");
    return PureFieldAlgebra::dot(left, right);
  }

  Real residual_norm(const MultiFab& value) const {
    require_current();
    require_collective_arguments_(vector_field_failure_(value),
                                  "PreparedAffineLinearProblem::residual_norm");
    return PureFieldAlgebra::norm(value);
  }

 private:
  static constexpr long kAliasFailure = 1;
  static constexpr long kGhostFailure = 2;
  static constexpr long kVectorSpaceFailure = 3;

  long vector_field_failure_(const MultiFab& value) const noexcept {
    return PureFieldAlgebra::same_vector_space(value, zero_) ? 0 : kVectorSpaceFailure;
  }

  long operator_field_failure_(const MultiFab& value) const noexcept {
    const long vector_failure = vector_field_failure_(value);
    if (vector_failure != 0)
      return vector_failure;
    return value.n_grow() == footprint_.input_ghosts ? 0 : kGhostFailure;
  }

  static long distinct_storage_failure_(const MultiFab& out, const MultiFab& in) noexcept {
    return out.shares_storage_with(in) ? kAliasFailure : 0;
  }

  static void require_collective_arguments_(long local_failure, const char* where) {
    const long collective_failure = all_reduce_max(local_failure);
    if (collective_failure == kVectorSpaceFailure)
      throw std::invalid_argument(std::string(where) + ": incompatible vector space");
    if (collective_failure == kGhostFailure)
      throw std::invalid_argument(std::string(where) + ": incompatible ghost footprint");
    if (collective_failure == kAliasFailure)
      throw std::invalid_argument(std::string(where) + ": output aliases an input field");
  }

  void require_prepared_local_(const char* where) const {
    if (!snapshot_)
      throw std::logic_error(std::string(where) + ": problem is not prepared");
  }

  void require_current() const {
    OperatorEvaluationSnapshot observed{};
    bool probe_failed = false;
    try {
      observed = snapshot_probe_();
    } catch (...) {
      probe_failed = true;
    }
    long local_failure = 0;
    if (!snapshot_)
      local_failure = 3;
    else if (probe_failed)
      local_failure = 2;
    else if (observed != *snapshot_)
      local_failure = 1;
    const long collective_failure = all_reduce_max(local_failure);
    if (collective_failure == 3)
      throw std::logic_error(
          "PreparedAffineLinearProblem is not prepared on every communicator rank");
    if (collective_failure == 2)
      throw std::logic_error("operator snapshot probe failed on at least one communicator rank");
    if (collective_failure == 1)
      throw std::logic_error(
          "operator snapshot mutated after preparation on at least one communicator rank");
  }

  friend struct detail::PreparedProblemAccess;

  enum class PrepareStage : std::uint8_t {
    kFreeze,
    kNullspace,
    kPreconditioner,
    kOperatorConstant,
  };

  struct NullspaceCertificateIdentity {
    std::uint64_t topology_revision = 0;
    OperatorFingerprint topology{};
    OperatorFingerprint resources{};

    friend bool operator==(const NullspaceCertificateIdentity&,
                           const NullspaceCertificateIdentity&) = default;
  };

  static NullspaceCertificateIdentity nullspace_certificate_identity_(
      const OperatorEvaluationSnapshot& snapshot) noexcept {
    return {snapshot.topology_revision, snapshot.topology, snapshot.resources};
  }

  OperatorFingerprint compute_nullspace_plan_fingerprint_() const noexcept {
    OperatorFingerprint hash = detail::fingerprint_seed();
    const FieldNullspacePlan& plan = nullspace_policy_.plan_;
    detail::fingerprint_mix(hash, static_cast<std::uint64_t>(
                                      static_cast<std::int64_t>(nullspace_policy_.first_level_)));
    detail::fingerprint_mix(hash, plan.identity);
    detail::fingerprint_mix(hash, plan.layout_identity);
    detail::fingerprint_mix(hash, static_cast<std::uint64_t>(plan.scope));
    detail::fingerprint_mix(hash, static_cast<std::uint64_t>(plan.bases.size()));
    for (const FieldNullspaceBasis& basis : plan.bases) {
      detail::fingerprint_mix(hash, basis.identity);
      detail::fingerprint_mix(hash, basis.provenance);
      detail::fingerprint_mix(hash, basis.recipe_identity);
      detail::fingerprint_mix(
          hash, static_cast<std::uint64_t>(static_cast<std::int64_t>(basis.field_component)));
      detail::fingerprint_mix(hash, static_cast<std::uint64_t>(basis.masks.size()));
      for (const auto& mask : basis.masks) {
        detail::fingerprint_mix(hash, static_cast<std::uint64_t>(static_cast<bool>(mask)));
        if (mask)
          for (const std::uint64_t word : detail::layout_fingerprint(*mask))
            detail::fingerprint_mix(hash, word);
      }
      detail::fingerprint_mix(hash, static_cast<std::uint64_t>(basis.coverage.size()));
      for (const auto& coverage : basis.coverage) {
        detail::fingerprint_mix(hash, static_cast<std::uint64_t>(static_cast<bool>(coverage)));
        if (coverage)
          for (const std::uint64_t word : detail::layout_fingerprint(*coverage))
            detail::fingerprint_mix(hash, word);
      }
      detail::fingerprint_mix(hash, static_cast<std::uint64_t>(basis.cell_measure.size()));
      for (const Real measure : basis.cell_measure)
        detail::fingerprint_mix(hash, std::bit_cast<std::uint64_t>(measure));
    }
    detail::fingerprint_mix(hash, static_cast<std::uint64_t>(plan.gauges.size()));
    for (const FieldGaugeConstraint& gauge : plan.gauges) {
      detail::fingerprint_mix(hash, gauge.basis_identity);
      detail::fingerprint_mix(hash, std::bit_cast<std::uint64_t>(gauge.value));
    }
    return hash;
  }

  void append_collective_contract_(detail::KrylovCollectivePayload& payload) const noexcept {
    payload.append(properties_.bits);
    payload.append(footprint_.components);
    payload.append(footprint_.input_ghosts);
    payload.append(footprint_.restart);
    payload.append(static_cast<std::uint8_t>(footprint_.preconditioned));
    payload.append(static_cast<std::uint8_t>(preconditioner_.is_identity()));
    payload.append(static_cast<std::uint8_t>(preconditioner_.snapshot_.has_value()));
    payload.append(preconditioner_.snapshot_.value_or(OperatorEvaluationSnapshot{}));
    payload.append(static_cast<std::uint8_t>(apply_purity_));
    payload.append(layout_);
    payload.append(static_cast<std::uint8_t>(nullspace_policy_.singular_));
    payload.append(static_cast<std::uint8_t>(nullspace_policy_.prepared_));
    payload.append(static_cast<std::uint64_t>(nullspace_policy_.moments_.size()));
    payload.append(nullspace_policy_.plan_fingerprint_);
    payload.append(static_cast<std::uint8_t>(nullspace_certificate_identity_cache_.has_value()));
    const NullspaceCertificateIdentity nullspace_identity =
        nullspace_certificate_identity_cache_.value_or(NullspaceCertificateIdentity{});
    payload.append(nullspace_identity.topology_revision);
    payload.append(nullspace_identity.topology);
    payload.append(nullspace_identity.resources);
    payload.append(static_cast<std::uint8_t>(snapshot_.has_value()));
    payload.append(snapshot_.value_or(OperatorEvaluationSnapshot{}));
  }

  void require_collective_prepare_contract_() const {
    detail::KrylovCollectivePayload payload;
    append_collective_contract_(payload);
    if (!detail::collective_payload_agrees(payload))
      throw std::logic_error(
          "prepared affine problem contract differs across communicator ranks before preparation");
  }

  template <class Operation>
  void run_collective_prepare_stage_(PrepareStage stage, Operation&& operation) {
    long local_failure = 0;
    try {
      std::forward<Operation>(operation)();
    } catch (...) {
      local_failure = 1;
    }
    if (all_reduce_max(local_failure) == 0)
      return;
    switch (stage) {
      case PrepareStage::kFreeze:
        throw std::logic_error("prepared resource freeze failed on at least one communicator rank");
      case PrepareStage::kNullspace:
        throw std::logic_error("prepared nullspace setup failed on at least one communicator rank");
      case PrepareStage::kPreconditioner:
        throw std::logic_error(
            "prepared preconditioner setup failed on at least one communicator rank");
      case PrepareStage::kOperatorConstant:
        throw std::logic_error(
            "prepared affine constant evaluation failed on at least one communicator rank");
    }
    throw std::logic_error("unknown prepared affine setup stage");
  }

  void prepare_nullspace_collectively_(const OperatorEvaluationSnapshot& snapshot) {
    // The prepared bit, plan fingerprint and cache identity have already passed the fixed
    // collective contract. A change in the authenticated topology/resource identity invalidates
    // the Gram certificate even when the provider keeps the same field allocation; time/stage-only
    // changes retain the cold certificate and never add work to a Krylov iteration.
    const NullspaceCertificateIdentity identity = nullspace_certificate_identity_(snapshot);
    if (nullspace_policy_.prepared_ && nullspace_certificate_identity_cache_ == identity)
      return;
    nullspace_policy_.invalidate_collective_certificate_();
    nullspace_certificate_identity_cache_.reset();
    try {
      run_collective_prepare_stage_(PrepareStage::kNullspace, [&] {
        nullspace_policy_.prepare_after_collective_preflight_(zero_);
      });
      nullspace_certificate_identity_cache_ = identity;
    } catch (...) {
      // run_collective_prepare_stage_ has completed its common failure reduction. Clear the
      // certificate everywhere before exposing the uniform failure, otherwise a rank that
      // completed the Gram check could make the next preflight disagree with a failing rank.
      nullspace_policy_.invalidate_collective_certificate_();
      nullspace_certificate_identity_cache_.reset();
      throw;
    }
  }

  void require_collective_prepare_snapshot_(const OperatorEvaluationSnapshot& expected) const {
    long local_failure = expected.valid() ? 0 : 3;
    try {
      const OperatorEvaluationSnapshot observed = snapshot_probe_();
      if (local_failure == 0 && observed != expected)
        local_failure = 1;
    } catch (...) {
      if (local_failure == 0)
        local_failure = 2;
    }

    auto minimum_payload = detail::snapshot_consensus_payload(expected);
    auto maximum_payload = minimum_payload;
    all_reduce_min_inplace(minimum_payload.data(), minimum_payload.size());
    all_reduce_max_inplace(maximum_payload.data(), maximum_payload.size());
    const long collective_failure = all_reduce_max(local_failure);

    if (collective_failure == 3)
      throw std::invalid_argument(
          "PreparedAffineLinearProblem received an invalid snapshot on at least one "
          "communicator rank");
    if (collective_failure == 2)
      throw std::logic_error(
          "operator snapshot probe failed before preparation on at least one communicator rank");
    if (collective_failure == 1)
      throw std::logic_error(
          "operator snapshot changed before preparation on at least one communicator rank");
    if (minimum_payload != maximum_payload)
      throw std::logic_error(
          "operator snapshot differs across communicator ranks before preparation");
  }

  void require_collective_snapshot_match_(const OperatorEvaluationSnapshot& expected,
                                          PrepareStage stage) const {
    long local_failure = 0;
    try {
      if (snapshot_probe_() != expected)
        local_failure = 1;
    } catch (...) {
      local_failure = 2;
    }
    const long collective_failure = all_reduce_max(local_failure);
    if (collective_failure == 2) {
      if (stage == PrepareStage::kFreeze)
        throw std::logic_error(
            "operator snapshot probe failed during resource preparation on at least one "
            "communicator rank");
      if (stage == PrepareStage::kPreconditioner)
        throw std::logic_error(
            "operator snapshot probe failed during preconditioner preparation on at least one "
            "communicator rank");
      throw std::logic_error(
          "operator snapshot probe failed during operator preparation on at least one "
          "communicator rank");
    }
    if (collective_failure == 1) {
      if (stage == PrepareStage::kFreeze)
        throw std::logic_error(
            "operator snapshot changed during resource preparation on at least one communicator "
            "rank");
      if (stage == PrepareStage::kPreconditioner)
        throw std::logic_error(
            "operator snapshot changed during preconditioner preparation on at least one "
            "communicator rank");
      throw std::logic_error(
          "operator snapshot changed during operator preparation on at least one communicator "
          "rank");
    }
  }

  void require_hot_apply_ready_() const {
    // The public direct-call methods and solve preflight already authenticate the current snapshot.
    // Authenticated Program callbacks additionally promise that, after successful prepare(), no
    // rank can throw locally and no captured evaluation identity can mutate. Their generated hot
    // loop therefore needs neither a probe nor an extra MPI reduction for each matvec.
    if (!snapshot_)
      throw std::logic_error("authenticated prepared operator was used before preparation");
  }

  template <class Operation>
  void run_external_apply_(Operation&& operation) const {
    if (apply_purity_ == OperatorApplyPurity::kAuthenticatedProgram) {
      std::forward<Operation>(operation)();
      return;
    }

    long local_failure = 0;
    try {
      std::forward<Operation>(operation)();
    } catch (...) {
      local_failure = 3;
    }
    OperatorEvaluationSnapshot observed{};
    try {
      observed = snapshot_probe_();
      if (local_failure == 0 && (!snapshot_ || observed != *snapshot_))
        local_failure = 1;
    } catch (...) {
      if (local_failure == 0)
        local_failure = 2;
    }
    const long collective_failure = all_reduce_max(local_failure);
    if (collective_failure == 3)
      throw std::logic_error(
          "external prepared operator callback failed on at least one communicator rank");
    if (collective_failure == 2)
      throw std::logic_error("operator snapshot probe failed on at least one communicator rank");
    if (collective_failure == 1)
      throw std::logic_error(
          "operator snapshot mutated after preparation on at least one communicator rank");
  }

  void effective_rhs_prepared_(MultiFab& out, const MultiFab& rhs) const {
    require_hot_apply_ready_();
    detail::PreparedFieldAlgebra::lincomb(out, Real(1), rhs, Real(-1), constant_);
  }

  PreparedEquationReference prepare_compatibility_rhs_prepared_(MultiFab& out,
                                                                const MultiFab& rhs) const {
    require_hot_apply_ready_();
    // Form the exact floating-point R(0)=b-A(0) first. The scale-safe norm below avoids squaring
    // overflow/underflow without globally rescaling b and A(0) before subtraction, which could
    // erase a small but representable residual on cells far below an unrelated global maximum.
    detail::PreparedFieldAlgebra::lincomb(out, Real(1), rhs, Real(-1), constant_);
    const Real reference = PureFieldAlgebra::norm(out);
    if (!std::isfinite(static_cast<double>(reference)))
      return {reference};
    if (reference > Real(0)) {
      // Divide directly instead of materializing 1/reference, which can overflow for a finite
      // subnormal reference.
      detail::PreparedFieldAlgebra::divide(out, reference);
      return {reference};
    }
    return {Real(0)};
  }

  void apply_linear_prepared_(MultiFab& out, const MultiFab& direction) const {
    require_hot_apply_ready_();
    run_external_apply_([&] { raw_apply_(out, direction); });
    detail::PreparedFieldAlgebra::axpy(out, Real(-1), constant_);
  }

  void apply_linear_normalized_prepared_(MultiFab& out, const MultiFab& direction,
                                         Real equation_scale) const {
    require_hot_apply_ready_();
    run_external_apply_([&] { raw_apply_(out, direction); });
    detail::PreparedFieldAlgebra::normalized_difference(out, out, constant_, equation_scale);
  }

  void true_residual_prepared_(MultiFab& out, const MultiFab& rhs, const MultiFab& iterate) const {
    require_hot_apply_ready_();
    run_external_apply_([&] { raw_apply_(out, iterate); });
    detail::PreparedFieldAlgebra::lincomb(out, Real(1), rhs, Real(-1), out);
  }

  void apply_preconditioner_prepared_(MultiFab& out, const MultiFab& in) const {
    require_hot_apply_ready_();
    run_external_apply_([&] { preconditioner_.apply(out, in, *snapshot_); });
  }

  Real inner_product_prepared_(const MultiFab& left, const MultiFab& right) const {
    return detail::PreparedFieldAlgebra::dot(left, right);
  }

  Real local_inner_product_prepared_(const MultiFab& left, const MultiFab& right) const {
    return detail::PreparedFieldAlgebra::local_dot(left, right);
  }

  void local_robust_inner_product_payload_prepared_(const MultiFab& left, const MultiFab& right,
                                                    double* payload) const {
    detail::PreparedFieldAlgebra::local_robust_dot_payload(left, right, payload);
  }

  Real inner_product_from_global_robust_payload_prepared_(const double* payload) const {
    return detail::PreparedFieldAlgebra::dot_from_global_robust_payload(payload);
  }

  Real residual_norm_prepared_(const MultiFab& value) const {
    return detail::PreparedFieldAlgebra::norm(value);
  }

  ApplyFn raw_apply_;
  mutable PreparedLinearPreconditioner preconditioner_;
  LinearOperatorProperties properties_{};
  KrylovFootprint footprint_{};
  PreparedNullspacePolicy nullspace_policy_;
  OperatorSnapshotProbe snapshot_probe_;
  PreparedResourceFn freeze_resources_;
  OperatorApplyPurity apply_purity_ = OperatorApplyPurity::kVerifyAfterApply;
  MultiFab zero_;
  MultiFab constant_;
  OperatorFingerprint layout_{};
  std::optional<OperatorEvaluationSnapshot> snapshot_{};
  std::optional<NullspaceCertificateIdentity> nullspace_certificate_identity_cache_{};
};

namespace detail {

/// Private, allocation-free access used only after solve_prepared_affine has authenticated the
/// caller-owned iterate/RHS and KrylovWorkspace against the prepared problem. Public direct calls on
/// PreparedAffineLinearProblem retain their complete defensive layout validation.
struct PreparedProblemAccess {
  static long append_collective_state(const PreparedAffineLinearProblem& problem,
                                      KrylovCollectivePayload& payload) noexcept {
    problem.append_collective_contract_(payload);

    OperatorEvaluationSnapshot observed{};
    bool probe_failed = false;
    try {
      observed = problem.snapshot_probe_();
    } catch (...) {
      probe_failed = true;
    }
    payload.append(static_cast<std::uint8_t>(probe_failed));
    payload.append(observed);
    if (!problem.snapshot_)
      return 3;
    if (probe_failed)
      return 2;
    return observed == *problem.snapshot_ ? 0 : 1;
  }

  static const std::optional<OperatorEvaluationSnapshot>& stored_snapshot(
      const PreparedAffineLinearProblem& problem) noexcept {
    return problem.snapshot_;
  }

  static bool matches_vector_space(const PreparedAffineLinearProblem& problem,
                                   const MultiFab& value) noexcept {
    return PureFieldAlgebra::same_vector_space(problem.zero_, value);
  }

  static PreparedEquationReference prepare_compatibility_rhs(
      const PreparedAffineLinearProblem& problem, MultiFab& out, const MultiFab& rhs) {
    return problem.prepare_compatibility_rhs_prepared_(out, rhs);
  }
  static void require_nullspace_compatible(const PreparedAffineLinearProblem& problem,
                                           const MultiFab& normalized_rhs) {
    problem.nullspace_policy_.require_compatible(normalized_rhs);
  }
  static void apply_nullspace_gauge(const PreparedAffineLinearProblem& problem, MultiFab& iterate) {
    problem.nullspace_policy_.apply_gauge(iterate);
  }
  static void apply_linear(const PreparedAffineLinearProblem& problem, MultiFab& out,
                           const MultiFab& direction, Real equation_scale) {
    problem.apply_linear_normalized_prepared_(out, direction, equation_scale);
  }
  static void true_residual_physical(const PreparedAffineLinearProblem& problem, MultiFab& out,
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
  static void local_robust_inner_product_payload(const PreparedAffineLinearProblem& problem,
                                                 const MultiFab& left, const MultiFab& right,
                                                 double* payload) {
    problem.local_robust_inner_product_payload_prepared_(left, right, payload);
  }
  static Real inner_product_from_global_robust_payload(const PreparedAffineLinearProblem& problem,
                                                       const double* payload) {
    return problem.inner_product_from_global_robust_payload_prepared_(payload);
  }
  static Real residual_norm(const PreparedAffineLinearProblem& problem, const MultiFab& value) {
    return problem.residual_norm_prepared_(value);
  }
};

}  // namespace detail

}  // namespace pops
