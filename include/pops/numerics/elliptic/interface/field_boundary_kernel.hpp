#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/layout/field_distribution.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace pops {

/// Logical time carried by one residual evaluation.  Stage identity is an immutable numeric wire id
/// emitted by the Program; physical time is the exact clock coordinate evaluated from the current
/// macro time and dt.  No solver infers a stage from call order.
struct FieldLogicalTimePoint {
  Real time = Real(0);
  Real dt = Real(0);
  int clock_slot = 0;
  int partition_slot = 0;
  int stage_slot = 0;
  int level = 0;
  std::int64_t step = 0;
  int substep = 0;
  int iteration = 0;
  std::int64_t stage_fraction_numerator = 0;
  std::int64_t stage_fraction_denominator = 1;
};

/// Fallible device evaluation report. Generated launchers perform their own device reduction and
/// publish at most one deterministic witness here after the launch; device functors never throw.
template <int Dim>
struct FieldBoundaryFailure {
  static_assert(Dim >= 1 && Dim <= 3, "FieldBoundaryFailure only supports dimensions 1, 2, and 3");

  int code = 0;
  int face = -1;
  Index<Dim> cell{};
  Real value = Real(0);

  void reset() { *this = {}; }
  bool failed() const { return code != 0; }

  /// Make a device-local failure a rank-consistent decision before any caller can branch around a
  /// subsequent collective.  The lowest failing rank owns the deterministic witness; all ranks
  /// receive the same code/face/cell/value.  This intentionally lives at the host launcher seam:
  /// device functors first reduce into their local witness, then every rank calls this method in the
  /// same order even when it saw no local failure.
  bool synchronize_across_ranks(const ExecutionLane& lane) {
    const bool local_failed = failed();
    const long failure_count = all_reduce_sum(local_failed ? 1L : 0L, lane);
    if (failure_count == 0) {
      reset();
      return false;
    }

    const int rank = lane.rank();
    const int owner = static_cast<int>(
        all_reduce_min(static_cast<double>(local_failed ? rank : lane.size()), lane));
    const bool publish = local_failed && rank == owner;
    code = static_cast<int>(all_reduce_sum(publish ? static_cast<long>(code) : 0L, lane));
    face = static_cast<int>(all_reduce_sum(publish ? static_cast<long>(face) : 0L, lane));
    for (int axis = 0; axis < Dim; ++axis)
      cell[axis] =
          static_cast<int>(all_reduce_sum(publish ? static_cast<long>(cell[axis]) : 0L, lane));
    value = static_cast<Real>(all_reduce_sum(publish ? static_cast<double>(value) : 0.0, lane));
    return true;
  }
};

/// All dependencies are resolved to prepared execution views before entering a nonlinear/linear
/// iteration. Each distribution describes its view, independently of the iterate distribution. A
/// generated launcher resolves iterate global-patch ids to dependency-local ids once per local patch;
/// this supports replicated dependencies in a distributed solve without assuming equal local-index
/// order. A source that does not materialize every patch needed by the iterate must be remapped by the
/// runtime before installation. The device kernel sees only the selected Array4 values: no Python
/// callback, string map, virtual dispatch or registry lookup enters a face-cell loop.
template <int Dim>
struct FieldBoundaryExecutionContext {
  static_assert(Dim >= 1 && Dim <= 3,
                "FieldBoundaryExecutionContext only supports dimensions 1, 2, and 3");

  FieldLogicalTimePoint point{};
  const MultiFab<Dim>* const* states = nullptr;
  const FieldDistribution* state_distributions = nullptr;
  // Ordered owner-qualified identities travel beside the host pointer tables. They never enter a
  // device kernel; collective prepared solvers use them to distinguish equal-layout dependencies
  // and to reject a rank-local permutation before publishing a context.
  const std::string* state_identities = nullptr;
  int state_count = 0;
  const MultiFab<Dim>* const* fields = nullptr;
  const FieldDistribution* field_distributions = nullptr;
  const std::string* field_identities = nullptr;
  int field_count = 0;
  // Host-owned carrier selected by the launcher before a device submission.  Generated launchers
  // copy the exact scalars they use into their named POD functor; a std::vector pointer is therefore
  // never captured by, nor dereferenced on, the device.
  const std::vector<Real>* parameters = nullptr;
  int parameter_count = 0;
  FieldBoundaryFailure<Dim>* failure = nullptr;
  // Optional host-only Program Clock authority used by external host-batch executors. It is last
  // so generated device-kernel aggregate initializers retain their established numeric layout.
  const std::string* clock_identity = nullptr;
};

/// Stable owner for an already-routed set of level-qualified execution views. The retained
/// lifetime pins every pointer/string table referenced by the views; solvers retain this owner and
/// copy only one lightweight scalar/pointer view when stamping an iteration number.
template <int Dim>
class PreparedFieldBoundaryContextSet {
 public:
  using context_type = FieldBoundaryExecutionContext<Dim>;

  PreparedFieldBoundaryContextSet(std::shared_ptr<void> retained_lifetime, std::size_t levels)
      : retained_lifetime_(std::move(retained_lifetime)), contexts_(levels) {
    if (!retained_lifetime_ || levels == 0)
      throw std::invalid_argument(
          "prepared field boundary contexts require a retained owner and live levels");
  }

  PreparedFieldBoundaryContextSet(const PreparedFieldBoundaryContextSet&) = delete;
  PreparedFieldBoundaryContextSet& operator=(const PreparedFieldBoundaryContextSet&) = delete;

  std::size_t size() const noexcept { return contexts_.size(); }
  std::span<const context_type> contexts() const noexcept { return contexts_; }

  /// Binding happens before solver entry while its owning System/AMR session is exclusively held.
  /// No allocation or ownership change occurs; the solver receives only the resulting const owner.
  void bind(std::size_t level, context_type context) {
    if (level >= contexts_.size() || context.failure == nullptr)
      throw std::invalid_argument(
          "prepared field boundary context binding requires a valid level and failure channel");
    contexts_[level] = context;
  }

  context_type view(std::size_t level, int iteration) const {
    if (level >= contexts_.size() || contexts_[level].failure == nullptr)
      throw std::logic_error("prepared field boundary context view is unavailable");
    context_type result = contexts_[level];
    result.point.iteration = iteration;
    return result;
  }

 private:
  std::shared_ptr<void> retained_lifetime_;
  std::vector<context_type> contexts_;
};

/// Prepared residual and JVP launchers. A call handles one complete physical face. Generated
/// launchers remain device-clean named Kokkos functors; external providers may instead retain a
/// session-owned host-batch executor. The callable is selected and authenticated before the solve,
/// and no registry lookup or process-global trampoline enters the iterative path.
template <int Dim>
using FieldBoundaryPrepareResidualFn = std::function<void(
    int face, const MultiFab<Dim>& iterate, MultiFab<Dim>& operator_view,
    const Geometry<Dim>& geometry, const FieldBoundaryExecutionContext<Dim>& context)>;
template <int Dim>
using FieldBoundaryPrepareJvpFn =
    std::function<void(int face, const MultiFab<Dim>& iterate, const MultiFab<Dim>& direction,
                       MultiFab<Dim>& direction_view, const Geometry<Dim>& geometry,
                       const FieldBoundaryExecutionContext<Dim>& context)>;
/// Residual launchers use additive semantics: @c residual already contains `f-L(phi)` and the
/// launcher adds the exact boundary closure/elimination term `C(phi)` on boundary cells.
///
/// JVP launchers use the Newton-correction convention, not the derivative-of-residual convention:
/// @c output already contains `L'(phi)d` (including differentiated ghost elimination) and the
/// launcher adds `-C'(phi)d`.  Therefore the complete operator is `K=L'-C'=-R'` for
/// `R(phi)=f-L(phi)+C(phi)`, and Newton solves `K delta = R` before trying `phi + delta`.  Keeping
/// this sign at the generated-kernel ABI makes a residual/JVP finite-difference check unambiguous.
/// The iterate and direction are immutable mathematical inputs.
template <int Dim>
using FieldBoundaryResidualFn = std::function<void(
    int face, const MultiFab<Dim>& iterate, MultiFab<Dim>& residual, const Geometry<Dim>& geometry,
    const FieldBoundaryExecutionContext<Dim>& context)>;
template <int Dim>
using FieldBoundaryJvpFn = std::function<void(
    int face, const MultiFab<Dim>& iterate, const MultiFab<Dim>& direction, MultiFab<Dim>& output,
    const Geometry<Dim>& geometry, const FieldBoundaryExecutionContext<Dim>& context)>;

template <int Dim>
struct CompiledFieldBoundaryKernel {
  static_assert(Dim >= 1 && Dim <= 3,
                "CompiledFieldBoundaryKernel only supports dimensions 1, 2, and 3");

  std::string identity;
  std::string residual_identity;
  std::string jvp_identity;
  FieldBoundaryPrepareResidualFn<Dim> prepare_residual{};
  FieldBoundaryPrepareJvpFn<Dim> prepare_jvp{};
  FieldBoundaryResidualFn<Dim> residual{};
  FieldBoundaryJvpFn<Dim> jvp{};
  bool observes_iteration = false;

  bool empty() const { return !residual; }
  bool prepares_operator_views() const noexcept {
    return static_cast<bool>(prepare_residual) || static_cast<bool>(prepare_jvp);
  }

  void validate() const {
    if (identity.empty() || residual_identity.empty() || jvp_identity.empty() || !residual ||
        !jvp || static_cast<bool>(prepare_residual) != static_cast<bool>(prepare_jvp))
      throw std::runtime_error(
          "prepared field boundary kernel requires an exact residual/JVP pair and a complete "
          "optional operator-view preparation pair");
  }

  void prepare_residual_view(int face, const MultiFab<Dim>& iterate, MultiFab<Dim>& operator_view,
                             const Geometry<Dim>& geometry,
                             const FieldBoundaryExecutionContext<Dim>& context) const {
    if (prepare_residual)
      prepare_residual(face, iterate, operator_view, geometry, context);
  }

  void prepare_jvp_view(int face, const MultiFab<Dim>& iterate, const MultiFab<Dim>& direction,
                        MultiFab<Dim>& direction_view, const Geometry<Dim>& geometry,
                        const FieldBoundaryExecutionContext<Dim>& context) const {
    if (prepare_jvp)
      prepare_jvp(face, iterate, direction, direction_view, geometry, context);
  }

  void add_residual(int face, const MultiFab<Dim>& iterate, MultiFab<Dim>& output,
                    const Geometry<Dim>& geometry,
                    const FieldBoundaryExecutionContext<Dim>& context) const {
    residual(face, iterate, output, geometry, context);
  }

  void apply_jvp(int face, const MultiFab<Dim>& iterate, const MultiFab<Dim>& direction,
                 MultiFab<Dim>& output, const Geometry<Dim>& geometry,
                 const FieldBoundaryExecutionContext<Dim>& context) const {
    if (!jvp)
      throw std::runtime_error("field boundary closure has no compiled JVP launcher");
    jvp(face, iterate, direction, output, geometry, context);
  }
};

}  // namespace pops
