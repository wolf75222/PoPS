#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/layout/field_distribution.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>

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
  int step = 0;
  int substep = 0;
  int iteration = 0;
};

/// Fallible device evaluation report. Generated launchers perform their own device reduction and
/// publish at most one deterministic witness here after the launch; device functors never throw.
struct FieldBoundaryFailure {
  int code = 0;
  int face = -1;
  int i = 0;
  int j = 0;
  Real value = Real(0);

  void reset() { *this = {}; }
  bool failed() const { return code != 0; }

  /// Make a device-local failure a rank-consistent decision before any caller can branch around a
  /// subsequent collective.  The lowest failing rank owns the deterministic witness; all ranks
  /// receive the same code/face/cell/value.  This intentionally lives at the host launcher seam:
  /// device functors first reduce into their local witness, then every rank calls this method in the
  /// same order even when it saw no local failure.
  bool synchronize_across_ranks() {
    const bool local_failed = failed();
    const long failure_count = all_reduce_sum(local_failed ? 1L : 0L);
    if (failure_count == 0) {
      reset();
      return false;
    }

    const int rank = my_rank();
    const int owner =
        static_cast<int>(all_reduce_min(static_cast<double>(local_failed ? rank : n_ranks())));
    const bool publish = local_failed && rank == owner;
    code = static_cast<int>(all_reduce_sum(publish ? static_cast<long>(code) : 0L));
    face = static_cast<int>(all_reduce_sum(publish ? static_cast<long>(face) : 0L));
    i = static_cast<int>(all_reduce_sum(publish ? static_cast<long>(i) : 0L));
    j = static_cast<int>(all_reduce_sum(publish ? static_cast<long>(j) : 0L));
    value = static_cast<Real>(all_reduce_sum(publish ? static_cast<double>(value) : 0.0));
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
struct FieldBoundaryExecutionContext {
  FieldLogicalTimePoint point{};
  const MultiFab* const* states = nullptr;
  const FieldDistribution* state_distributions = nullptr;
  int state_count = 0;
  const MultiFab* const* fields = nullptr;
  const FieldDistribution* field_distributions = nullptr;
  int field_count = 0;
  // Host-owned carrier selected by the launcher before a device submission.  Generated launchers
  // copy the exact scalars they use into their named POD functor; a std::vector pointer is therefore
  // never captured by, nor dereferenced on, the device.
  const std::vector<Real>* parameters = nullptr;
  int parameter_count = 0;
  FieldBoundaryFailure* failure = nullptr;
};

/// Generated residual and JVP launchers.  A call handles one complete physical face and launches its
/// device-clean named Kokkos functor over all local face cells.  The function pointer is selected once
/// per solve/face outside the iterative hot loop; the function itself contains no runtime registry.
using FieldBoundaryPrepareResidualFn = void (*)(int face, const MultiFab& iterate,
                                                MultiFab& operator_view, const Geometry& geometry,
                                                const FieldBoundaryExecutionContext& context);
using FieldBoundaryPrepareJvpFn = void (*)(int face, const MultiFab& iterate,
                                           const MultiFab& direction, MultiFab& direction_view,
                                           const Geometry& geometry,
                                           const FieldBoundaryExecutionContext& context);
/// Residual launchers use additive semantics: @c residual already contains `f-L(phi)` and the
/// launcher adds the exact boundary closure/elimination term `C(phi)` on boundary cells.
///
/// JVP launchers use the Newton-correction convention, not the derivative-of-residual convention:
/// @c output already contains `L'(phi)d` (including differentiated ghost elimination) and the
/// launcher adds `-C'(phi)d`.  Therefore the complete operator is `K=L'-C'=-R'` for
/// `R(phi)=f-L(phi)+C(phi)`, and Newton solves `K delta = R` before trying `phi + delta`.  Keeping
/// this sign at the generated-kernel ABI makes a residual/JVP finite-difference check unambiguous.
/// The iterate and direction are immutable mathematical inputs.
using FieldBoundaryResidualFn = void (*)(int face, const MultiFab& iterate, MultiFab& residual,
                                         const Geometry& geometry,
                                         const FieldBoundaryExecutionContext& context);
using FieldBoundaryJvpFn = void (*)(int face, const MultiFab& iterate, const MultiFab& direction,
                                    MultiFab& output, const Geometry& geometry,
                                    const FieldBoundaryExecutionContext& context);

struct CompiledFieldBoundaryKernel {
  std::string identity;
  std::string residual_identity;
  std::string jvp_identity;
  FieldBoundaryPrepareResidualFn prepare_residual = nullptr;
  FieldBoundaryPrepareJvpFn prepare_jvp = nullptr;
  FieldBoundaryResidualFn residual = nullptr;
  FieldBoundaryJvpFn jvp = nullptr;
  bool observes_iteration = false;

  bool empty() const { return residual == nullptr; }

  void validate() const {
    if (identity.empty() || residual_identity.empty() || prepare_residual == nullptr ||
        residual == nullptr)
      throw std::runtime_error(
          "compiled field boundary kernel requires exact identity and residual launcher");
    if ((jvp == nullptr) != jvp_identity.empty() || (jvp == nullptr) != (prepare_jvp == nullptr))
      throw std::runtime_error(
          "compiled field boundary kernel JVP pointer and identity must be installed together");
    if (observes_iteration && jvp == nullptr)
      throw std::runtime_error(
          "iterate-dependent field boundary kernel requires an exact compiled JVP");
  }

  void prepare_residual_view(int face, const MultiFab& iterate, MultiFab& operator_view,
                             const Geometry& geometry,
                             const FieldBoundaryExecutionContext& context) const {
    prepare_residual(face, iterate, operator_view, geometry, context);
  }

  void prepare_jvp_view(int face, const MultiFab& iterate, const MultiFab& direction,
                        MultiFab& direction_view, const Geometry& geometry,
                        const FieldBoundaryExecutionContext& context) const {
    if (prepare_jvp == nullptr)
      throw std::runtime_error("field boundary closure has no compiled JVP preparation launcher");
    prepare_jvp(face, iterate, direction, direction_view, geometry, context);
  }

  void add_residual(int face, const MultiFab& iterate, MultiFab& output, const Geometry& geometry,
                    const FieldBoundaryExecutionContext& context) const {
    residual(face, iterate, output, geometry, context);
  }

  void apply_jvp(int face, const MultiFab& iterate, const MultiFab& direction, MultiFab& output,
                 const Geometry& geometry, const FieldBoundaryExecutionContext& context) const {
    if (jvp == nullptr)
      throw std::runtime_error("field boundary closure has no compiled JVP launcher");
    jvp(face, iterate, direction, output, geometry, context);
  }
};

}  // namespace pops
