#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <pops/amr/hierarchy/refinement_ratio.hpp>  // kAmrRefRatio (ratio 2)
#include <pops/core/foundation/kokkos_env.hpp>      // device_fence
#include <pops/mesh/layout/box_array.hpp>           // BoxArray / Box2D
#include <pops/mesh/layout/refinement.hpp>          // parallel_copy
#include <pops/mesh/storage/mf_arith.hpp>           // pops::lincomb (device-clean copy / negate)
#include <pops/mesh/storage/multifab.hpp>           // MultiFab / DistributionMapping
#include <pops/numerics/elliptic/mg/composite_fac_poisson.hpp>  // CompositeFacPoisson (composite FAC elliptic)
#include <pops/numerics/elliptic/linear/solve_report.hpp>  // SolveReport
#include <pops/runtime/amr/amr_runtime.hpp>  // AmrRuntime (the engine this helper reads)
#include <pops/runtime/amr/hierarchy_tensor_solver_provider.hpp>

/// @file
/// @brief AmrTensorElliptic -- the composite tensor-coefficient elliptic driver a compiled
///        condensed-implicit time Program routes to on a REFINED AMR hierarchy (ADC-633 / ADC-637).
///
/// The compiled condensed-implicit Program lowers, per AMR level, to inline block-inverse assembly
/// kernels (no coupling/schur call, ADC-637). On a FLAT hierarchy those run the emitted matrix-free
/// BiCGStab on level 0 -- bit-identical to the uniform Program. On a REFINED hierarchy (>= one fine
/// patch) the single-level matrix-free solve cannot address the fine levels, so the tensor elliptic is
/// solved COMPOSITELY: this helper owns per-level tensor-coefficient buffers (eps_x / eps_y / a_xy /
/// a_yx), a per-level right-hand side and a per-level potential, plus a lazily built, box-cached
/// pops::CompositeFacPoisson (two-way, variable coefficient + cross terms). The emitted assembly ops
/// write THROUGH AmrProgramContext::assembly_target into these level-shaped buffers (the level-0-bound
/// emitted scratch is unusable on a fine level); solve_composite() copies them into the FAC's per-level
/// fields, solves, and publishes each level's potential for the emitted reconstruction to READ through
/// AmrProgramContext::assembly_source.
///
/// GENERIC LAYER (owner directive, ADC-637): this driver names ONLY mathematical objects -- tensor
/// coefficients, right-hand side, potential, composite FAC. No B_z / Lorentz / electrostatic / Schur
/// vocabulary: the physics is authored in the DSL and emitted inline; this helper just co-distributes
/// the level buffers and drives the composite solve.
///
/// SCOPE. Inherited from the selected provider. The builtin CompositeFac provider supports N ratio-2
/// levels, replicated coarse ownership, distributed fine patches, MPI execution and the complete
/// two-dimensional tensor A=[[eps_x,a_xy],[a_yx,eps_y]]. Invalid nesting or ownership is rejected by
/// provider preparation; no coefficient is silently discarded.

namespace pops {
namespace runtime {
namespace program {

namespace detail {
/// Native FAC controls. CompositeTensorFAC owns the outer tolerance and iteration budget; those are
/// joined with these optional overrides only at the native direct-solve boundary.
struct TensorFacControls {
  std::optional<int> fine_sweeps;
  std::optional<Real> coarse_rel_tol;
  std::optional<Real> coarse_abs_tol;
  std::optional<int> coarse_cycles;
  std::optional<bool> verbose;
};

inline void validate_tensor_fac_controls(const TensorFacControls& options) {
  if (options.fine_sweeps && *options.fine_sweeps <= 0)
    throw std::invalid_argument("CompositeTensorFAC fine_sweeps must be positive");
  if (options.coarse_rel_tol &&
      (!std::isfinite(static_cast<double>(*options.coarse_rel_tol)) ||
       *options.coarse_rel_tol <= Real(0) || *options.coarse_rel_tol >= Real(1)))
    throw std::invalid_argument("CompositeTensorFAC coarse_rel_tol must be finite and in (0, 1)");
  if (options.coarse_abs_tol && (!std::isfinite(static_cast<double>(*options.coarse_abs_tol)) ||
                                 *options.coarse_abs_tol < Real(0)))
    throw std::invalid_argument("CompositeTensorFAC coarse_abs_tol must be finite and nonnegative");
  if (options.coarse_cycles && *options.coarse_cycles <= 0)
    throw std::invalid_argument("CompositeTensorFAC coarse_cycles must be positive");
}

inline TensorFacControls tensor_fac_controls(int fine_sweeps, Real coarse_rel_tol,
                                             Real coarse_abs_tol, int coarse_cycles, int verbose) {
  if (fine_sweeps < 0 || coarse_rel_tol < Real(0) || coarse_abs_tol < Real(0) || coarse_cycles < 0)
    throw std::invalid_argument(
        "CompositeTensorFAC wire options use zero for native default or a positive override");
  if (verbose < -1 || verbose > 1)
    throw std::invalid_argument(
        "CompositeTensorFAC verbose wire option must be -1 (native default), 0, or 1");
  TensorFacControls options{
      fine_sweeps == 0 ? std::nullopt : std::optional<int>(fine_sweeps),
      coarse_rel_tol == Real(0) ? std::nullopt : std::optional<Real>(coarse_rel_tol),
      coarse_abs_tol == Real(0) ? std::nullopt : std::optional<Real>(coarse_abs_tol),
      coarse_cycles == 0 ? std::nullopt : std::optional<int>(coarse_cycles),
      verbose == -1 ? std::nullopt : std::optional<bool>(verbose == 1)};
  validate_tensor_fac_controls(options);
  return options;
}

inline CompositeFacOptions tensor_fac_options(const TensorFacControls& controls, Real rel_tol,
                                              Real abs_tol, int max_iter) {
  validate_tensor_fac_controls(controls);
  if (!std::isfinite(static_cast<double>(rel_tol)) || rel_tol <= Real(0))
    throw std::invalid_argument(
        "CompositeTensorFAC Program solver rel_tol must be finite and positive");
  if (!std::isfinite(static_cast<double>(abs_tol)) || abs_tol < Real(0))
    throw std::invalid_argument(
        "CompositeTensorFAC Program solver abs_tol must be finite and nonnegative");
  if (max_iter <= 0)
    throw std::invalid_argument("CompositeTensorFAC Program solver max_iter must be positive");
  // CompositeFacOptions is the single native source of truth for omitted FAC knobs. The
  // direct solver controls are always overwritten, and only explicitly present controls
  // override the canonical native defaults.
  CompositeFacOptions options;
  options.max_iters = max_iter;
  options.rel_tol = rel_tol;
  options.abs_tol = abs_tol;
  if (controls.fine_sweeps)
    options.fine_sweeps = *controls.fine_sweeps;
  if (controls.coarse_rel_tol)
    options.coarse_rel_tol = *controls.coarse_rel_tol;
  if (controls.coarse_abs_tol)
    options.coarse_abs_tol = *controls.coarse_abs_tol;
  if (controls.coarse_cycles)
    options.coarse_cycles = *controls.coarse_cycles;
  if (controls.verbose)
    options.verbose = *controls.verbose;
  return options;
}
}  // namespace detail

/// Per-level tensor-coefficient buffers + a cached composite FAC solve, for one AMR block's condensed
/// tensor elliptic on a refined hierarchy. Owned by AmrProgramContext (one per installed Program on the
/// refined path); rebuilt lazily when the fine tiling changes. Indexed by AMR level (0 = coarse).
class AmrTensorElliptic final : public PreparedHierarchyTensorSolver {
 public:
  /// @p eng: the AMR engine (levels / geom / bc); @p block: the exact AMR system block index;
  /// @p ncomp: the authenticated operator component count. The native tensor route is scalar.
  AmrTensorElliptic(AmrRuntime* eng, int block, int ncomp,
                    std::string prepared_contract = {})
      : eng_(eng), block_(block), ncomp_(ncomp), prepared_contract_(std::move(prepared_contract)) {
    if (ncomp_ != 1)
      throw std::invalid_argument("AmrTensorElliptic requires exactly one component");
  }

  /// Install the hierarchy-solver controls. Zero marks omitted numeric knobs and -1 marks omitted
  /// verbosity; those values resolve from CompositeFacOptions, the sole native defaults authority.
  void configure_composite_tensor_fac(int fine_sweeps, Real coarse_rel_tol, Real coarse_abs_tol,
                                      int coarse_cycles, int verbose) {
    fac_controls_ = detail::tensor_fac_controls(fine_sweeps, coarse_rel_tol, coarse_abs_tol,
                                                coarse_cycles, verbose);
  }

  /// Join native controls with CompositeTensorFAC tolerances / iteration budget. Public on this
  /// internal driver so the bridge is inspectable and unit-testable without constructing AMR storage.
  CompositeFacOptions composite_fac_options(Real rel_tol, Real abs_tol, int max_iter) const {
    if (!fac_controls_)
      throw std::logic_error(
          "AmrTensorElliptic requires configure_composite_tensor_fac before a composite solve");
    return detail::tensor_fac_options(*fac_controls_, rel_tol, abs_tol, max_iter);
  }

  std::string_view provider_identity() const noexcept override {
    return "pops.hierarchy.composite-tensor-fac";
  }
  std::uint64_t provider_version() const noexcept override { return 1; }
  std::string_view exact_prepared_contract() const noexcept override { return prepared_contract_; }

  HierarchyTensorSolverExecutionPath execution_path() const noexcept override {
    // This provider delegates a genuinely flat topology to the independently prepared Krylov
    // contract. Once a populated fine level exists it owns the complete gather/solve/publication
    // path. The Program core never names FAC and never infers this choice from the provider id.
    if (eng_ == nullptr)
      return HierarchyTensorSolverExecutionPath::PreparedKrylovFallback;
    for (int level = 1; level < eng_->nlev(); ++level)
      if (eng_->level_state(static_cast<std::size_t>(block_), level).box_array().size() != 0)
        return HierarchyTensorSolverExecutionPath::DirectProvider;
    return HierarchyTensorSolverExecutionPath::PreparedKrylovFallback;
  }

  /// The level-shaped WRITE target for an assembly field of @p role at level @p k. The emitted
  /// assembly kernel reaches it via AmrProgramContext::assembly_target so its per-cell write lands in
  /// the composite buffer instead of the level-0-bound emitted scratch. Slot identities belong to
  /// this concrete 2-D tensor provider; the provider-neutral Program context forwards them opaquely.
  MultiFab& assembly_target(std::string_view field_slot_identity, int k) override {
    ensure_level_buffers(k);
    LevelBuffers& lb = levels_[static_cast<std::size_t>(k)];
    if (field_slot_identity == "pops.tensor-elliptic.diagonal.x")
      return lb.eps_x;
    if (field_slot_identity == "pops.tensor-elliptic.diagonal.y")
      return lb.eps_y;
    if (field_slot_identity == "pops.tensor-elliptic.cross.xy")
      return lb.a_xy;
    if (field_slot_identity == "pops.tensor-elliptic.cross.yx")
      return lb.a_yx;
    if (field_slot_identity == "pops.tensor-elliptic.rhs")
      return lb.rhs;
    if (field_slot_identity == "pops.tensor-elliptic.flux")
      return lb.flux;
    throw std::runtime_error("AmrTensorElliptic received an unknown prepared field slot '" +
                             std::string(field_slot_identity) + "'");
  }

  /// The published composite potential of level @p k (filled by solve_composite): the emitted
  /// reconstruction reads it as phi^{n+theta} on that level (via AmrProgramContext::assembly_source).
  MultiFab& solution(int k) override {
    ensure_level_buffers(k);
    return levels_[static_cast<std::size_t>(k)].phi;
  }

  /// Stage the current level's explicit solve initial guess.  Gathering this separately from the
  /// published solution is load-bearing: a rejected/non-converged FAC attempt must not publish its
  /// partial iterate, and a retry must start from the authored guess rather than leaked solver state.
  /// @p guess == nullptr is the declared zero initial guess.
  void stage_initial_guess(int k, const MultiFab* guess) override {
    ensure_level_buffers(k);
    MultiFab& staged = levels_[static_cast<std::size_t>(k)].initial_guess;
    if (guess)
      copy0(staged, *guess);
    else
      staged.set_val(Real(0));
  }

  /// Solve the composite tensor elliptic across the whole nested tower: build/reuse the FAC on the fine
  /// tilings, copy the per-level coefficient / RHS buffers into the FAC's level fields, enable variable
  /// coefficient + cross terms + two-way, solve, then publish each level's potential into phi(k). REUSES
  /// pops::CompositeFacPoisson wholesale. This is a direct solver with a structurally authenticated
  /// tensor operator. The builtin FAC provider owns MPI redistribution and both diagonal fields.
  SolveReport solve(const HierarchyTensorSolveControls& controls) override {
    return solve_composite(controls.relative_tolerance, controls.absolute_tolerance,
                           controls.maximum_iterations);
  }

  SolveReport solve_composite(Real rel_tol, Real abs_tol, int max_iter) {
    const int L = eng_->nlev();
    if (L < 2)
      return SolveReport::capability_failure();
    for (int k = 0; k < L; ++k)
      ensure_level_buffers(k);

    // The fine tilings (levels 1..L-1) key the FAC build; rebuild only when a tiling changes.
    std::vector<BoxArray> level_boxes;
    for (int k = 1; k < L; ++k)
      level_boxes.push_back(eng_->level_state(static_cast<std::size_t>(block_), k).box_array());
    ensure_fac(level_boxes);

    fac_->use_variable_coefficient(true);
    fac_->use_anisotropic_coefficient(true);
    fac_->use_cross_terms(true);
    fac_->set_two_way(true);
    for (int k = 0; k < L; ++k) {
      LevelBuffers& lb = levels_[static_cast<std::size_t>(k)];
      // The tensor coefficient A = [[eps_x, a_xy], [a_yx, eps_y]] per level.
      copy0(fac_->eps_level(k), lb.eps_x);
      copy0(fac_->eps_y_level(k), lb.eps_y);
      copy0(fac_->a_xy_level(k), lb.a_xy);
      copy0(fac_->a_yx_level(k), lb.a_yx);
      // the emitted condensed_rhs builds -Lap phi^n - g div(F): the matrix-free operator sign is
      // -div(A grad); the FAC solves div(eps grad phi) = f, so f = -rhs (the sign convention #126).
      negate_into(fac_->rhs_level(k), lb.rhs);
      // Do not inherit a partial FAC iterate from a rejected attempt.  Every attempt starts from the
      // per-level guess gathered from the Program (zero, or the carried phi^n history).
      copy0(fac_->phi_level(k), lb.initial_guess);
    }

    // The FAC owns the exact composite R(0), mixed stop, iteration count and scientific relative
    // residual.  rel_tol/abs_tol are the outer composite controls, while the independently authored
    // coarse_rel_tol/coarse_abs_tol stay on every internal GeometricMG solve; no RHS-norm adapter may
    // reinterpret either contract.
    const CompositeFacOptions options = composite_fac_options(rel_tol, abs_tol, max_iter);
    fac_->set_options(options);
    fac_->solve(options.max_iters, options.fine_sweeps, options.rel_tol, options.abs_tol);
    const SolveReport report = fac_->last_solve_report();
    if (!report.solved_value_available())
      return report;
    // Publication is atomic with respect to solve success: reconstruction cannot observe a partial
    // iterate, and the final SolveOutcome/StepTransaction contract can roll back later phases without
    // a failed solve having exposed a value.
    for (int k = 0; k < L; ++k)
      copy0(levels_[static_cast<std::size_t>(k)].phi, fac_->phi_level(k));
    return report;
  }

 private:
  struct LevelBuffers {
    MultiFab eps_x, eps_y, a_xy, a_yx;  ///< tensor coefficient A = [[eps_x, a_xy], [a_yx, eps_y]]
    MultiFab rhs;                       ///< condensed right-hand side (-Lap phi^n - g div F)
    MultiFab flux;            ///< transient explicit-flux scratch (2-comp, if the body uses it)
    MultiFab initial_guess;   ///< gathered per-level initial guess for the next solve attempt
    MultiFab phi;             ///< published composite potential of this level
    bool built = false;
  };

  /// Allocate level @p k's buffers on that level's grid (co-distributed with its state), once. eps /
  /// coefficient / phi carry 1 ghost (the operator face mean + the centered gradient); rhs 0 ghost.
  void ensure_level_buffers(int k) {
    if (k >= static_cast<int>(levels_.size()))
      levels_.resize(static_cast<std::size_t>(k) + 1);
    const MultiFab& U = eng_->level_state(static_cast<std::size_t>(block_), k);
    const BoxArray ba = U.box_array();
    const DistributionMapping dm = U.dmap();
    LevelBuffers& lb = levels_[static_cast<std::size_t>(k)];
    // Regrid may replace a level with a different patch tiling while retaining the level index.  The
    // old built flag alone would then route assembly into stale storage.  Multi-level execution is
    // The BoxArray determines allocation shape; redistribution is provider-owned at solve time.
    if (lb.built && lb.phi.box_array().boxes() == ba.boxes() &&
        lb.phi.dmap().ranks() == dm.ranks())
      return;
    lb = LevelBuffers{};
    lb.eps_x = MultiFab(ba, dm, 1, 1);
    lb.eps_y = MultiFab(ba, dm, 1, 1);
    lb.a_xy = MultiFab(ba, dm, 1, 1);
    lb.a_yx = MultiFab(ba, dm, 1, 1);
    lb.rhs = MultiFab(ba, dm, 1, 0);
    lb.flux = MultiFab(ba, dm, 2, 1);
    lb.initial_guess = MultiFab(ba, dm, 1, 1);
    lb.phi = MultiFab(ba, dm, 1, 1);
    lb.eps_x.set_val(Real(0));
    lb.eps_y.set_val(Real(0));
    lb.a_xy.set_val(Real(0));
    lb.a_yx.set_val(Real(0));
    lb.rhs.set_val(Real(0));
    lb.flux.set_val(Real(0));
    lb.initial_guess.set_val(Real(0));
    lb.phi.set_val(Real(0));
    lb.built = true;
  }

  /// Build (or rebuild on a fine-tiling change) the composite FAC over ALL fine levels -- the verbatim
  /// ensure_fac idiom of the native source stepper (compare per-level boxes + order, rebuild only on a
  /// change). A single fine level uses the 2-level ctor (bit-identical), deeper towers the N-level ctor;
  /// the FAC ctor refuses ratio != 2 / non-nested / misaligned patches, precisely.
  void ensure_fac(const std::vector<BoxArray>& level_boxes) {
    std::vector<std::vector<Box2D>> key;
    key.reserve(level_boxes.size());
    for (const BoxArray& ba : level_boxes)
      key.push_back(ba.boxes());
    if (fac_ && fac_level_boxes_ == key)
      return;
    const Geometry geom_c = eng_->level_geom(0);
    const BoxArray coarse_ba = eng_->level_state(static_cast<std::size_t>(block_), 0).box_array();
    if (level_boxes.size() == 1)
      fac_ = std::make_unique<CompositeFacPoisson>(geom_c, coarse_ba, eng_->poisson_bc(),
                                                   level_boxes[0], kAmrRefRatio);
    else
      fac_ = std::make_unique<CompositeFacPoisson>(geom_c, coarse_ba, eng_->poisson_bc(),
                                                   level_boxes, kAmrRefRatio);
    fac_level_boxes_ = std::move(key);
  }

  static void copy0(MultiFab& dst, const MultiFab& src) {
    dst.set_val(Real(0));
    parallel_copy(dst, src);
  }
  static void negate_into(MultiFab& dst, const MultiFab& src) {
    copy0(dst, src);
    pops::scale(dst, Real(-1));
  }

  AmrRuntime* eng_;
  int block_;
  int ncomp_;
  std::vector<LevelBuffers> levels_;
  std::unique_ptr<CompositeFacPoisson> fac_;
  std::vector<std::vector<Box2D>> fac_level_boxes_;
  std::optional<detail::TensorFacControls> fac_controls_;
  std::string prepared_contract_;
};

namespace detail {

inline constexpr std::string_view kCompositeTensorFacProvider =
    "pops.hierarchy.composite-tensor-fac";
inline constexpr std::string_view kCompositeTensorFacOptionSchema =
    "pops.hierarchy.composite-tensor-fac.options@1";
inline constexpr std::string_view kScalarTensorElliptic2dContract =
    "pops.operator.scalar-tensor-elliptic-2d@1";

inline const std::vector<std::string>& scalar_tensor_elliptic_2d_assembly_slots() {
  static const std::vector<std::string> slots{
      "pops.tensor-elliptic.diagonal.x", "pops.tensor-elliptic.diagonal.y",
      "pops.tensor-elliptic.cross.xy",   "pops.tensor-elliptic.cross.yx",
      "pops.tensor-elliptic.rhs",        "pops.tensor-elliptic.flux"};
  return slots;
}

inline bool request_matches_populated_hierarchy(
    const HierarchyTensorSolverBuildRequest& request) noexcept {
  try {
    if (request.runtime == nullptr || request.block < 0 ||
        request.block >= static_cast<int>(request.runtime->n_blocks()) ||
        request.levels != request.runtime->nlev())
      return false;
    if (request.level_populated.size() != static_cast<std::size_t>(request.levels) ||
        request.level_distributions.size() != static_cast<std::size_t>(request.levels))
      return false;
    for (int level = 0; level < request.levels; ++level) {
      const bool populated =
          request.runtime->level_state(static_cast<std::size_t>(request.block), level)
              .box_array()
              .size() != 0;
      const FieldDistribution distribution =
          request.runtime->level_is_replicated(level) ? FieldDistribution::Replicated
                                                      : FieldDistribution::Distributed;
      if (request.level_populated[static_cast<std::size_t>(level)] != populated ||
          request.level_distributions[static_cast<std::size_t>(level)] != distribution)
        return false;
    }
    return true;
  } catch (...) {
    return false;
  }
}

inline PreparedProviderOptions default_composite_tensor_fac_provider_options() {
  PreparedProviderOptions options;
  options.schema_identity = std::string(kCompositeTensorFacOptionSchema);
  return options;
}

inline TensorFacControls decode_composite_tensor_fac_provider_options(
    const PreparedProviderOptions& options) {
  if (options.schema_identity != kCompositeTensorFacOptionSchema)
    throw std::invalid_argument("invalid composite tensor FAC provider option schema");
  TensorFacControls controls;
  for (const auto& [key, value] : options.values) {
    if (key == "fac.fine_sweeps") {
      if (!std::holds_alternative<std::int64_t>(value))
        throw std::invalid_argument("fac.fine_sweeps must be int64");
      const auto typed = std::get<std::int64_t>(value);
      if (typed <= 0 || typed > std::numeric_limits<int>::max())
        throw std::invalid_argument("fac.fine_sweeps is outside the native range");
      controls.fine_sweeps = static_cast<int>(typed);
    } else if (key == "fac.coarse_rel_tol") {
      if (!std::holds_alternative<double>(value))
        throw std::invalid_argument("fac.coarse_rel_tol must be float64");
      controls.coarse_rel_tol = static_cast<Real>(std::get<double>(value));
    } else if (key == "fac.coarse_abs_tol") {
      if (!std::holds_alternative<double>(value))
        throw std::invalid_argument("fac.coarse_abs_tol must be float64");
      controls.coarse_abs_tol = static_cast<Real>(std::get<double>(value));
    } else if (key == "fac.coarse_cycles") {
      if (!std::holds_alternative<std::int64_t>(value))
        throw std::invalid_argument("fac.coarse_cycles must be int64");
      const auto typed = std::get<std::int64_t>(value);
      if (typed <= 0 || typed > std::numeric_limits<int>::max())
        throw std::invalid_argument("fac.coarse_cycles is outside the native range");
      controls.coarse_cycles = static_cast<int>(typed);
    } else if (key == "fac.verbose") {
      if (!std::holds_alternative<bool>(value))
        throw std::invalid_argument("fac.verbose must be bool");
      controls.verbose = std::get<bool>(value);
    } else {
      throw std::invalid_argument("unknown composite tensor FAC provider option '" + key + "'");
    }
  }
  validate_tensor_fac_controls(controls);
  return controls;
}

inline std::string composite_tensor_fac_prepared_contract(
    const HierarchyTensorSolverBuildRequest& request) {
  if (request.runtime == nullptr)
    throw std::invalid_argument("composite tensor FAC request has no AMR runtime");
  ExactContractBuilder contract;
  contract.text("pops.hierarchy.prepared-tensor-solver")
      .scalar(std::uint32_t{1})
      .text(kCompositeTensorFacProvider)
      .scalar(std::uint64_t{1})
      .text(request.plan_identity)
      .text(request.operator_contract_identity)
      .sequence(request.assembly_field_slots,
                [](ExactContractBuilder& item, const std::string& slot) { item.text(slot); })
      .text(request.solution_field_slot)
      .scalar(static_cast<std::int32_t>(request.block))
      .scalar(static_cast<std::int32_t>(request.components))
      .scalar(static_cast<std::int32_t>(request.levels))
      .sequence(request.level_populated,
                [](ExactContractBuilder& item, bool populated) { item.scalar(populated); })
      .sequence(request.level_distributions,
                [](ExactContractBuilder& item, FieldDistribution distribution) {
                  item.scalar(distribution);
                })
      .bytes(request.options.exact_contract());
  for (int level = 0; level < request.levels; ++level) {
    const MultiFab& layout =
        request.runtime->level_state(static_cast<std::size_t>(request.block), level);
    contract.scalar(static_cast<std::int32_t>(level))
        .sequence(layout.box_array().boxes(), [](ExactContractBuilder& item, const Box2D& box) {
          item.scalar(static_cast<std::int32_t>(box.lo[0]))
              .scalar(static_cast<std::int32_t>(box.lo[1]))
              .scalar(static_cast<std::int32_t>(box.hi[0]))
              .scalar(static_cast<std::int32_t>(box.hi[1]));
        })
        .scalar(request.runtime->level_is_replicated(level));
    if (!request.runtime->level_is_replicated(level))
      contract.sequence(layout.dmap().ranks());
  }
  return std::move(contract).release();
}

class CompositeTensorFacHierarchyProvider final : public HierarchyTensorSolverProvider {
 public:
  std::string_view identity() const noexcept override { return kCompositeTensorFacProvider; }
  std::uint64_t interface_version() const noexcept override { return 1; }
  std::string_view collective_contract() const noexcept override {
    return "pops.hierarchy.composite-tensor-fac@1";
  }
  std::vector<std::string> capability_contracts() const override {
    return {"pops.hierarchy.composite-tensor-fac.flat-krylov@1",
            "pops.hierarchy.composite-tensor-fac.refined-direct@1",
            "pops.hierarchy.composite-tensor-fac.mixed-level-distribution@1",
            "pops.hierarchy.composite-tensor-fac.exact-preparation@1"};
  }
  PreparedProviderOptions default_options() const override {
    return default_composite_tensor_fac_provider_options();
  }
  PreparedProviderSupport accepts_options(
      const PreparedProviderOptions& options) const noexcept override {
    try {
      (void)decode_composite_tensor_fac_provider_options(options);
      (void)options.exact_contract();
      return PreparedProviderSupport::accept();
    } catch (...) {
      return PreparedProviderSupport::reject(
          1, "composite tensor FAC options do not match the provider schema");
    }
  }
  PreparedProviderSupport supports(
      const HierarchyTensorSolverBuildRequest& request) const noexcept override {
    if (request.runtime == nullptr)
      return PreparedProviderSupport::reject(10, "AMR runtime authority is missing");
    if (request.block < 0 || request.components != 1 || request.levels < 1)
      return PreparedProviderSupport::reject(
          11, "request must select one scalar component on a non-empty hierarchy");
    if (request.level_populated.size() != static_cast<std::size_t>(request.levels) ||
        request.level_distributions.size() != static_cast<std::size_t>(request.levels))
      return PreparedProviderSupport::reject(
          12, "level population and distribution authorities do not cover the hierarchy");
    const bool refined = request.level_populated.size() > 1 &&
                         std::any_of(request.level_populated.begin() + 1,
                                     request.level_populated.end(), [](bool value) { return value; });
    const bool supported_distribution =
        !refined ||
        (request.level_distributions.size() == static_cast<std::size_t>(request.levels) &&
         request.level_distributions.front() == FieldDistribution::Replicated &&
         std::all_of(request.level_distributions.begin() + 1,
                     request.level_distributions.end(),
                     [](FieldDistribution distribution) {
                       return distribution == FieldDistribution::Distributed;
                     }));
    if (!supported_distribution)
      return PreparedProviderSupport::reject(
          13, "refined composite FAC requires replicated coarse and distributed refined levels");
    if (request.operator_contract_identity != kScalarTensorElliptic2dContract)
      return PreparedProviderSupport::reject(14, "operator contract is not scalar tensor elliptic 2D");
    if (request.assembly_field_slots != scalar_tensor_elliptic_2d_assembly_slots())
      return PreparedProviderSupport::reject(15, "assembly field-slot contract is incompatible");
    if (request.solution_field_slot != "pops.tensor-elliptic.solution")
      return PreparedProviderSupport::reject(16, "solution field-slot contract is incompatible");
    if (request.block >= static_cast<int>(request.runtime->n_blocks()))
      return PreparedProviderSupport::reject(17, "selected block is outside the AMR runtime");
    if (!request_matches_populated_hierarchy(request))
      return PreparedProviderSupport::reject(
          18, "request hierarchy does not match runtime population and distribution");
    if (!accepts_options(request.options).accepted())
      return PreparedProviderSupport::reject(19, "provider options are incompatible");
    return PreparedProviderSupport::accept();
  }
  PreparedProviderSupport accepts_execution(
      const HierarchyTensorSolverBuildRequest& request,
      HierarchyTensorSolverExecutionPath execution) const noexcept override {
    const PreparedProviderSupport request_support = supports(request);
    if (!request_support.accepted())
      return request_support;
    const bool refined = std::any_of(request.level_populated.begin() + 1,
                                     request.level_populated.end(),
                                     [](bool value) { return value; });
    const auto expected = refined ? HierarchyTensorSolverExecutionPath::DirectProvider
                                  : HierarchyTensorSolverExecutionPath::PreparedKrylovFallback;
    if (execution != expected)
      return PreparedProviderSupport::reject(
          20, "prepared execution path is incompatible with the resolved hierarchy");
    return PreparedProviderSupport::accept();
  }
  std::string expected_prepared_contract(
      const HierarchyTensorSolverBuildRequest& request) const override {
    return composite_tensor_fac_prepared_contract(request);
  }
  std::unique_ptr<PreparedHierarchyTensorSolver> prepare(
      const HierarchyTensorSolverBuildRequest& request) const override {
    if (!supports(request).accepted())
      throw std::invalid_argument("composite tensor FAC provider rejected the build request");
    const std::string contract = expected_prepared_contract(request);
    const TensorFacControls controls = decode_composite_tensor_fac_provider_options(request.options);
    auto prepared =
        std::make_unique<AmrTensorElliptic>(request.runtime, request.block, request.components,
                                           contract);
    prepared->configure_composite_tensor_fac(
        controls.fine_sweeps.value_or(0), controls.coarse_rel_tol.value_or(Real(0)),
        controls.coarse_abs_tol.value_or(Real(0)), controls.coarse_cycles.value_or(0),
        controls.verbose ? static_cast<int>(*controls.verbose) : -1);
    return prepared;
  }
};

}  // namespace detail

inline std::shared_ptr<HierarchyTensorSolverProviderRegistry>
make_default_hierarchy_tensor_solver_provider_registry() {
  auto registry = std::make_shared<HierarchyTensorSolverProviderRegistry>();
  registry->add(std::make_shared<detail::CompositeTensorFacHierarchyProvider>());
  return registry;
}

}  // namespace program
}  // namespace runtime
}  // namespace pops
