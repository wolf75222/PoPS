#pragma once

#include <pops/core/state/state.hpp>       // kAuxBaseComps (base component of the aux channel)
#include <pops/core/foundation/types.hpp>  // Real
#include <pops/diagnostics/runtime_diagnostics.hpp>
#include <pops/mesh/storage/multifab.hpp>  // MultiFab, Array4, ConstArray4
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/index/box2d.hpp>           // Box2D
#include <pops/mesh/execution/for_each.hpp>    // device_fence
#include <pops/mesh/boundary/physical_bc.hpp>  // BCRec, fill_ghosts, fill_boundary
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_bc_rec_adapter.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_builtins.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_prepare.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_provider.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_workspace.hpp>
#include <pops/numerics/elliptic/interface/field_provider.hpp>
#include <pops/numerics/elliptic/linear/pure_field_algebra.hpp>
#include <pops/numerics/elliptic/poisson/poisson_fft_solver.hpp>
#include <pops/numerics/elliptic/polar/polar_poisson_solver.hpp>  // PolarPoissonSolver (direct polar Poisson)
#include <pops/parallel/comm.hpp>                                 // n_ranks() (FFT MPI guard)
#include <pops/runtime/builders/block/block_builder_polar.hpp>  // derive_aux_polar (polar aux in local basis)
#include <pops/runtime/context/wall_predicate.hpp>              // detail::wall_predicate
#include <pops/runtime/config/generated_component_catalog.hpp>
#include <pops/core/identity/prepared_provider_options.hpp>
#include <pops/core/identity/sha256.hpp>
#include <pops/runtime/system/system_poisson_options.hpp>  // GeometricMgOptions (ADC-613 V-cycle knobs)
#include <pops/runtime/system/prepared_field_solver_component.hpp>
#include <pops/runtime/system/system_elliptic_backend.hpp>
#include <pops/mesh/storage/field_replica_consensus.hpp>

#include <algorithm>
#include <cstdlib>  // getenv
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>  // named_aux_: NAMED aux fields (comp -> field), re-applied after channel realloc
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

/// @file
/// @brief SystemFieldSolver: the ELLIPTIC SOLVE + FIELD DERIVATION responsibility extracted
///        from the god-class System::Impl (audit Lot B, cf. docs/SYSTEM_CPP_EXTRACTION_PLAN.md section 2).
///        Extracted VERBATIM from python/system.cpp: no change to the numerics, to the order of
///        operations, to fill_ghosts/fill_boundary, to device_fence or to tolerance. STRICTLY
///        bit-identical -- the code is moved as is, only access to the SHARED members of Impl
///        (aux, sp, cfg, geom, pgeom_, ba, dm, bc_, dom, per_, polar_) goes through the
///        back-pointer owner_->.
///
/// CONTRACT / INVARIANTS
/// - OWNS: the prepared cartesian elliptic backend selected by provider identity and its accepted
///   build request
///   (ell_), the separate direct polar backend (pell_), the Poisson configuration tokens (p_rhs/p_solver/p_bc/p_wall/
///   p_wall_radius/p_eps_), the typed coefficient form (scalar/diagonal/full tensor plus kappa) and
///   the aux field APPLICATION buffers (bz_field_ and te_src_) with their methods apply_bz /
///   apply_te.
/// - READS (without owning) the SHARED aux and the block list via owner_->: the aux is populated by
///   solve_fields (phi, grad phi, B_z, T_e) then its halos are filled; the block list provides the
///   right-hand side of the Poisson (sum of the per-block elliptic bricks) and the fluid source of T_e.
/// - DISPATCH cartesian vs polar: solve_fields() routes to solve_fields_polar() when owner_->polar_,
///   otherwise the cartesian path. The two paths are independent (ell_ never touched in polar and
///   vice versa). ensure_elliptic / ensure_elliptic_polar build the solver lazily.
/// - CRITICAL device INVARIANT: the device_fence() between ell_solve() and the derivation of grad phi MUST
///   stay atomic (without it, the GPU V-cycle is not finished when phi is read). Same in polar
///   after pell_->solve(). DO NOT reorder.
/// - MPI INVARIANT: the derivation / population loops (B_z, T_e, eps, kappa) iterate over the LOCAL
///   fabs (local_size()), never fab(0) hardcoded: no-op on a rank without a box, bit-identical to the
///   owner. This guard is PRESERVED by the extraction.
///
/// Since System::Impl stays PRIVATE to python/system.cpp, this helper is a TEMPLATE parametrized on the real
/// Impl type (same technique as native_loader): python/system.cpp instantiates it with System::Impl after
/// defining Impl. owner_ is an Impl* (the lifetime of the helper is subordinate to that of Impl).

namespace pops {

namespace detail {
/// Publishes one solved named potential and its optional signed centered gradient into the
/// block-owned aux components.  Keep this as a named device-callable functor: external generated
/// translation units instantiate this path, where an extended lambda is not portable under nvcc.
struct SystemNamedFieldPostprocessKernel {
  Array4 aux;
  ConstArray4 phi;
  int phi_component;
  int gradient_x_component;
  int gradient_y_component;
  Real gradient_scale;
  Real dx;
  Real dy;
  bool has_gradient;

  POPS_HD void operator()(int i, int j) const {
    aux(i, j, phi_component) = phi(i, j);
    if (has_gradient) {
      aux(i, j, gradient_x_component) =
          gradient_scale * (phi(i + 1, j) - phi(i - 1, j)) / (Real(2) * dx);
      aux(i, j, gradient_y_component) =
          gradient_scale * (phi(i, j + 1) - phi(i, j - 1)) / (Real(2) * dy);
    }
  }
};

struct SystemNamedAuxCopyKernel {
  Array4 aux;
  const Real* field;
  int component;
  int row_width;
  int origin_i;
  int origin_j;

  POPS_HD void operator()(int i, int j) const {
    aux(i, j, component) =
        field[static_cast<std::int64_t>(j - origin_j) * row_width + (i - origin_i)];
  }
};

/// Recomputes the electron temperature aux component from one compressible conservative state.
/// The functor is intentionally independent of any concrete model type: the prepared runtime has
/// already authenticated the canonical (rho, rho*u, rho*v, E) component contract and gamma.
struct SystemElectronTemperatureKernel {
  Array4 aux;
  ConstArray4 state;
  Real gamma_minus_one;
  int component;

  POPS_HD void operator()(int i, int j) const {
    const Real rho = state(i, j, 0);
    const Real mx = state(i, j, 1);
    const Real my = state(i, j, 2);
    const Real energy = state(i, j, 3);
    const Real pressure =
        gamma_minus_one * (energy - Real(0.5) * (mx * mx + my * my) / rho);
    aux(i, j, component) = pressure / rho;
  }
};

/// Copies one component between fields over an explicitly supplied valid region.  This is used by
/// the polar potential materialization seam, where the source component lives in the shared aux
/// channel while the destination is a dedicated one-ghost scalar field.
struct SystemFieldComponentCopyKernel {
  Array4 destination;
  ConstArray4 source;
  int destination_component;
  int source_component;

  POPS_HD void operator()(int i, int j) const {
    destination(i, j, destination_component) = source(i, j, source_component);
  }
};

static_assert(std::is_trivially_copyable_v<SystemNamedFieldPostprocessKernel>);
static_assert(std::is_trivially_copyable_v<SystemNamedAuxCopyKernel>);
static_assert(std::is_trivially_copyable_v<SystemElectronTemperatureKernel>);
static_assert(std::is_trivially_copyable_v<SystemFieldComponentCopyKernel>);
}  // namespace detail
namespace field_solver {

/// SystemFieldSolver<Impl>: see contract above. All methods are MEMBERS (not free
/// functions) because they share the elliptic state owned by this class; accesses to the SHARED
/// state of Impl go through owner_-> verbatim. Templated on Impl to stay free of any dependency on the
/// (private) definition of System::Impl.
template <class Impl>
class SystemFieldSolver {
 public:
  using NamedAuxField = std::vector<Real, fab_allocator<Real>>;

  /// Exactly one mathematically complete diffusion-coefficient shape can be present in a prepared
  /// request.  The provider-neutral native request accepts scalar, diagonal, or full 2-D tensor
  /// fields, so an extension can submit its exact materialized operator without parallel presence
  /// flags.  The current System authoring surface exposes only scalar and diagonal coefficients;
  /// there is deliberately no claim that named-field lowering already supplies full tensors.
  template <class Field>
  struct ScalarDiffusionCoefficient {
    Field value;
  };

  template <class Field>
  struct DiagonalDiffusionCoefficient {
    Field x;
    Field y;
  };

  template <class Field>
  struct FullTensorDiffusionCoefficient {
    Field xx;
    Field xy;
    Field yx;
    Field yy;
  };

  template <class Field>
  using DiffusionCoefficient =
      std::variant<std::monostate, ScalarDiffusionCoefficient<Field>,
                   DiagonalDiffusionCoefficient<Field>, FullTensorDiffusionCoefficient<Field>>;

  using AuthoredDiffusionCoefficient =
      std::variant<std::monostate, ScalarDiffusionCoefficient<std::vector<double>>,
                   DiagonalDiffusionCoefficient<std::vector<double>>>;
  using MaterializedDiffusionCoefficient = DiffusionCoefficient<MultiFab>;

  /// @param owner back-pointer to System::Impl (lifetime subordinate to that of Impl).
  explicit SystemFieldSolver(Impl* owner)
      : owner_(owner),
        nullspace_provider_registry_(make_default_field_nullspace_provider_registry()) {
    install_builtin_provider_presets_();
  }

  struct FieldSolveConfig;

  class NamedFieldBackend {
   public:
    virtual ~NamedFieldBackend() = default;
    [[nodiscard]] virtual std::string_view provider_identity() const noexcept = 0;
    [[nodiscard]] virtual std::uint64_t provider_version() const noexcept = 0;
    [[nodiscard]] virtual std::string_view provider_contract() const noexcept = 0;
    virtual MultiFab& rhs() = 0;
    virtual MultiFab& phi() = 0;
    [[nodiscard]] virtual const Geometry& geometry() const noexcept = 0;
    [[nodiscard]] virtual FieldDistribution field_distribution() const noexcept = 0;
    [[nodiscard]] virtual MultiFab snapshot() = 0;
    virtual void restore(const MultiFab&) = 0;
    virtual void configure_boundary(FieldSolveConfig&) = 0;
    virtual void prepare_rhs(SystemFieldSolver&, MultiFab&, const FieldSolveConfig&,
                             FieldNullspaceWorkspace&) = 0;
    virtual SolveReport solve(SystemFieldSolver&) = 0;
    virtual void finalize(SystemFieldSolver&, const FieldSolveConfig&,
                          FieldNullspaceWorkspace&) = 0;
    virtual void reset_diagnostics() {}
    [[nodiscard]] virtual RuntimeDiagnosticsReport diagnostics_report() const {
      return make_runtime_diagnostics_report("pops.runtime.elliptic_backend");
    }
    [[nodiscard]] virtual runtime::system::EllipticBackendMetrics metrics() const noexcept {
      return {};
    }
    [[nodiscard]] virtual const EllipticOperatorContract* operator_contract() const noexcept {
      return nullptr;
    }
    [[nodiscard]] virtual std::vector<runtime::field::FieldTopologyReportRow> topology_report()
        const {
      return {};
    }
  };

  /// Canonical component of T_e (after phi/grad/B_z); cf. pops::Aux and AUX_CANONICAL on the DSL side.
  static constexpr int kTeComp = kAuxBaseComps + 1;  // = 4

  /// True if the named aux field is provided at install time, for Spec-2 criterion 24 install-time
  /// requirement validation (ADC-446). Only the user-supplied APPLICATION fields can be a hard
  /// requirement: B_z (System::set_magnetic_field) and T_e (System::set_electron_temperature). The
  /// derived fields phi/grad_x/grad_y are always available (the elliptic solver builds lazily from
  /// the default Poisson config), and a generic named aux is keyed only by component C++-side (its
  /// name is not retained), so neither can be a hard failure here -- they return true (cannot block).
  bool provides_aux(const std::string& name) const {
    if (name == "B_z") {
      return !bz_field_.empty();
    }
    if (name == "T_e") {
      return te_src_ >= 0;
    }
    return true;
  }

  const RuntimeDiagnosticsReport& diagnostics_report() const { return diagnostics_; }
  void reset_diagnostics() {
    diagnostics_.clear();
    if (ell_)
      ell_->reset_diagnostics();
  }

  RuntimeDiagnosticsReport combined_diagnostics_report() const {
    RuntimeDiagnosticsReport report = diagnostics_;
    if (ell_) {
      const RuntimeDiagnosticsReport backend_report = ell_->diagnostics_report();
      report.events.insert(report.events.end(), backend_report.events.begin(),
                           backend_report.events.end());
      const std::size_t remaining = std::numeric_limits<std::size_t>::max() - report.dropped_events;
      report.dropped_events += std::min(backend_report.dropped_events, remaining);
    }
    return report;
  }

  void trace_mark(const char* marker) noexcept {
    if (std::getenv("POPS_TRACE_SOLVE_FIELDS") == nullptr)
      return;
    (void)diagnostics_.try_record("runtime.solve_fields.trace", "SystemFieldSolver", "trace",
                                  marker);
  }

  /// Mutable elliptic state that belongs to one step attempt. The solver objects themselves are
  /// structural and stay installed; only their warm-start potential and polar source buffer may be
  /// provisionally changed by a failed attempt.
  struct StepSnapshot {
    std::optional<MultiFab> potential;
    std::optional<MultiFab> polar_potential;
    std::optional<MultiFab> polar_source;
    bool had_elliptic = false;
    bool had_polar_solver = false;
    RuntimeDiagnosticsReport diagnostics;
    std::map<std::string, MultiFab> named_potentials;
    std::vector<std::string> named_unbuilt;
  };

  StepSnapshot step_snapshot() {
    StepSnapshot out;
    out.had_elliptic = static_cast<bool>(ell_);
    out.had_polar_solver = pell_.has_value();
    if (ell_)
      out.potential = ell_->snapshot();
    if (pell_)
      out.polar_potential = pell_->phi();
    if (phi_src_polar_)
      out.polar_source = *phi_src_polar_;
    for (auto& item : named_fields_) {
      if (!item.second.backend) {
        out.named_unbuilt.push_back(item.first);
        continue;
      }
      out.named_potentials.emplace(item.first, item.second.backend->snapshot());
    }
    out.diagnostics = diagnostics_;
    return out;
  }

  void restore_step_snapshot(const StepSnapshot& snapshot) {
    if (!snapshot.had_elliptic)
      invalidate_primary_backend_();
    else if (snapshot.potential && ell_)
      ell_->restore(*snapshot.potential);
    if (!snapshot.had_polar_solver)
      pell_.reset();
    else if (snapshot.polar_potential && pell_)
      pell_->phi() = *snapshot.polar_potential;
    phi_src_polar_ = snapshot.polar_source;
    for (auto& item : named_fields_) {
      if (std::find(snapshot.named_unbuilt.begin(), snapshot.named_unbuilt.end(), item.first) !=
          snapshot.named_unbuilt.end()) {
        invalidate_named_backend_(item.second);
        continue;
      }
      const auto saved = snapshot.named_potentials.find(item.first);
      if (saved != snapshot.named_potentials.end() && item.second.backend)
        item.second.backend->restore(saved->second);
    }
    diagnostics_ = snapshot.diagnostics;
  }

  // --- OWNED state (elliptic solve + coefficient fields + application buffers) --------
  // Poisson configuration (elliptic solver built lazily).
  std::string p_rhs = "charge_density";
  std::string p_solver = "geometric_mg";
  std::string p_bc = "auto";
  bool p_has_explicit_bc = false;
  BCRec p_explicit_bc{};
  FieldNullspacePlan p_nullspace_{};
  FieldNullspaceProviderSelection p_nullspace_provider_ = operator_topology_zero_mean_nullspace();
  bool p_nullspace_ready_ = false;
  // Compatibility and gauge are sequential operations on the same physical vector space.  Halo
  // width is not part of its scientific identity, so one persistent allocation-free evaluator is
  // shared by the backend RHS and solution representations.
  std::unique_ptr<FieldNullspaceWorkspace> p_nullspace_workspace_;
  std::string p_wall = "none";
  double p_wall_radius = 0.0;
  Real p_eps_ = 1;  // CONSTANT permittivity: div(eps grad phi) = f <=> lap phi = f/eps
  AuthoredDiffusionCoefficient p_diffusion_coefficient_{};
  bool has_kappa_field_ = false;  // REACTION term kappa(x) provided: div(eps grad phi) - kappa phi
  std::vector<double> p_kappa_field_;  // field kappa(x), n*n row-major (if has_kappa_field_)

  [[nodiscard]] bool has_variable_diffusion_coefficient() const noexcept {
    return !std::holds_alternative<std::monostate>(p_diffusion_coefficient_);
  }

  [[nodiscard]] bool has_scalar_diffusion_coefficient() const noexcept {
    return std::holds_alternative<ScalarDiffusionCoefficient<std::vector<double>>>(
        p_diffusion_coefficient_);
  }

  [[nodiscard]] bool has_anisotropic_diffusion_coefficient() const noexcept {
    return std::holds_alternative<DiagonalDiffusionCoefficient<std::vector<double>>>(
        p_diffusion_coefficient_);
  }

  void configure_scalar_diffusion_coefficient(std::vector<double> values) {
    if (!std::holds_alternative<std::monostate>(p_diffusion_coefficient_) &&
        !std::holds_alternative<ScalarDiffusionCoefficient<std::vector<double>>>(
            p_diffusion_coefficient_))
      throw std::runtime_error(
          "System diffusion coefficient already has another exact shape; scalar, diagonal and "
          "full-tensor coefficients cannot be combined");
    p_diffusion_coefficient_ = ScalarDiffusionCoefficient<std::vector<double>>{std::move(values)};
    invalidate_primary_backend_();
  }

  void configure_diagonal_diffusion_coefficient(std::vector<double> x, std::vector<double> y) {
    if (!std::holds_alternative<std::monostate>(p_diffusion_coefficient_) &&
        !std::holds_alternative<DiagonalDiffusionCoefficient<std::vector<double>>>(
            p_diffusion_coefficient_))
      throw std::runtime_error(
          "System diffusion coefficient already has another exact shape; scalar, diagonal and "
          "full-tensor coefficients cannot be combined");
    p_diffusion_coefficient_ =
        DiagonalDiffusionCoefficient<std::vector<double>>{std::move(x), std::move(y)};
    invalidate_primary_backend_();
  }
  // Prepared through the provider registry.  The core owns one type-erased backend and never
  // switches on a numerical implementation type.
  std::unique_ptr<NamedFieldBackend> ell_;
  // Transaction scratch is materialized on the first prepared evaluation (A(0)) and then reused by
  // every field-coupled Jacobian apply.  No MultiFab construction remains in the Krylov loop.
  std::optional<MultiFab> published_phi_scratch_;
  // Direct POLAR Poisson solver (FFT-in-theta + tridiag-in-r), built lazily when
  // polar_ (cf. ensure_elliptic_polar). SEPARATE from ell_ (geom() returns a PolarGeometry, not a
  // Geometry): the cartesian path is never touched. INERT (nullopt) in cartesian.
  std::optional<PolarPoissonSolver> pell_;
  // Ghosted polar potential buffer consumed by generated tensor Programs. The direct
  // PolarPoissonSolver stores only valid cells, whereas centered tensor/gradient operators need one
  // ghost layer. In Cartesian geometry this buffer is inert.
  std::optional<MultiFab> phi_src_polar_;
  NamedAuxField bz_field_;  // field B_z(x) n*n row-major, device-addressable (empty if absent)
  int te_src_ = -1;             // index of the fluid block source of T_e (-1 = none)
  // NAMED aux fields (ADC-70 phase 1) provided by the user via System::set_aux_field: key =
  // canonical component (>= kAuxNamedBase = 5), value = field n*n (cartesian) / nr*ntheta (polar)
  // row-major. PERSISTENT like bz_field_: solve_fields touches ONLY components 0..2 (phi,
  // grad) and 4 (T_e via apply_te), so components >= 5 survive from one step to the next; but a
  // REALLOCATION of the aux channel (ensure_aux_width) starts again from a zeroed MultiFab -> we re-apply
  // them then (apply_named_aux), exactly like apply_bz / apply_te.
  std::map<int, NamedAuxField> named_aux_;
  // Per-field aux HALO policy (ADC-369): key = canonical component (>= kAuxNamedBase), value = the
  // uniform boundary policy declared via pops.AuxHalo. Applied by apply_named_aux_bc() AFTER the shared
  // aux ghost fill, overriding only that component's PHYSICAL-face ghosts (periodic faces -- Cartesian
  // periodic, polar theta -- keep their wrap). Empty -> shared aux BC for every field, bit-identical.
  std::map<int, AuxHaloPolicy> named_aux_bc_;
  RuntimeDiagnosticsReport diagnostics_ =
      make_runtime_diagnostics_report("pops.runtime.system_field_solver");

  // NAMED multi-elliptic fields (ADC-428): a SECOND elliptic solve (beyond the default Poisson) for a
  // user-named field m.elliptic_field("phi2", rhs=..., aux=[...]). Each named field owns:
  //   - phi_comp / gx_comp / gy_comp: the aux channel components (>= kAuxNamedBase) its solution and
  //     centered gradient are written to (the model declares them as aux_field slots; a source then
  //     reads them like any other named aux). gx_comp / gy_comp < 0 => the field declares fewer than 3
  //     aux slots and the gradient is not derived (only phi is written).
  //   - one DEDICATED backend instance, built lazily behind the same protocol for builtin and
  //     external providers. Builtin backends reuse the native GeometricMG / FFT implementations;
  //     the default Poisson path (ell_) is untouched.
  // The RHS = sum over blocks of s.named_poisson_rhs[name] (the per-field brick, ADC-428), assembled
  // exactly like assemble_poisson_rhs but reading the named closures.
  struct FieldSolveConfig {
    std::string provider_identity;
    std::string plan_identity;
    std::string output_owner_identity;
    std::string output_block;
    std::string output_key;
    std::vector<FieldProviderBinding> providers;
    // Exact registry key. Builtins use the resolved route identity; installed native components
    // replace it with their manifest-qualified PreparedFieldSolverComponent identity.
    std::string backend_provider_identity = "geometric_mg";
    std::string bc = "auto";
    bool has_explicit_bc = false;
    BCRec explicit_bc{};
    bool has_boundary_kernel = false;
    CompiledFieldBoundaryKernel boundary_kernel{};
    std::shared_ptr<std::vector<Real>> boundary_parameters = std::make_shared<std::vector<Real>>();
    std::vector<std::string> boundary_state_blocks;
    std::vector<int> boundary_state_components;
    std::vector<std::string> boundary_field_blocks;
    std::vector<std::string> boundary_field_keys;
    std::vector<int> boundary_field_components;
    std::vector<const MultiFab*> boundary_state_buffers;
    std::vector<FieldDistribution> boundary_state_distributions;
    std::vector<const MultiFab*> boundary_field_buffers;
    std::vector<FieldDistribution> boundary_field_distributions;
    FieldBoundaryExecutionContext boundary_context{};
    bool has_reaction = false;
    Real reaction = Real(0);
    bool has_newton = false;
    FieldNewtonOptions newton{};
    FieldNullspacePlan nullspace{};
    FieldNullspaceProviderSelection nullspace_provider = operator_topology_zero_mean_nullspace();
    std::string topology_provider_kind;
    std::string topology_provenance;
    std::string topology_digest;
  };

  [[nodiscard]] static EllipticOperatorContract compose_provider_operator_contract_(
      std::string_view provider_identity, const EllipticOperatorContract& backend_contract,
      std::string_view exact_configuration_contract) {
    ExactContractBuilder options;
    options.text(provider_identity).bytes(exact_configuration_contract);
    return EllipticOperatorContract::make({"pops.runtime.system-elliptic-provider", 1},
                                          std::string(backend_contract.exact_fingerprint()),
                                          std::move(options).release());
  }

  class GeometricMgPreparedBackend final : public NamedFieldBackend {
   public:
    template <class... Args>
    GeometricMgPreparedBackend(std::string provider_identity, std::string provider_contract,
                               GeometricMgOptions solve_controls,
                               std::string exact_configuration_contract,
                               std::string topology_digest, std::string topology_provenance,
                               std::size_t material_points, Args&&... args)
        : solver_(std::forward<Args>(args)...),
          provider_identity_(std::move(provider_identity)),
          provider_contract_(std::move(provider_contract)),
          solve_controls_(solve_controls),
          topology_digest_(std::move(topology_digest)),
          topology_provenance_(std::move(topology_provenance)),
          material_points_(material_points) {
      operator_contract_ = compose_provider_operator_contract_(
          provider_identity_, solver_.prepared_operator_contract(), exact_configuration_contract);
    }

    [[nodiscard]] std::string_view provider_identity() const noexcept override {
      return provider_identity_;
    }
    [[nodiscard]] std::uint64_t provider_version() const noexcept override { return 1; }
    [[nodiscard]] std::string_view provider_contract() const noexcept override {
      return provider_contract_;
    }
    MultiFab& rhs() override { return solver_.rhs(); }
    MultiFab& phi() override { return solver_.phi(); }
    [[nodiscard]] const Geometry& geometry() const noexcept override { return solver_.geom(); }
    [[nodiscard]] FieldDistribution field_distribution() const noexcept override {
      return solver_.field_distribution();
    }
    [[nodiscard]] MultiFab snapshot() override { return MultiFab(solver_.phi()); }
    void restore(const MultiFab& value) override { solver_.phi() = value; }
    void configure_boundary(FieldSolveConfig& plan) override {
      if (plan.has_boundary_kernel)
        solver_.set_boundary_context(plan.boundary_context);
    }
    void prepare_rhs(SystemFieldSolver& owner, MultiFab& value, const FieldSolveConfig&,
                     FieldNullspaceWorkspace& nullspace) override {
      nullspace.require_compatible(value);
      scale(value, Real(-1) / owner.p_eps_);
    }
    SolveReport solve(SystemFieldSolver&) override {
      if (boundary_observes_iteration_)
        solver_.solve();
      else
        solver_.solve(solve_controls_.rel_tol, solve_controls_.max_cycles, solve_controls_.abs_tol);
      return solver_.last_solve_report();
    }
    void finalize(SystemFieldSolver&, const FieldSolveConfig&,
                  FieldNullspaceWorkspace& nullspace) override {
      nullspace.apply_gauge(solver_.phi());
    }
    void set_scalar_coefficient(const MultiFab& coefficient) { solver_.set_epsilon(coefficient); }
    void set_diagonal_tensor_coefficient(const MultiFab& x, const MultiFab& y) {
      solver_.set_epsilon_anisotropic(x, y);
    }
    void set_full_tensor_coefficient(const MultiFab& xx, const MultiFab& xy, const MultiFab& yx,
                                     const MultiFab& yy) {
      solver_.set_epsilon_anisotropic(xx, yy);
      solver_.set_cross_terms(xy, yx);
    }
    void set_reaction_coefficient(const MultiFab& coefficient) {
      solver_.set_reaction(coefficient);
    }
    void set_reaction_coefficient(ScalarFieldProvider2D coefficient) {
      solver_.set_reaction(std::move(coefficient));
    }
    void set_dynamic_boundary(const CompiledFieldBoundaryKernel& kernel,
                              const FieldBoundaryExecutionContext& context) {
      solver_.set_boundary_kernel(kernel, context);
      boundary_observes_iteration_ = kernel.observes_iteration;
    }
    void set_nonlinear_boundary(const FieldNewtonOptions& options) {
      solver_.set_field_newton_options(options);
    }
    void reset_diagnostics() override { solver_.reset_diagnostics(); }
    [[nodiscard]] RuntimeDiagnosticsReport diagnostics_report() const override {
      return solver_.diagnostics_report();
    }
    [[nodiscard]] runtime::system::EllipticBackendMetrics metrics() const noexcept override {
      return {solver_.last_cycles(), 0, solver_.num_levels(), solver_.last_bottom_seconds()};
    }
    [[nodiscard]] const EllipticOperatorContract* operator_contract() const noexcept override {
      return &operator_contract_;
    }
    [[nodiscard]] std::vector<runtime::field::FieldTopologyReportRow> topology_report()
        const override {
      if (topology_digest_.empty() || topology_provenance_.empty())
        return {};
      return {{"builtin:domain", topology_digest_, topology_provenance_, material_points_, 1}};
    }

   private:
    GeometricMG solver_;
    EllipticOperatorContract operator_contract_;
    std::string provider_identity_;
    std::string provider_contract_;
    GeometricMgOptions solve_controls_{};
    bool boundary_observes_iteration_ = false;
    std::string topology_digest_;
    std::string topology_provenance_;
    std::size_t material_points_ = 0;
  };

  template <class Solver>
  class DirectPreparedBackend final : public NamedFieldBackend {
   public:
    template <class... Args>
    DirectPreparedBackend(std::string provider_identity, std::string provider_contract,
                          std::string exact_configuration_contract, std::string topology_digest,
                          std::string topology_provenance, std::size_t material_points,
                          Args&&... args)
        : solver_(std::forward<Args>(args)...),
          provider_identity_(std::move(provider_identity)),
          provider_contract_(std::move(provider_contract)),
          topology_digest_(std::move(topology_digest)),
          topology_provenance_(std::move(topology_provenance)),
          material_points_(material_points) {
      operator_contract_ = compose_provider_operator_contract_(
          provider_identity_, solver_.prepared_operator_contract(), exact_configuration_contract);
    }

    [[nodiscard]] std::string_view provider_identity() const noexcept override {
      return provider_identity_;
    }
    [[nodiscard]] std::uint64_t provider_version() const noexcept override { return 1; }
    [[nodiscard]] std::string_view provider_contract() const noexcept override {
      return provider_contract_;
    }
    MultiFab& rhs() override { return solver_.rhs(); }
    MultiFab& phi() override { return solver_.phi(); }
    [[nodiscard]] const Geometry& geometry() const noexcept override { return solver_.geom(); }
    [[nodiscard]] FieldDistribution field_distribution() const noexcept override {
      return solver_.field_distribution();
    }
    [[nodiscard]] MultiFab snapshot() override { return MultiFab(solver_.phi()); }
    void restore(const MultiFab& value) override { solver_.phi() = value; }
    void configure_boundary(FieldSolveConfig& plan) override {
      if (plan.has_boundary_kernel)
        throw std::logic_error(
            "direct elliptic backend received a dynamic boundary after request validation");
    }
    void prepare_rhs(SystemFieldSolver& owner, MultiFab& value, const FieldSolveConfig&,
                     FieldNullspaceWorkspace& nullspace) override {
      nullspace.require_compatible(value);
      scale(value, Real(-1) / owner.p_eps_);
    }
    SolveReport solve(SystemFieldSolver&) override {
      solver_.solve();
      SolveReport report;
      report.mark_solved();
      return report;
    }
    void finalize(SystemFieldSolver&, const FieldSolveConfig&,
                  FieldNullspaceWorkspace& nullspace) override {
      nullspace.apply_gauge(solver_.phi());
    }
    void reset_diagnostics() override {}
    [[nodiscard]] RuntimeDiagnosticsReport diagnostics_report() const override {
      return make_runtime_diagnostics_report("pops.runtime.direct_elliptic_backend");
    }
    [[nodiscard]] runtime::system::EllipticBackendMetrics metrics() const noexcept override {
      return {};
    }
    [[nodiscard]] const EllipticOperatorContract* operator_contract() const noexcept override {
      return &operator_contract_;
    }
    [[nodiscard]] std::vector<runtime::field::FieldTopologyReportRow> topology_report()
        const override {
      if (topology_digest_.empty() || topology_provenance_.empty())
        return {};
      return {{"builtin:domain", topology_digest_, topology_provenance_, material_points_, 1}};
    }

   private:
    Solver solver_;
    EllipticOperatorContract operator_contract_;
    std::string provider_identity_;
    std::string provider_contract_;
    std::string topology_digest_;
    std::string topology_provenance_;
    std::size_t material_points_ = 0;
  };

  class ExternalNamedFieldBackend final : public NamedFieldBackend {
   public:
    ExternalNamedFieldBackend(
        const Geometry& geometry, const BoxArray& boxes, const DistributionMapping& mapping,
        int rhs_ghosts, int phi_ghosts,
        std::shared_ptr<runtime::field::PreparedFieldSolverComponent> component)
        : geometry_(geometry),
          rhs_(boxes, mapping, 1, rhs_ghosts),
          phi_(boxes, mapping, 1, phi_ghosts),
          component_(std::move(component)) {
      if (!component_)
        throw std::invalid_argument("external named field backend has no component pair");
    }

    [[nodiscard]] std::string_view provider_identity() const noexcept override {
      return component_->provider_identity();
    }
    [[nodiscard]] std::uint64_t provider_version() const noexcept override { return 1; }
    [[nodiscard]] std::string_view provider_contract() const noexcept override {
      return component_->collective_contract();
    }

    MultiFab& rhs() override { return rhs_; }
    MultiFab& phi() override { return phi_; }
    [[nodiscard]] const Geometry& geometry() const noexcept override { return geometry_; }
    [[nodiscard]] FieldDistribution field_distribution() const noexcept override {
      return FieldDistribution::Distributed;
    }
    [[nodiscard]] MultiFab snapshot() override { return MultiFab(phi_); }
    void restore(const MultiFab& value) override { phi_ = value; }
    void configure_boundary(FieldSolveConfig& plan) override {
      if (plan.has_boundary_kernel)
        throw std::runtime_error(
            "System: external FieldSolver v2 cannot consume generated dynamic boundary kernels");
    }
    void prepare_rhs(SystemFieldSolver& owner, MultiFab& value, const FieldSolveConfig&,
                     FieldNullspaceWorkspace& nullspace) override {
      nullspace.require_compatible(value);
      if (owner.p_eps_ != Real(1))
        scale(value, Real(1) / owner.p_eps_);
    }
    SolveReport solve(SystemFieldSolver& owner) override {
      return component_->solve(rhs_, phi_, owner.owner_->geom, owner.owner_->per_);
    }
    void finalize(SystemFieldSolver& owner, const FieldSolveConfig& plan,
                  FieldNullspaceWorkspace& nullspace) override {
      nullspace.apply_gauge(phi_);
      fill_ghosts(phi_, owner.owner_->dom, owner.named_field_bc(plan));
    }
    void reset_diagnostics() override {}
    [[nodiscard]] RuntimeDiagnosticsReport diagnostics_report() const override {
      return make_runtime_diagnostics_report("pops.runtime.external_elliptic_backend");
    }
    [[nodiscard]] runtime::system::EllipticBackendMetrics metrics() const noexcept override {
      return {};
    }
    [[nodiscard]] const EllipticOperatorContract* operator_contract() const noexcept override {
      return nullptr;
    }
    [[nodiscard]] std::vector<runtime::field::FieldTopologyReportRow> topology_report()
        const override {
      return component_->topology_report();
    }

   private:
    Geometry geometry_;
    MultiFab rhs_;
    MultiFab phi_;
    std::shared_ptr<runtime::field::PreparedFieldSolverComponent> component_;
  };

  struct EllipticBackendBuildRequest {
    EllipticBuildRequest elliptic;
    MaterializedDiffusionCoefficient diffusion_coefficient{};
    std::optional<MultiFab> reaction_coefficient;
    std::optional<CompiledFieldBoundaryKernel> dynamic_boundary;
    FieldBoundaryExecutionContext boundary_context{};
    std::optional<FieldNewtonOptions> nonlinear_boundary;
    std::string topology_digest;
    std::string topology_provenance;
    std::size_t material_points = 0;
    bool require_exact_operator_contract = true;
    std::string exact_configuration_contract;
  };

  class EllipticBackendProvider {
   public:
    virtual ~EllipticBackendProvider() = default;
    [[nodiscard]] virtual std::string_view identity() const noexcept = 0;
    [[nodiscard]] virtual std::uint64_t interface_version() const noexcept = 0;
    [[nodiscard]] virtual std::string_view collective_contract() const noexcept = 0;
    /// Opaque declarations used for authentication and inspection only.  The registry never
    /// assigns semantics to an entry; supports(request) is the sole compatibility authority.
    [[nodiscard]] virtual std::vector<std::string> capability_contracts() const = 0;
    [[nodiscard]] virtual PreparedProviderSupport supports(
        const EllipticBackendBuildRequest& request) const noexcept = 0;
    virtual void write_effective_options(EffectivePoissonOptions&) const = 0;
    [[nodiscard]] virtual std::unique_ptr<EllipticBackendProvider> configured(
        std::string identity, const PreparedProviderOptions& options) const = 0;
    [[nodiscard]] virtual std::optional<EllipticOperatorContract> expected_operator_contract(
        const EllipticBuildRequest&) const = 0;
    [[nodiscard]] virtual std::unique_ptr<NamedFieldBackend> prepare(
        EllipticBackendBuildRequest) const = 0;
  };

  class GeometricMgBackendProvider final : public EllipticBackendProvider {
   public:
    GeometricMgBackendProvider(std::string identity, const PreparedProviderOptions& options)
        : identity_(std::move(identity)),
          options_contract_(options.exact_contract()),
          options_(decode_options_(options)) {
      if (identity_.empty())
        throw std::invalid_argument("geometric-MG provider requires an exact identity");
      ExactContractBuilder contract;
      contract.text("pops.runtime.geometric-mg-provider")
          .scalar(std::uint32_t{1})
          .text(identity_)
          .bytes(options_contract_);
      collective_contract_ = std::move(contract).release();
    }

    [[nodiscard]] std::string_view identity() const noexcept override { return identity_; }
    [[nodiscard]] std::uint64_t interface_version() const noexcept override { return 1; }
    [[nodiscard]] std::string_view collective_contract() const noexcept override {
      return collective_contract_;
    }
    [[nodiscard]] std::vector<std::string> capability_contracts() const override { return {}; }
    [[nodiscard]] PreparedProviderSupport supports(
        const EllipticBackendBuildRequest&) const noexcept override {
      return PreparedProviderSupport::accept();
    }

    void write_effective_options(EffectivePoissonOptions& report) const override {
      report.rel_tol = static_cast<double>(options_.rel_tol);
      report.abs_tol = static_cast<double>(options_.abs_tol);
      report.max_cycles = options_.max_cycles;
      report.min_coarse = options_.min_coarse;
      report.pre_smooth = options_.nu1;
      report.post_smooth = options_.nu2;
      report.bottom_sweeps = options_.nbottom;
      report.coarse_threshold = options_.coarse_threshold;
    }

    [[nodiscard]] std::unique_ptr<EllipticBackendProvider> configured(
        std::string identity, const PreparedProviderOptions& options) const override {
      return std::make_unique<GeometricMgBackendProvider>(std::move(identity), options);
    }

    [[nodiscard]] std::optional<EllipticOperatorContract> expected_operator_contract(
        const EllipticBuildRequest& request) const override {
      return GeometricMG::expected_operator_contract(
          request, options_.min_coarse, options_.nu1, options_.nu2, options_.nbottom, false, {},
          kEbCutFractionFloor, options_.coarse_threshold);
    }

    [[nodiscard]] std::unique_ptr<NamedFieldBackend> prepare(
        EllipticBackendBuildRequest request) const override {
      auto backend = std::make_unique<GeometricMgPreparedBackend>(
          std::string(identity()), collective_contract_, options_,
          request.exact_configuration_contract, std::move(request.topology_digest),
          std::move(request.topology_provenance), request.material_points,
          request.elliptic.geometry, request.elliptic.boxes, request.elliptic.mapping,
          request.elliptic.boundary, std::move(request.elliptic.active), options_.min_coarse,
          options_.nu1, options_.nu2, options_.nbottom, options_.coarse_threshold,
          request.elliptic.distribution);
      if (const auto* scalar =
              std::get_if<ScalarDiffusionCoefficient<MultiFab>>(&request.diffusion_coefficient))
        backend->set_scalar_coefficient(scalar->value);
      else if (const auto* diagonal = std::get_if<DiagonalDiffusionCoefficient<MultiFab>>(
                   &request.diffusion_coefficient))
        backend->set_diagonal_tensor_coefficient(diagonal->x, diagonal->y);
      else if (const auto* tensor = std::get_if<FullTensorDiffusionCoefficient<MultiFab>>(
                   &request.diffusion_coefficient))
        backend->set_full_tensor_coefficient(tensor->xx, tensor->xy, tensor->yx, tensor->yy);
      if (request.reaction_coefficient)
        backend->set_reaction_coefficient(*request.reaction_coefficient);
      if (request.dynamic_boundary)
        backend->set_dynamic_boundary(*request.dynamic_boundary, request.boundary_context);
      if (request.nonlinear_boundary)
        backend->set_nonlinear_boundary(*request.nonlinear_boundary);
      return backend;
    }

   private:
    template <class Value>
    [[nodiscard]] static Value option_(const PreparedProviderOptions& options,
                                       std::string_view name) {
      const auto found = options.values.find(std::string(name));
      if (found == options.values.end() || !std::holds_alternative<Value>(found->second))
        throw std::invalid_argument("geometric-MG provider option '" + std::string(name) +
                                    "' is missing or has the wrong exact type");
      return std::get<Value>(found->second);
    }

    [[nodiscard]] static int int_option_(const PreparedProviderOptions& options,
                                         std::string_view name) {
      const std::int64_t value = option_<std::int64_t>(options, name);
      if (value < static_cast<std::int64_t>(std::numeric_limits<int>::min()) ||
          value > static_cast<std::int64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("geometric-MG provider option '" + std::string(name) +
                                    "' is outside the native integer range");
      return static_cast<int>(value);
    }

    [[nodiscard]] static GeometricMgOptions decode_options_(
        const PreparedProviderOptions& options) {
      if (options.schema_identity != "pops.system.geometric-mg-options@1" ||
          options.values.size() != 8)
        throw std::invalid_argument("geometric-MG provider received an incompatible option schema");
      GeometricMgOptions decoded;
      decoded.rel_tol = static_cast<Real>(option_<double>(options, "rel_tol"));
      decoded.abs_tol = static_cast<Real>(option_<double>(options, "abs_tol"));
      decoded.max_cycles = int_option_(options, "max_cycles");
      decoded.min_coarse = int_option_(options, "min_coarse");
      decoded.nu1 = int_option_(options, "pre_smooth");
      decoded.nu2 = int_option_(options, "post_smooth");
      decoded.nbottom = int_option_(options, "bottom_sweeps");
      decoded.coarse_threshold = int_option_(options, "coarse_threshold");
      if (!std::isfinite(static_cast<double>(decoded.rel_tol)) || decoded.rel_tol <= Real(0) ||
          !std::isfinite(static_cast<double>(decoded.abs_tol)) || decoded.abs_tol < Real(0) ||
          decoded.max_cycles < 1 || decoded.min_coarse < 1 || decoded.nu1 < 0 || decoded.nu2 < 0 ||
          decoded.nbottom < 0 || decoded.coarse_threshold < 0)
        throw std::invalid_argument("geometric-MG provider options are outside their exact domain");
      return decoded;
    }

    std::string identity_;
    std::string options_contract_;
    GeometricMgOptions options_;
    std::string collective_contract_;
  };

  template <class Solver>
  static std::unique_ptr<NamedFieldBackend> make_fft_backend_(
      const EllipticBackendBuildRequest& request, std::string_view identity,
      std::string_view provider_contract, bool spectral) {
    return std::make_unique<DirectPreparedBackend<Solver>>(
        std::string(identity), std::string(provider_contract), request.exact_configuration_contract,
        request.topology_digest, request.topology_provenance, request.material_points,
        request.elliptic.geometry, request.elliptic.boxes, request.elliptic.boundary,
        request.elliptic.active, spectral);
  }

  class FftBackendProvider final : public EllipticBackendProvider {
   public:
    FftBackendProvider(std::string identity, const PreparedProviderOptions& options)
        : identity_(std::move(identity)),
          options_contract_(options.exact_contract()),
          spectral_(decode_spectral_(options)) {
      if (identity_.empty())
        throw std::invalid_argument("FFT provider requires an exact identity");
      ExactContractBuilder contract;
      contract.text("pops.runtime.fft-provider")
          .scalar(std::uint32_t{1})
          .text(identity_)
          .bytes(options_contract_);
      collective_contract_ = std::move(contract).release();
    }

    [[nodiscard]] std::string_view identity() const noexcept override { return identity_; }
    [[nodiscard]] std::uint64_t interface_version() const noexcept override { return 1; }
    [[nodiscard]] std::string_view collective_contract() const noexcept override {
      return collective_contract_;
    }
    [[nodiscard]] std::vector<std::string> capability_contracts() const override { return {}; }
    [[nodiscard]] PreparedProviderSupport supports(
        const EllipticBackendBuildRequest& request) const noexcept override {
      if (request.elliptic.active)
        return PreparedProviderSupport::reject(
            1, "FFT provider does not accept an active-region field");
      if (!std::holds_alternative<std::monostate>(request.diffusion_coefficient))
        return PreparedProviderSupport::reject(
            2, "FFT provider does not accept a materialized diffusion coefficient");
      if (request.reaction_coefficient)
        return PreparedProviderSupport::reject(
            4, "FFT provider does not accept a reaction coefficient field");
      if (request.dynamic_boundary)
        return PreparedProviderSupport::reject(
            5, "FFT provider does not accept a generated dynamic boundary");
      if (request.nonlinear_boundary)
        return PreparedProviderSupport::reject(
            6, "FFT provider does not accept a nonlinear boundary solve");
      const BCRec& boundary = request.elliptic.boundary;
      if (boundary.xlo != BCType::Periodic || boundary.xhi != BCType::Periodic ||
          boundary.ylo != BCType::Periodic || boundary.yhi != BCType::Periodic)
        return PreparedProviderSupport::reject(
            7, "FFT provider requires periodic physical boundaries");
      return PreparedProviderSupport::accept();
    }
    void write_effective_options(EffectivePoissonOptions&) const override {}
    [[nodiscard]] std::unique_ptr<EllipticBackendProvider> configured(
        std::string identity, const PreparedProviderOptions& options) const override {
      return std::make_unique<FftBackendProvider>(std::move(identity), options);
    }

    [[nodiscard]] std::optional<EllipticOperatorContract> expected_operator_contract(
        const EllipticBuildRequest& request) const override {
      if (n_ranks() > 1)
        return RemappedFFTSolver::expected_operator_contract(request, spectral_);
      return PoissonFFTSolver::expected_operator_contract(request, spectral_);
    }

    [[nodiscard]] std::unique_ptr<NamedFieldBackend> prepare(
        EllipticBackendBuildRequest request) const override {
      // The provider owns this backend-specific layout choice.  SystemFieldSolver sees one provider
      // identity and one capability contract regardless of communicator size.
      if (n_ranks() > 1)
        return make_fft_backend_<RemappedFFTSolver>(request, identity(), collective_contract(),
                                                    spectral_);
      return make_fft_backend_<PoissonFFTSolver>(request, identity(), collective_contract(),
                                                 spectral_);
    }

   private:
    [[nodiscard]] static bool decode_spectral_(const PreparedProviderOptions& options) {
      const auto spectral = options.values.find("spectral");
      if (options.schema_identity != "pops.system.fft-options@1" || options.values.size() != 1 ||
          spectral == options.values.end() || !std::holds_alternative<bool>(spectral->second))
        throw std::invalid_argument("FFT provider received an incompatible option schema");
      return std::get<bool>(spectral->second);
    }

    std::string identity_;
    std::string options_contract_;
    bool spectral_ = false;
    std::string collective_contract_;
  };

  class ExternalComponentBackendProvider final : public EllipticBackendProvider {
   public:
    explicit ExternalComponentBackendProvider(
        std::shared_ptr<runtime::field::PreparedFieldSolverComponent> component)
        : component_(std::move(component)) {
      if (!component_)
        throw std::invalid_argument("external elliptic provider requires a prepared component");
    }

    [[nodiscard]] std::string_view identity() const noexcept override {
      return component_->provider_identity();
    }
    [[nodiscard]] std::uint64_t interface_version() const noexcept override { return 1; }
    [[nodiscard]] std::string_view collective_contract() const noexcept override {
      return component_->collective_contract();
    }
    [[nodiscard]] std::vector<std::string> capability_contracts() const override { return {}; }
    [[nodiscard]] PreparedProviderSupport supports(
        const EllipticBackendBuildRequest& request) const noexcept override {
      if (request.elliptic.active)
        return PreparedProviderSupport::reject(
            1, "external field component does not accept an active-region field");
      if (!std::holds_alternative<std::monostate>(request.diffusion_coefficient))
        return PreparedProviderSupport::reject(
            2, "external field component does not accept a materialized diffusion coefficient");
      if (request.reaction_coefficient)
        return PreparedProviderSupport::reject(
            4, "external field component does not accept a reaction coefficient field");
      if (request.dynamic_boundary)
        return PreparedProviderSupport::reject(
            5, "external field component does not accept a generated dynamic boundary");
      if (request.nonlinear_boundary)
        return PreparedProviderSupport::reject(
            6, "external field component does not accept a nonlinear boundary solve");
      if (request.require_exact_operator_contract)
        return PreparedProviderSupport::reject(
            7, "external field component does not publish an exact operator contract");
      const BCRec& boundary = request.elliptic.boundary;
      if (boundary.xlo != BCType::Periodic || boundary.xhi != BCType::Periodic ||
          boundary.ylo != BCType::Periodic || boundary.yhi != BCType::Periodic)
        return PreparedProviderSupport::reject(
            8, "external field component requires periodic physical boundaries");
      return PreparedProviderSupport::accept();
    }
    void write_effective_options(EffectivePoissonOptions&) const override {}
    [[nodiscard]] std::unique_ptr<EllipticBackendProvider> configured(
        std::string, const PreparedProviderOptions& options) const override {
      if (options.schema_identity != "pops.system.prepared-provider-reference@1" ||
          !options.values.empty())
        throw std::invalid_argument(
            "external field providers are already exact and accept only a reference schema");
      return std::make_unique<ExternalComponentBackendProvider>(component_);
    }
    [[nodiscard]] std::optional<EllipticOperatorContract> expected_operator_contract(
        const EllipticBuildRequest&) const override {
      return std::nullopt;
    }
    [[nodiscard]] std::unique_ptr<NamedFieldBackend> prepare(
        EllipticBackendBuildRequest request) const override {
      return std::make_unique<ExternalNamedFieldBackend>(
          request.elliptic.geometry, request.elliptic.boxes, request.elliptic.mapping,
          request.elliptic.rhs_ghosts, request.elliptic.phi_ghosts, component_);
    }

   private:
    std::shared_ptr<runtime::field::PreparedFieldSolverComponent> component_;
  };

  class EllipticBackendRegistry {
   public:
    void add(std::unique_ptr<EllipticBackendProvider> provider) {
      if (!provider)
        throw std::invalid_argument("elliptic backend registry requires a provider");
      add(std::string(provider->identity()), std::move(provider));
    }

    void add(std::string route, std::unique_ptr<EllipticBackendProvider> provider) {
      if (route.empty() || !provider || provider->identity().empty() ||
          provider->interface_version() == 0 || provider->collective_contract().empty())
        throw std::invalid_argument("elliptic backend registry requires a named provider");
      (void)exact_provider_declaration_(*provider);
      if (!providers_.emplace(route, std::move(provider)).second)
        throw std::invalid_argument("duplicate elliptic backend provider route '" + route + "'");
    }

    void replace(std::string route, std::unique_ptr<EllipticBackendProvider> provider) {
      if (route.empty() || !provider || provider->identity().empty() ||
          provider->interface_version() == 0 || provider->collective_contract().empty())
        throw std::invalid_argument("elliptic backend registry requires a named provider");
      (void)exact_provider_declaration_(*provider);
      providers_.insert_or_assign(std::move(route), std::move(provider));
    }

    [[nodiscard]] bool contains(std::string_view route) const {
      return providers_.find(std::string(route)) != providers_.end();
    }

    [[nodiscard]] std::string add_configured(std::string_view family_route,
                                             std::string instance_route,
                                             const PreparedProviderOptions& options) {
      const auto family = providers_.find(std::string(family_route));
      if (family == providers_.end())
        throw std::invalid_argument("unknown elliptic provider family route '" +
                                    std::string(family_route) + "'");
      const std::string option_contract = options.exact_contract();
      const std::vector<std::uint8_t> option_bytes(option_contract.begin(), option_contract.end());
      const std::string configured_identity =
          std::string(family->second->identity()) +
          ":configured:sha256:" + identity::sha256_hex(option_bytes);
      auto provider = family->second->configured(configured_identity, options);
      const std::string exact_identity(provider->identity());
      add(std::move(instance_route), std::move(provider));
      return exact_identity;
    }

    void reconfigure(std::string_view route, const PreparedProviderOptions& options) {
      const auto found = providers_.find(std::string(route));
      if (found == providers_.end())
        throw std::invalid_argument("unknown elliptic provider route '" + std::string(route) + "'");
      const std::string identity(found->second->identity());
      auto provider = found->second->configured(identity, options);
      replace(std::string(route), std::move(provider));
    }

    void write_effective_options(std::string_view route, EffectivePoissonOptions& report) const {
      const auto found = providers_.find(std::string(route));
      if (found == providers_.end())
        throw std::invalid_argument("unknown elliptic backend provider route '" +
                                    std::string(route) + "'");
      found->second->write_effective_options(report);
    }

    [[nodiscard]] std::unique_ptr<NamedFieldBackend> prepare(
        std::string_view identity, EllipticBackendBuildRequest request) const {
      request.elliptic.boundary.dx = request.elliptic.geometry.dx();
      request.elliptic.boundary.dy = request.elliptic.geometry.dy();
      const bool valid_configuration_shape =
          !request.nonlinear_boundary || request.dynamic_boundary;
      const long invalid_request = all_reduce_max(
          detail::elliptic_build_request_is_valid(request.elliptic, my_rank(), n_ranks()) &&
                  valid_configuration_shape
              ? 0L
              : 1L);
      if (invalid_request != 0)
        throw std::invalid_argument(
            "elliptic backend registry received an invalid construction request");
      const std::string construction_contract =
          detail::elliptic_build_request_contract(request.elliptic);
      if (!all_ranks_agree_exact_ordered_byte_pairs(
              {{std::string_view("provider-identity"), identity},
               {std::string_view("construction"), construction_contract},
               {std::string_view("configuration"), request.exact_configuration_contract}}))
        throw std::invalid_argument("elliptic provider request differs between communicator ranks");

      const auto found = providers_.find(std::string(identity));
      const long provider_missing = all_reduce_max(found == providers_.end() ? 1L : 0L);
      if (provider_missing != 0)
        throw std::invalid_argument("unknown elliptic backend provider '" + std::string(identity) +
                                    "' (registered: " + registered_identities_() + ")");
      const bool require_exact_operator_contract = request.require_exact_operator_contract;
      std::optional<EllipticOperatorContract> expected_operator_contract;
      PreparedProviderSupport support;
      std::string provider_declaration_contract;
      std::string exact_support_contract;
      std::string support_reason;
      bool local_declaration_failed = false;
      try {
        provider_declaration_contract = exact_provider_declaration_(*found->second);
        support = found->second->supports(request);
        support_reason = std::string(support.reason);
        exact_support_contract = exact_prepared_provider_support(support);
        if (support.accepted()) {
          const auto base_contract = found->second->expected_operator_contract(request.elliptic);
          if (base_contract) {
            if (!base_contract->valid())
              throw std::invalid_argument(
                  "elliptic provider returned an invalid expected operator contract");
            expected_operator_contract = compose_provider_operator_contract_(
                found->second->identity(), *base_contract, request.exact_configuration_contract);
          }
        }
      } catch (...) {
        local_declaration_failed = true;
      }
      if (all_reduce_max(local_declaration_failed ? 1L : 0L) != 0)
        throw std::runtime_error(
            "elliptic provider declaration failed on at least one communicator rank");
      ExactContractBuilder support_declaration;
      support_declaration.bytes(provider_declaration_contract)
          .bytes(exact_support_contract)
          .presence(expected_operator_contract.has_value());
      if (expected_operator_contract)
        support_declaration.bytes(expected_operator_contract->exact_fingerprint());
      const std::string support_declaration_contract = std::move(support_declaration).release();
      if (!all_ranks_agree_exact_ordered_byte_pairs(
              {{std::string_view("provider-support"), support_declaration_contract}}))
        throw std::runtime_error(
            "elliptic provider declaration differs between communicator ranks");
      if (!support.accepted())
        throw std::invalid_argument("elliptic provider '" + std::string(identity) +
                                    "' rejected request (code " + std::to_string(support.code) +
                                    "): " + support_reason);
      if (require_exact_operator_contract && !expected_operator_contract)
        throw std::runtime_error(
            "elliptic provider accepted a request requiring an exact operator contract but "
            "published none");

      const Geometry expected_geometry = request.elliptic.geometry;
      const auto expected_boxes = request.elliptic.boxes.boxes();
      const auto expected_owners = request.elliptic.mapping.ranks();
      const FieldDistribution expected_distribution = request.elliptic.distribution;
      const int expected_rhs_ghosts = request.elliptic.rhs_ghosts;
      const int expected_phi_ghosts = request.elliptic.phi_ghosts;
      std::unique_ptr<NamedFieldBackend> backend;
      bool local_build_failed = false;
      try {
        backend = found->second->prepare(std::move(request));
        local_build_failed = !backend;
      } catch (...) {
        local_build_failed = true;
      }
      if (all_reduce_max(local_build_failed ? 1L : 0L) != 0)
        throw std::runtime_error(
            "elliptic provider construction failed on at least one communicator rank");

      bool local_inspection_failed = false;
      bool local_materialization_mismatch = false;
      std::string actual_identity;
      std::uint64_t actual_provider_version = 0;
      std::string actual_provider_contract;
      std::string actual_operator_contract;
      std::string rhs_layout_contract;
      std::string phi_layout_contract;
      try {
        actual_identity = backend->provider_identity();
        actual_provider_version = backend->provider_version();
        actual_provider_contract = backend->provider_contract();
        MultiFab& rhs = backend->rhs();
        MultiFab& phi = backend->phi();
        const auto* actual_contract = backend->operator_contract();
        const bool actual_contract_present = actual_contract != nullptr;
        if (actual_contract_present)
          actual_operator_contract = std::string(actual_contract->exact_fingerprint());
        const auto field_mismatch = [&](const MultiFab& field, int expected_ghosts) {
          return field.box_array().boxes() != expected_boxes ||
                 field.dmap().ranks() != expected_owners || field.ncomp() != 1 ||
                 field.n_grow() != expected_ghosts ||
                 !detail::field_distribution_layout_matches(field, expected_distribution);
        };
        local_materialization_mismatch =
            actual_contract_present != expected_operator_contract.has_value() ||
            (actual_contract_present &&
             (actual_operator_contract.empty() ||
              actual_operator_contract != expected_operator_contract->exact_fingerprint())) ||
            backend->field_distribution() != expected_distribution ||
            field_mismatch(rhs, expected_rhs_ghosts) || field_mismatch(phi, expected_phi_ghosts) ||
            rhs.shares_storage_with(phi) ||
            !detail::elliptic_geometry_exactly_matches(backend->geometry(), expected_geometry);
        rhs_layout_contract =
            detail::field_distribution_layout_contract(rhs, expected_distribution);
        phi_layout_contract =
            detail::field_distribution_layout_contract(phi, expected_distribution);
        local_materialization_mismatch =
            local_materialization_mismatch || actual_identity != found->second->identity() ||
            actual_provider_version != found->second->interface_version() ||
            actual_provider_contract != found->second->collective_contract();
      } catch (...) {
        local_inspection_failed = true;
      }
      if (all_reduce_max(local_inspection_failed ? 1L : 0L) != 0)
        throw std::runtime_error(
            "elliptic provider inspection failed on at least one communicator rank");
      if (all_reduce_max(local_materialization_mismatch ? 1L : 0L) != 0)
        throw std::runtime_error(
            "elliptic provider materialized a backend that diverges from its exact contract");
      ExactContractBuilder actual_provider_declaration;
      actual_provider_declaration.text(actual_identity)
          .scalar(actual_provider_version)
          .bytes(actual_provider_contract);
      const std::string actual_provider_declaration_contract =
          std::move(actual_provider_declaration).release();
      if (!all_ranks_agree_exact_ordered_byte_pairs(
              {{std::string_view("provider-materialization"), actual_provider_declaration_contract},
               {std::string_view("operator-contract"), actual_operator_contract},
               {std::string_view("rhs-layout"), rhs_layout_contract},
               {std::string_view("phi-layout"), phi_layout_contract}}))
        throw std::runtime_error(
            "elliptic provider materialization differs between communicator ranks");
      return backend;
    }

   private:
    [[nodiscard]] static std::string exact_provider_declaration_(
        const EllipticBackendProvider& provider) {
      if (provider.identity().empty() || provider.interface_version() == 0 ||
          provider.collective_contract().empty())
        throw std::invalid_argument("elliptic backend provider requires exact identities");
      std::vector<std::string> capabilities = provider.capability_contracts();
      std::sort(capabilities.begin(), capabilities.end());
      if (std::any_of(capabilities.begin(), capabilities.end(),
                      [](const std::string& value) { return value.empty(); }) ||
          std::adjacent_find(capabilities.begin(), capabilities.end()) != capabilities.end())
        throw std::invalid_argument(
            "elliptic backend provider capabilities require unique exact identities");
      ExactContractBuilder declaration;
      declaration.text("pops.system.elliptic-backend-provider-declaration")
          .scalar(std::uint32_t{1})
          .text(provider.identity())
          .scalar(provider.interface_version())
          .bytes(provider.collective_contract())
          .sequence(capabilities, [](ExactContractBuilder& item, const std::string& capability) {
            item.text(capability);
          });
      return std::move(declaration).release();
    }

    [[nodiscard]] std::string registered_identities_() const {
      std::string result;
      for (const auto& [identity, provider] : providers_) {
        if (!result.empty())
          result += "|";
        result += identity;
      }
      return result;
    }

    std::map<std::string, std::unique_ptr<EllipticBackendProvider>> providers_;
  };

  [[nodiscard]] static std::unique_ptr<EllipticBackendProvider> geometric_mg_provider(
      std::string identity, const PreparedProviderOptions& options) {
    return std::make_unique<GeometricMgBackendProvider>(std::move(identity), options);
  }

  [[nodiscard]] static PreparedProviderOptions geometric_mg_provider_options(
      const GeometricMgOptions& options) {
    return {"pops.system.geometric-mg-options@1",
            {{"abs_tol", static_cast<double>(options.abs_tol)},
             {"bottom_sweeps", static_cast<std::int64_t>(options.nbottom)},
             {"coarse_threshold", static_cast<std::int64_t>(options.coarse_threshold)},
             {"max_cycles", static_cast<std::int64_t>(options.max_cycles)},
             {"min_coarse", static_cast<std::int64_t>(options.min_coarse)},
             {"post_smooth", static_cast<std::int64_t>(options.nu2)},
             {"pre_smooth", static_cast<std::int64_t>(options.nu1)},
             {"rel_tol", static_cast<double>(options.rel_tol)}}};
  }

  [[nodiscard]] static std::unique_ptr<EllipticBackendProvider> fft_provider(
      std::string identity, const PreparedProviderOptions& options) {
    return std::make_unique<FftBackendProvider>(std::move(identity), options);
  }

  [[nodiscard]] static PreparedProviderOptions fft_provider_options(bool spectral) {
    return {"pops.system.fft-options@1", {{"spectral", spectral}}};
  }

  void register_elliptic_provider(std::string route,
                                  std::unique_ptr<EllipticBackendProvider> provider) {
    elliptic_registry_.add(std::move(route), std::move(provider));
  }

  [[nodiscard]] std::string register_configured_elliptic_provider(
      std::string_view family_route, std::string instance_route,
      const PreparedProviderOptions& options) {
    return elliptic_registry_.add_configured(family_route, std::move(instance_route), options);
  }

  void reconfigure_primary_provider_preset(std::string_view route,
                                           const PreparedProviderOptions& options) {
    for (const auto& entry : named_field_plans_)
      if (entry.second.backend_provider_identity == route)
        throw std::logic_error(
            "cannot reconfigure a primary provider preset referenced by a named field plan");
    elliptic_registry_.reconfigure(route, options);
    if (p_solver == route)
      invalidate_primary_backend_();
  }

  [[nodiscard]] bool has_elliptic_provider(std::string_view route) const {
    return elliptic_registry_.contains(route);
  }

  void write_effective_poisson_options(EffectivePoissonOptions& report) const {
    elliptic_registry_.write_effective_options(p_solver, report);
  }

  void install_builtin_provider_presets_() {
    const GeometricMgOptions defaults;
    elliptic_registry_.add(
        "geometric_mg",
        geometric_mg_provider("pops.system.geometric-mg", geometric_mg_provider_options(defaults)));
    elliptic_registry_.add("fft", fft_provider("pops.system.fft", fft_provider_options(false)));
    elliptic_registry_.add("fft_spectral",
                           fft_provider("pops.system.fft-spectral", fft_provider_options(true)));
  }

  struct NamedField {
    struct PreparedProvider {
      int block = -1;
      Real coefficient = Real(1);
      std::function<void(const MultiFab&, MultiFab&)> rhs;
    };
    int phi_comp = -1;
    int gx_comp = -1;
    int gy_comp = -1;
    int gradient_sign = 0;
    bool has_plan = false;
    FieldSolveConfig plan{};
    std::vector<PreparedProvider> prepared_providers;
    std::unique_ptr<NamedFieldBackend> backend;
    std::optional<MultiFab> contribution_scratch;
    std::optional<MultiFab> published_phi_scratch;
    bool nullspace_ready = false;
    std::unique_ptr<FieldNullspaceWorkspace> nullspace_workspace;
  };

  static void invalidate_named_backend_(NamedField& field) {
    field.backend.reset();
    field.nullspace_ready = false;
    field.nullspace_workspace.reset();
  }

  void invalidate_primary_nullspace_() {
    p_nullspace_ = {};
    p_nullspace_ready_ = false;
    p_nullspace_workspace_.reset();
  }

  void invalidate_primary_backend_() {
    ell_.reset();
    invalidate_primary_nullspace_();
  }
  std::map<std::string, NamedField> named_fields_;
  // Field plans are installed before compiled block loaders register their output components.
  // Keep the exact per-provider plan independently, then attach it when register_named_field runs.
  std::map<std::string, FieldSolveConfig> named_field_plans_;
  std::map<std::string, std::shared_ptr<runtime::field::PreparedFieldSolverComponent>>
      external_field_components_;
  EllipticBackendRegistry elliptic_registry_;
  std::shared_ptr<FieldNullspaceProviderRegistry> nullspace_provider_registry_;

  /// Invalidate the collective witness after a new plan is added while the low-level C++ facade is
  /// still assembling.  Python normally validates once in System::mark_bound; the lazy-backend guard
  /// below also covers direct C++ users that intentionally never bind.
  void invalidate_field_plan_consensus() { field_plan_consensus_verified_ = false; }

  static void append_boundary_contract_(ExactContractBuilder& contract, const BCRec& boundary) {
    contract.scalar(boundary.xlo)
        .scalar(boundary.xhi)
        .scalar(boundary.ylo)
        .scalar(boundary.yhi)
        .scalar(boundary.xlo_val)
        .scalar(boundary.xhi_val)
        .scalar(boundary.ylo_val)
        .scalar(boundary.yhi_val)
        .scalar(boundary.xlo_alpha)
        .scalar(boundary.xlo_beta)
        .scalar(boundary.xhi_alpha)
        .scalar(boundary.xhi_beta)
        .scalar(boundary.ylo_alpha)
        .scalar(boundary.ylo_beta)
        .scalar(boundary.yhi_alpha)
        .scalar(boundary.yhi_beta)
        .scalar(boundary.dx)
        .scalar(boundary.dy);
  }

  [[nodiscard]] static std::string exact_field_plan_contract_(const FieldSolveConfig& plan) {
    ExactContractBuilder contract;
    contract.text("pops.system.native-field-plan")
        .scalar(std::uint32_t{1})
        .text(plan.provider_identity)
        .text(plan.plan_identity)
        .text(plan.output_owner_identity)
        .text(plan.output_block)
        .text(plan.output_key)
        .sequence(plan.providers,
                  [](ExactContractBuilder& item, const FieldProviderBinding& provider) {
                    item.text(provider.identity)
                        .text(provider.owner_block)
                        .text(provider.native_key)
                        .scalar(provider.coefficient);
                  })
        .text(plan.backend_provider_identity)
        .text(plan.bc)
        .presence(plan.has_explicit_bc);
    if (plan.has_explicit_bc)
      append_boundary_contract_(contract, plan.explicit_bc);
    contract.presence(plan.has_boundary_kernel);
    if (plan.has_boundary_kernel)
      contract.text(plan.boundary_kernel.identity)
          .text(plan.boundary_kernel.residual_identity)
          .text(plan.boundary_kernel.jvp_identity)
          .scalar(plan.boundary_kernel.observes_iteration);
    if (plan.boundary_parameters)
      contract.sequence(*plan.boundary_parameters);
    else
      contract.sequence(std::vector<Real>{});
    const auto append_text = [](ExactContractBuilder& item, const std::string& value) {
      item.text(value);
    };
    contract.sequence(plan.boundary_state_blocks, append_text)
        .sequence(plan.boundary_state_components)
        .sequence(plan.boundary_field_blocks, append_text)
        .sequence(plan.boundary_field_keys, append_text)
        .sequence(plan.boundary_field_components)
        .presence(plan.has_reaction);
    if (plan.has_reaction)
      contract.scalar(plan.reaction);
    contract.presence(plan.has_newton);
    if (plan.has_newton)
      contract.scalar(plan.newton.tolerance)
          .scalar(static_cast<std::int64_t>(plan.newton.max_iterations))
          .scalar(plan.newton.linear_tolerance)
          .scalar(static_cast<std::int64_t>(plan.newton.linear_max_iterations))
          .scalar(static_cast<std::int64_t>(plan.newton.restart))
          .scalar(plan.newton.armijo)
          .scalar(plan.newton.minimum_step);
    contract.bytes(plan.nullspace_provider.exact_contract())
        .text(plan.topology_provider_kind)
        .text(plan.topology_provenance)
        .text(plan.topology_digest);
    return std::move(contract).release();
  }

  /// Require exact rank agreement over the complete canonical field-plan registry.  std::map gives
  /// insertion-order-independent ordering, and the collective helper agrees every pair/component
  /// length before touching bytes, so missing plans cannot strand another rank in a payload call.
  void require_field_plan_consensus() {
    if (field_plan_consensus_verified_)
      return;
    ExactContractBuilder registry_contract;
    registry_contract.text("pops.system.field-plan-registry").scalar(std::uint32_t{1});
    for (const auto& [slot, plan] : named_field_plans_)
      registry_contract.text(slot).bytes(exact_field_plan_contract_(plan));
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"system-field-plan-registry", registry_contract.view()}}))
      throw std::runtime_error("System: ordered resolved field plans differ across MPI ranks");
    field_plan_consensus_verified_ = true;
  }

  std::vector<std::string> provider_slots() const {
    std::vector<std::string> result;
    result.reserve(named_fields_.size());
    for (const auto& item : named_fields_)
      result.push_back(item.first);
    return result;
  }

  MultiFab& provider_potential(const std::string& slot) {
    require_field_plan_consensus();
    auto found = named_fields_.find(slot);
    if (found == named_fields_.end())
      throw std::runtime_error("System: unknown qualified field provider slot '" + slot + "'");
    ensure_named_backend(found->second, slot);
    return found->second.backend->phi();
  }

  void set_topology_authority(const std::string& slot, const std::string& provider_kind,
                              const std::string& provenance, const std::string& digest) {
    if (provider_kind.empty() || provenance.empty() || digest.empty())
      throw std::invalid_argument("field topology authority is incomplete");
    auto found = named_field_plans_.find(slot);
    if (found == named_field_plans_.end())
      throw std::runtime_error("field topology authority names an unknown provider slot");
    found->second.topology_provider_kind = provider_kind;
    found->second.topology_provenance = provenance;
    found->second.topology_digest = digest;
    auto registered = named_fields_.find(slot);
    if (registered != named_fields_.end()) {
      registered->second.plan = found->second;
      invalidate_named_backend_(registered->second);
    }
    invalidate_field_plan_consensus();
  }

  void set_named_reaction(const std::string& slot, Real reaction) {
    if (!std::isfinite(static_cast<double>(reaction)) || reaction <= Real(0))
      throw std::invalid_argument(
          "screened field reaction coefficient must be finite and strictly positive");
    auto found = named_field_plans_.find(slot);
    if (found == named_field_plans_.end())
      throw std::runtime_error("screened field reaction names an unknown provider slot");
    found->second.has_reaction = true;
    found->second.reaction = reaction;
    auto registered = named_fields_.find(slot);
    if (registered != named_fields_.end()) {
      registered->second.plan = found->second;
      invalidate_named_backend_(registered->second);
    }
    invalidate_field_plan_consensus();
  }

  void register_field_nullspace_provider(std::shared_ptr<const FieldNullspaceProvider> provider) {
    nullspace_provider_registry_->add(std::move(provider));
  }

  void set_primary_nullspace_provider(FieldNullspaceProviderSelection selection) {
    const auto provider = nullspace_provider_registry_->resolve(selection.provider_identity);
    if (!provider->accepts_options(selection.options))
      throw std::invalid_argument("field-nullspace provider rejected its exact options");
    (void)selection.exact_contract();
    p_nullspace_provider_ = std::move(selection);
    invalidate_primary_nullspace_();
  }

  void set_named_nullspace_provider(const std::string& slot,
                                    FieldNullspaceProviderSelection selection) {
    auto found = named_field_plans_.find(slot);
    if (found == named_field_plans_.end())
      throw std::runtime_error("field nullspace names an unknown provider slot");
    const auto provider = nullspace_provider_registry_->resolve(selection.provider_identity);
    if (!provider->accepts_options(selection.options))
      throw std::invalid_argument("field-nullspace provider rejected its exact options");
    (void)selection.exact_contract();
    found->second.nullspace_provider = std::move(selection);
    found->second.nullspace = {};
    auto registered = named_fields_.find(slot);
    if (registered != named_fields_.end()) {
      registered->second.has_plan = true;
      registered->second.plan = found->second;
      invalidate_named_backend_(registered->second);
    }
    invalidate_field_plan_consensus();
  }

  [[nodiscard]] PreparedFieldNullspace prepare_field_nullspace_(
      const FieldNullspaceProviderSelection& selection,
      FieldNullspaceProviderRequest request) const {
    return prepare_field_nullspace_collectively(*nullspace_provider_registry_, selection,
                                                std::move(request));
  }

  [[nodiscard]] static std::unique_ptr<FieldNullspaceWorkspace> make_nullspace_workspace_(
      const FieldNullspacePlan& plan, const MultiFab& layout, FieldDistribution distribution) {
    return std::make_unique<FieldNullspaceWorkspace>(
        plan, std::vector<const MultiFab*>{&layout},
        std::vector<PreparedVectorDistribution>{PreparedVectorDistribution(distribution)});
  }

  void prepare_primary_nullspace_workspace_() {
    if (!ell_)
      throw std::logic_error("primary nullspace workspace requires a prepared elliptic backend");
    const FieldDistribution distribution = ell_->field_distribution();
    p_nullspace_workspace_ = make_nullspace_workspace_(p_nullspace_, ell_->phi(), distribution);
  }

  [[nodiscard]] FieldNullspaceProviderRequest uniform_nullspace_request_(
      std::string plan_identity, std::string topology_identity, const BCRec& boundary,
      bool has_reaction, bool has_internal_constraint, NamedFieldBackend& backend) const {
    FieldNullspaceProviderRequest request;
    request.plan_identity = std::move(plan_identity);
    request.operator_facts =
        field_nullspace_operator_facts_from_bc_rec(boundary, has_reaction, has_internal_constraint);
    request.topology.identity = std::move(topology_identity);
    request.topology.exact_layout_contract =
        detail::field_distribution_layout_contract(backend.phi(), backend.field_distribution());
    request.topology.field_component = 0;
    if (!has_internal_constraint)
      request.topology.connected_component_contract =
          request.topology.identity + ":connected-component@1";
    request.topology.layouts = {&backend.phi()};
    request.topology.cell_measure = {backend.geometry().dx() * backend.geometry().dy()};
    request.topology.level_distributions = {
        PreparedVectorDistribution(backend.field_distribution())};
    return request;
  }

  void prepare_named_nullspace_(NamedField& field) {
    if (field.nullspace_ready)
      return;
    if (!field.backend)
      throw std::logic_error("named field nullspace requires a prepared elliptic backend");
    const std::string topology_identity =
        (field.plan.topology_digest.empty() ? field.plan.plan_identity
                                            : field.plan.topology_digest) +
        ":uniform-layout";
    auto request = uniform_nullspace_request_(field.plan.provider_identity + ":topology-nullspace",
                                              topology_identity, named_field_bc(field.plan),
                                              field.plan.has_reaction, false, *field.backend);
    field.plan.nullspace =
        prepare_field_nullspace_(field.plan.nullspace_provider, std::move(request)).plan;
    const FieldDistribution distribution = field.backend->field_distribution();
    field.nullspace_workspace =
        make_nullspace_workspace_(field.plan.nullspace, field.backend->phi(), distribution);
    field.nullspace_ready = true;
  }

  std::string register_external_solver_provider(
      const std::string& slot, runtime::field::PreparedFieldSolverSpec spec,
      std::shared_ptr<component::LoadedComponent> topology,
      std::shared_ptr<component::LoadedComponent> solver) {
    auto found = named_field_plans_.find(slot);
    if (spec.provider_slot != slot)
      throw std::invalid_argument("external field solver provider slot mismatch");
    auto component = std::make_shared<runtime::field::PreparedFieldSolverComponent>(
        std::move(spec), std::move(topology), std::move(solver));
    register_elliptic_provider(slot, std::make_unique<ExternalComponentBackendProvider>(component));
    external_field_components_[slot] = component;
    if (found == named_field_plans_.end())
      return std::string(component->provider_identity());
    found->second.backend_provider_identity = slot;
    found->second.topology_provider_kind = std::string(component->provider_identity());
    found->second.topology_provenance = "pending-component-materialization";
    found->second.topology_digest.clear();
    auto registered = named_fields_.find(slot);
    if (registered != named_fields_.end()) {
      registered->second.plan = found->second;
      invalidate_named_backend_(registered->second);
    }
    invalidate_field_plan_consensus();
    return std::string(component->provider_identity());
  }

  std::vector<runtime::field::FieldTopologyReportRow> topology_report(
      const std::string& slot) const {
    auto field = named_fields_.find(slot);
    if (field != named_fields_.end() && field->second.backend)
      return field->second.backend->topology_report();
    auto external = external_field_components_.find(slot);
    if (external != external_field_components_.end())
      return external->second->topology_report();
    if (named_field_plans_.find(slot) == named_field_plans_.end())
      throw std::runtime_error("unknown qualified field provider slot");
    return {};
  }

  void prepare_boundary_dependencies(NamedField& field, int stage_block,
                                     const MultiFab* stage_state) {
    auto& plan = field.plan;
    plan.boundary_state_buffers.clear();
    plan.boundary_state_buffers.reserve(plan.boundary_state_blocks.size());
    plan.boundary_state_distributions.clear();
    plan.boundary_state_distributions.reserve(plan.boundary_state_blocks.size());
    for (std::size_t index = 0; index < plan.boundary_state_blocks.size(); ++index) {
      const int block = owner_->index(plan.boundary_state_blocks[index]);
      const MultiFab* value = (stage_state != nullptr && block == stage_block)
                                  ? stage_state
                                  : &owner_->sp[static_cast<std::size_t>(block)].U;
      if (plan.boundary_state_components[index] < 0 ||
          plan.boundary_state_components[index] >= value->ncomp())
        throw std::runtime_error("System: boundary state dependency component is out of range");
      plan.boundary_state_buffers.push_back(value);
      plan.boundary_state_distributions.push_back(FieldDistribution::Distributed);
    }
    plan.boundary_field_buffers.clear();
    plan.boundary_field_buffers.reserve(plan.boundary_field_keys.size());
    plan.boundary_field_distributions.clear();
    plan.boundary_field_distributions.reserve(plan.boundary_field_keys.size());
    for (std::size_t index = 0; index < plan.boundary_field_keys.size(); ++index) {
      auto dependency =
          std::find_if(named_fields_.begin(), named_fields_.end(), [&](const auto& item) {
            return item.second.plan.output_block == plan.boundary_field_blocks[index] &&
                   item.second.plan.output_key == plan.boundary_field_keys[index];
          });
      if (dependency == named_fields_.end() || &dependency->second == &field)
        throw std::runtime_error(
            "System: boundary field dependency is missing or recursively reads its own solve");
      MultiFab& value = provider_potential(dependency->first);
      if (plan.boundary_field_components[index] < 0 ||
          plan.boundary_field_components[index] >= value.ncomp())
        throw std::runtime_error("System: boundary field dependency component is out of range");
      plan.boundary_field_buffers.push_back(&value);
      plan.boundary_field_distributions.push_back(FieldDistribution::Distributed);
    }
    plan.boundary_context.states =
        plan.boundary_state_buffers.empty() ? nullptr : plan.boundary_state_buffers.data();
    plan.boundary_context.state_distributions = plan.boundary_state_distributions.empty()
                                                    ? nullptr
                                                    : plan.boundary_state_distributions.data();
    plan.boundary_context.state_count = static_cast<int>(plan.boundary_state_buffers.size());
    plan.boundary_context.fields =
        plan.boundary_field_buffers.empty() ? nullptr : plan.boundary_field_buffers.data();
    plan.boundary_context.field_distributions = plan.boundary_field_distributions.empty()
                                                    ? nullptr
                                                    : plan.boundary_field_distributions.data();
    plan.boundary_context.field_count = static_cast<int>(plan.boundary_field_buffers.size());
  }

  void prepare_field_providers(NamedField& field) {
    if (!field.prepared_providers.empty())
      return;
    std::vector<typename NamedField::PreparedProvider> prepared;
    prepared.reserve(field.plan.providers.size());
    for (const auto& binding : field.plan.providers) {
      const int block = owner_->index(binding.owner_block);
      auto& state = owner_->sp[static_cast<std::size_t>(block)];
      auto found = state.named_poisson_rhs.find(binding.native_key);
      if (found == state.named_poisson_rhs.end() || !found->second)
        throw std::runtime_error("System: authenticated field provider has no RHS closure");
      prepared.push_back({block, binding.coefficient, found->second});
    }
    field.prepared_providers = std::move(prepared);
  }

  /// Register a named elliptic field (ADC-428): records the aux output components (where the field's
  /// solved phi and centered gradient land). @p gx_comp / @p gy_comp both equal -1 for phi-only;
  /// otherwise @p gradient_sign (exactly -1 or +1) scales both centered derivatives. Idempotent
  /// (re-register overwrites the
  /// component map, drops the lazily-built solver so the next solve rebuilds it). The DEDICATED solver
  /// is built on first solve, never here.
  void register_named_field(const std::string& block, const std::string& provider_key, int phi_comp,
                            int gx_comp, int gy_comp, int gradient_sign) {
    const bool has_gradient = gx_comp >= 0 && gy_comp >= 0;
    const bool has_no_gradient = gx_comp == -1 && gy_comp == -1;
    if (phi_comp < 0 || (!has_gradient && !has_no_gradient) ||
        (has_gradient && (phi_comp == gx_comp || phi_comp == gy_comp || gx_comp == gy_comp)))
      throw std::invalid_argument(
          "System: named elliptic field output components must be one potential or "
          "three distinct potential/gradient components");
    if (gradient_sign != -1 && gradient_sign != 1)
      throw std::invalid_argument(
          "System: named elliptic field gradient sign must be exactly -1 or 1");
    if (!has_gradient && gradient_sign != 1)
      throw std::invalid_argument(
          "System: a named elliptic field without gradient outputs must use sign +1");
    for (const auto& configured : named_field_plans_) {
      if (configured.second.output_block != block || configured.second.output_key != provider_key)
        continue;
      NamedField nf;
      nf.phi_comp = phi_comp;
      nf.gx_comp = gx_comp;
      nf.gy_comp = gy_comp;
      nf.gradient_sign = gradient_sign;
      nf.has_plan = true;
      nf.plan = configured.second;
      named_fields_[configured.first] = std::move(nf);
    }
  }

  /// Re-applies the per-field aux HALO policies (ADC-369) onto the shared channel, AFTER the shared
  /// fill_ghosts/fill_boundary. For each declared component, overrides ONLY that component's
  /// physical-face ghosts (aux_halo_override keeps periodic faces periodic). No-op when empty.
  void apply_named_aux_bc() {
    if (named_aux_bc_.empty())
      return;  // hot-path fast exit (parity with the AMR counterparts)
    for (const auto& kv : named_aux_bc_) {
      if (kv.first >= owner_->aux_ncomp_)
        continue;
      fill_physical_bc(owner_->aux, owner_->dom, aux_halo_override(owner_->bc_, kv.second),
                       kv.first);
    }
  }

  /// Populates the B_z component (index kAuxBaseComps) of the shared channel from bz_field_, over the
  /// valid cells. No-op if B_z not provided or if no block reads it (base width). The
  /// halos of B_z are filled by solve_fields (like grad); field_postprocess only writes comp 0..2.
  void apply_bz() {
    if (bz_field_.empty() || owner_->aux_ncomp_ <= kAuxBaseComps)
      return;
    // ROW WIDTH (fast axis i) of the row-major array bz_field_: n in cartesian (square n x n,
    // BIT-IDENTICAL), nr in POLAR (ring nr x ntheta, i = r of size nr, cf. set_magnetic_field).
    // The index stays flat[j * row + i]: in cartesian row == n (unchanged); in polar row == nr.
    const int row = owner_->polar_ ? owner_->aux.box(0).nx() : owner_->cfg.n;
    // LOCAL population on the owner rank (cf. solve_fields): iteration over the local fabs of the
    // aux channel instead of fab(0) hardcoded (no-op on a rank without a local box at np>1, bit-identical to the
    // owner).
    sync_device();
    for (int li = 0; li < owner_->aux.local_size(); ++li) {
      Array4 a = owner_->aux.fab(li).array();
      const Box2D v = owner_->aux.box(li);
      for_each_cell(v, detail::SystemNamedAuxCopyKernel{a, bz_field_.data(), kAuxBaseComps, row,
                                                        owner_->dom.lo[0], owner_->dom.lo[1]});
    }
  }

  /// Populates the T_e component (electron temperature) = p/rho of the fluid block source te_src_.
  /// RECOMPUTED on each solve_fields (T_e varies with the fluid, unlike the static B_z).
  /// No-op if no source or if no block reads T_e (insufficient width). The source block is
  /// compressible (4 var); p = (gamma-1)(E - 0.5 rho|v|^2), T = p/rho.
  void apply_te() {
    if (te_src_ < 0 || owner_->aux_ncomp_ <= kTeComp)
      return;
    const auto& s = owner_->sp[static_cast<std::size_t>(te_src_)];
    const Real gm1 = Real(s.gamma) - Real(1);
    // LOCAL population on the owner rank (cf. solve_fields): we iterate over the local fabs of the
    // aux channel instead of fab(0) hardcoded (no-op on a rank without a local box at np>1, bit-identical to the
    // owner). s.U and aux share the same DistributionMapping -> same local indexing.
    for (int li = 0; li < owner_->aux.local_size(); ++li) {
      const ConstArray4 us = s.U.fab(li).const_array();
      Array4 a = owner_->aux.fab(li).array();
      const Box2D v = owner_->aux.box(li);
      for_each_cell(v, detail::SystemElectronTemperatureKernel{a, us, gm1, kTeComp});
    }
  }

  /// Populates ONE NAMED aux component (canonical index @p comp >= kAuxNamedBase) of the shared channel
  /// from @p field (row-major), over the valid cells. No-op if the channel is too narrow
  /// (no block reads this component) or if the field is empty. SAME pattern as apply_bz: STATIC field
  /// provided by the user, never rewritten by solve_fields; its halos are filled by
  /// solve_fields (fill_ghosts/fill_boundary over the whole channel). LOCAL population on the rank (iteration
  /// over the local fabs, no-op on a rank without a box at np>1).
  void apply_named_aux_one(int comp, const NamedAuxField& field) {
    if (field.empty() || owner_->aux_ncomp_ <= comp)
      return;
    // ROW WIDTH (fast axis i): n in cartesian (square n x n), nr in polar (ring nr x
    // ntheta). Index flat[j * row + i], identical to apply_bz / set_density.
    const int row = owner_->polar_ ? owner_->aux.box(0).nx() : owner_->cfg.n;
    // NamedAuxField uses fab_allocator, hence Kokkos SharedSpace on native builds. Host population
    // completed before this call; declare the device residency before publishing the pointer to a
    // kernel. The current unified-memory implementation is a no-op, while preserving the explicit
    // deep-copy seam required by a future split-residency allocator.
    sync_device();
    for (int li = 0; li < owner_->aux.local_size(); ++li) {
      Array4 a = owner_->aux.fab(li).array();
      const Box2D v = owner_->aux.box(li);
      for_each_cell(v, detail::SystemNamedAuxCopyKernel{a, field.data(), comp, row,
                                                        owner_->dom.lo[0], owner_->dom.lo[1]});
    }
  }

  /// Publish a solved Cartesian scalar potential and its centered gradient into the shared aux
  /// channel.  This single Kokkos route is shared by the ordinary and simultaneous multi-block
  /// solves so neither path can regress to host cell loops independently.
  void derive_cartesian_aux(const MultiFab& phi) {
    const Real dx = owner_->geom.dx();
    const Real dy = owner_->geom.dy();
    for (int li = 0; li < owner_->aux.local_size(); ++li) {
      const ConstArray4 p = phi.fab(li).const_array();
      Array4 a = owner_->aux.fab(li).array();
      const Box2D valid = owner_->aux.box(li);
      for_each_cell(valid, detail::SystemNamedFieldPostprocessKernel{
                               a, p, 0, 1, 2, Real(1), dx, dy, true});
    }
  }

  /// Re-applies ALL the stored named aux fields (cf. named_aux_). Called by ensure_aux_width
  /// after a reallocation of the aux channel (which starts again from a zeroed MultiFab), like apply_bz / apply_te.
  void apply_named_aux() {
    for (const auto& kv : named_aux_)
      apply_named_aux_one(kv.first, kv.second);
  }

  // --- elliptic solver (system Poisson) -----------------------------
  /// Resolves the BC mode into a BCRec. ``auto`` preserves periodic axes and applies a physical
  /// Dirichlet condition only on the remaining axes.
  /// "periodic"|"dirichlet"|"neumann" (Foextrap). @throws std::runtime_error on an unknown mode.
  BCRec topology_bc(BCType physical) const {
    BCRec result;
    if (!owner_->per_.x)
      result.xlo = result.xhi = physical;
    if (!owner_->per_.y)
      result.ylo = result.yhi = physical;
    return result;
  }

  BCRec poisson_bc() {
    if (p_has_explicit_bc)
      return p_explicit_bc;
    std::string mode = p_bc;
    if (mode == "auto")
      return topology_bc(BCType::Dirichlet);
    BCRec b;
    if (mode == "periodic")
      return b;
    if (mode == "dirichlet") {
      b.xlo = b.xhi = b.ylo = b.yhi = BCType::Dirichlet;
      return b;
    }
    if (mode == "neumann") {
      b.xlo = b.xhi = b.ylo = b.yhi = BCType::Foextrap;
      return b;
    }
    throw std::runtime_error("System::set_poisson: unknown bc '" + mode + "'");
  }

  BCRec named_field_bc(const FieldSolveConfig& plan) const {
    if (plan.has_explicit_bc)
      return plan.explicit_bc;
    std::string mode = plan.bc;
    if (mode == "auto")
      return topology_bc(BCType::Dirichlet);
    BCRec result;
    if (mode == "periodic")
      return result;
    if (mode == "dirichlet") {
      result.xlo = result.xhi = result.ylo = result.yhi = BCType::Dirichlet;
      return result;
    }
    if (mode == "neumann") {
      result.xlo = result.xhi = result.ylo = result.yhi = BCType::Foextrap;
      return result;
    }
    throw std::runtime_error("System: named field plan has unknown bc '" + mode + "'");
  }
  /// "Conductor interior" predicate from p_wall / p_wall_radius / cfg.L (cf. wall_predicate);
  /// empty if no wall.
  ActiveRegionProvider2D wall_active() {
    return detail::wall_predicate(p_wall, p_wall_radius, owner_->cfg.L, "System::set_poisson",
                                  owner_->cfg.xlo, owner_->cfg.ylo);
  }
  [[nodiscard]] MultiFab materialize_system_coefficient_(const std::vector<double>& values) const {
    const std::size_t expected = static_cast<std::size_t>(owner_->cfg.n) * owner_->cfg.n;
    if (values.size() != expected)
      throw std::invalid_argument("system elliptic coefficient has invalid materialized extent");
    MultiFab field(owner_->ba, owner_->dm, 1, 0);
    const int n = owner_->cfg.n;
    NamedAuxField device_values(values.begin(), values.end());
    sync_device();
    for (int local = 0; local < field.local_size(); ++local) {
      Array4 output = field.fab(local).array();
      const Box2D valid = field.box(local);
      for_each_cell(valid, detail::SystemNamedAuxCopyKernel{
                               output, device_values.data(), 0, n, owner_->dom.lo[0],
                               owner_->dom.lo[1]});
    }
    device_fence();
    return field;
  }

  /// Builds the cartesian elliptic backend lazily through the prepared provider registry.  The
  /// provider identity selects an implementation; the core only materializes physical inputs and
  /// submits one complete typed request before construction.
  void ensure_elliptic() {
    const bool operator_has_reaction =
        has_kappa_field_ && std::any_of(p_kappa_field_.begin(), p_kappa_field_.end(),
                                        [](double value) { return value > 0.0; });
    if (ell_ && p_nullspace_ready_)
      return;
    if (ell_) {
      const bool has_internal_constraint = static_cast<bool>(wall_active());
      auto request =
          uniform_nullspace_request_("pops://system/default-field/nullspace-plan@1",
                                     "pops://system/default-field/uniform-layout@1", poisson_bc(),
                                     operator_has_reaction, has_internal_constraint, *ell_);
      p_nullspace_ = prepare_field_nullspace_(p_nullspace_provider_, std::move(request)).plan;
      prepare_primary_nullspace_workspace_();
      p_nullspace_ready_ = true;
      return;
    }
    // The system right-hand side is ALWAYS f = Sum_s elliptic_rhs_s(u_s), assembled by
    // solve_fields from the elliptic brick of EACH block (charge q n, background alpha (n-n0),
    // gravity coupling 4piG (rho-rho0)). The token is thus NOT a computation mode but a LABEL
    // of this composite right-hand side. "composite" names this behavior honestly; "charge_density"
    // stays the historical alias (default, bit-identical) since the usual case is a charge block.
    if (p_rhs != "charge_density" && p_rhs != "composite")
      throw std::runtime_error("System::set_poisson: unknown rhs '" + p_rhs +
                               "' (valid: " + kPoissonRhsRouteTokensCsv +
                               "; the right-hand side = sum of the "
                               "per-block elliptic bricks)");
    if (has_kappa_field_ && !has_variable_diffusion_coefficient() && p_eps_ != Real(1))
      throw std::runtime_error(
          "System: reaction term kappa(x) + constant permittivity eps != 1 is inconsistent; "
          "use eps = 1 or a materialized coefficient field");

    ActiveRegionProvider2D active_region = wall_active();
    const bool has_internal_constraint = static_cast<bool>(active_region);
    EllipticBackendBuildRequest request;
    request.elliptic = {owner_->geom,
                        owner_->ba,
                        owner_->dm,
                        poisson_bc(),
                        std::move(active_region),
                        FieldDistribution::Distributed,
                        0,
                        1};
    request.material_points = static_cast<std::size_t>(owner_->dom.num_cells());
    ExactContractBuilder configuration;
    configuration.text("pops.runtime.system-elliptic-configuration")
        .scalar(std::uint32_t{2})
        .scalar(p_eps_)
        .scalar(static_cast<std::uint32_t>(p_diffusion_coefficient_.index()));
    if (const auto* scalar =
            std::get_if<ScalarDiffusionCoefficient<std::vector<double>>>(&p_diffusion_coefficient_))
      configuration.sequence(scalar->value);
    else if (const auto* diagonal = std::get_if<DiagonalDiffusionCoefficient<std::vector<double>>>(
                 &p_diffusion_coefficient_))
      configuration.sequence(diagonal->x).sequence(diagonal->y);
    configuration.presence(has_kappa_field_);
    if (has_kappa_field_)
      configuration.sequence(p_kappa_field_);
    request.exact_configuration_contract = std::move(configuration).release();
    if (const auto* scalar =
            std::get_if<ScalarDiffusionCoefficient<std::vector<double>>>(&p_diffusion_coefficient_))
      request.diffusion_coefficient =
          ScalarDiffusionCoefficient<MultiFab>{materialize_system_coefficient_(scalar->value)};
    else if (const auto* diagonal = std::get_if<DiagonalDiffusionCoefficient<std::vector<double>>>(
                 &p_diffusion_coefficient_))
      request.diffusion_coefficient =
          DiagonalDiffusionCoefficient<MultiFab>{materialize_system_coefficient_(diagonal->x),
                                                 materialize_system_coefficient_(diagonal->y)};
    if (has_kappa_field_)
      request.reaction_coefficient = materialize_system_coefficient_(p_kappa_field_);

    ell_ = elliptic_registry_.prepare(p_solver, std::move(request));
    auto nullspace_request =
        uniform_nullspace_request_("pops://system/default-field/nullspace-plan@1",
                                   "pops://system/default-field/uniform-layout@1", poisson_bc(),
                                   operator_has_reaction, has_internal_constraint, *ell_);
    p_nullspace_ =
        prepare_field_nullspace_(p_nullspace_provider_, std::move(nullspace_request)).plan;
    prepare_primary_nullspace_workspace_();
    p_nullspace_ready_ = true;
  }
  MultiFab& ell_rhs() { return ell_->rhs(); }
  /// Ghosted potential used by field post-processing and generated Programs. Cartesian returns the
  /// active elliptic solver's field. Polar materializes a dedicated one-ghost copy because the direct
  /// PolarPoissonSolver stores valid cells only.
  MultiFab& ell_phi() {
    if (owner_->polar_) {
      // Allocate lazily (1 ghost) on the System layout, then copy phi^n from aux[0].
      if (!phi_src_polar_)
        phi_src_polar_.emplace(owner_->ba, owner_->dm, 1, 1);
      for (int li = 0; li < phi_src_polar_->local_size(); ++li) {
        const ConstArray4 a = owner_->aux.fab(li).const_array();
        Array4 p = phi_src_polar_->fab(li).array();
        const Box2D v = phi_src_polar_->box(li);
        for_each_cell(v, detail::SystemFieldComponentCopyKernel{p, a, 0, 0});
      }
      return *phi_src_polar_;
    }
    return ell_->phi();
  }
  /// Solves the active cartesian Poisson (GeometricMG V-cycle or direct FFT). Sets the trace
  /// markers; the device_fence after ell_solve is carried by the CALLER (solve_fields), not here.
  SolveReport ell_solve() {
    trace_mark("ell_solve: before provider solve");
    SolveReport report = ell_->solve(*this);
    trace_mark("ell_solve: after provider solve");
    return report;
  }

  // --- ELLIPTIC-SOLVER PROFILING COUNTERS (Spec 5 sec.13.11.1, ADC-479 criteria 42/43) ----------
  // Read-back accessors for the per-solve stats of the ACTIVE cartesian elliptic solver, queried at
  // the System solve_fields seam AFTER ell_solve()+device_fence() to populate the native profiler
  // counters (mg_cycles / krylov_iters / mg_levels / elliptic_bottom). LOW-INVASIVE by design: the
  // deep numerics never sees the profiler -- providers publish their exact backend metrics through
  // the prepared interface. Solvers that lack a notion (direct methods: no cycles/levels/iters)
  // return 0 HONESTLY -- not every solver has these. An absent backend (never solved) -> 0.
  //
  // krylov_iters is 0 on the cartesian field-solve path: the default Poisson uses GeometricMG (or a
  // direct FFT), never a Krylov solver (the Krylov path lives in the condensed Schur SOURCE stage,
  // which is not the ell_ elliptic solve). The counter is emitted for completeness / future Krylov
  // elliptic backends; it stays an honest 0 here.
  int last_mg_cycles() const { return ell_ ? ell_->metrics().multigrid_cycles : 0; }
  int last_krylov_iters() const { return ell_ ? ell_->metrics().krylov_iterations : 0; }
  int last_num_levels() const { return ell_ ? ell_->metrics().multigrid_levels : 0; }
  double last_bottom_seconds() const { return ell_ ? ell_->metrics().bottom_seconds : 0.0; }

  // --- direct POLAR Poisson (PolarPoissonSolver) -------------------------
  /// Builds the direct POLAR Poisson (PolarPoissonSolver, single-rank, single box covering
  /// the ring) LAZILY. The radial BC comes from poisson_bc() (Foextrap -> homogeneous Neumann, wall; the
  /// circular cartesian 'wall' makes no sense on a global ring and is not applied). theta is
  /// PERIODIC (handled by the FFT-in-theta, no azimuthal BC). ADDITIVE: never touches ell_.
  /// @throws std::runtime_error on unknown rhs/solver or variable/aniso/reaction permittivity (unsupported
  /// by the direct polar Poisson).
  void ensure_elliptic_polar() {
    if (pell_)
      return;
    if (p_rhs != "charge_density" && p_rhs != "composite")
      throw std::runtime_error("System::set_poisson (polar): unknown rhs '" + p_rhs +
                               "' (valid: " + kPoissonRhsRouteTokensCsv + ")");
    if (p_solver != "geometric_mg" && p_solver != "polar")
      throw std::runtime_error(
          "System::set_poisson (polar): solver '" + p_solver +
          "' unsupported on a ring; the polar Poisson is direct (FFT-in-theta + tridiag-in-r). "
          "Leave the default ('geometric_mg') or request 'polar'.");
    if (has_variable_diffusion_coefficient() || has_kappa_field_)
      throw std::runtime_error(
          "System::set_poisson (polar): variable / anisotropic permittivity / reaction unsupported "
          "by the direct polar Poisson (Phase 2b; operator (1/r) d_r(r d_r) + (1/r^2) d_theta^2)");
    // MULTI-BOX GUARD (ADC-67): the DIRECT polar Poisson (FFT-in-theta + tridiag-in-r) requires
    // complete theta ROWS and radial COLUMNS on ONE box (PolarPoissonSolver already rejects
    // ba.size()!=1). We raise a clear UPSTREAM message HERE -- before the construction of the solver, so from the
    // 1st solve_fields / potential / set_potential of a System with theta_boxes > 1 -- rather than letting
    // the low-level rejection of PolarPoissonSolver bubble up. The theta SPLITTING (theta_boxes > 1) only serves
    // the TRANSPORT; for a multi-box electrostatic field, go through the tensor Schur stage.
    if (owner_->ba.size() != 1)
      throw std::runtime_error(
          "System: DIRECT polar Poisson incompatible with the theta splitting (theta_boxes=" +
          std::to_string(owner_->ba.size()) +
          " boxes); it requires a single-box grid (theta_boxes=1). For a multi-box theta "
          "splitting, "
          "use a hierarchy-scoped Program.solve with a composite tensor provider, or "
          "go back to theta_boxes=1.");
    // Radial BC: Dirichlet/Neumann from poisson_bc() (xlo/xhi). theta always periodic.
    const BCRec pbc = poisson_bc();
    pell_.emplace(owner_->pgeom_, owner_->ba, pbc);
  }

  /// Assembles the system Poisson RIGHT-HAND SIDE into @p rhs: f = Sum_s elliptic_rhs_s(U_s),
  /// the elliptic brick of EACH block, summed in place (rhs is zeroed first). When
  /// @p target_block >= 0 and @p U_stage != nullptr, the target block reads @p U_stage INSTEAD of
  /// its live s.U (the rest of the blocks keep s.U); this is what solve_fields_from_state needs to
  /// re-solve a field-coupled multi-stage scheme from a per-stage state. With the default
  /// (target_block = -1) it is BIT-IDENTICAL to the historical inline loop (every block from s.U).
  /// STRIDE: a held (hold-then-catch-up) block stays FROZEN at its last advance, so its charge enters
  /// the sum with a STALE state until its next catch-up (loose Poisson coupling, assumed).
  void assemble_poisson_rhs(MultiFab& rhs, int target_block = -1,
                            const MultiFab* U_stage = nullptr) {
    rhs.set_val(Real(0));
    for (std::size_t b = 0; b < owner_->sp.size(); ++b) {
      auto& s = owner_->sp[b];
      const bool override_here = (U_stage != nullptr && static_cast<int>(b) == target_block);
      s.add_poisson_rhs(override_here ? *U_stage : s.U, rhs);
    }
  }

  /// Assembles the system Poisson RIGHT-HAND SIDE into @p rhs for a SIMULTANEOUS coupled multi-block
  /// solve (Spec 3 criterion 24, ADC-457): f = Sum_s elliptic_rhs_s(U_s) where EVERY block reads its
  /// OWN stage state at once, instead of overriding a single target block (assemble_poisson_rhs). @p
  /// U_stages is indexed BY BLOCK INDEX (size == the number of blocks): U_stages[b] != nullptr -> block
  /// b contributes its stage state, U_stages[b] == nullptr -> block b contributes its live s.U. With
  /// every entry pointing at the corresponding live s.U it is the same f as assemble_poisson_rhs(rhs)
  /// (the default head solve), so a coupled solve from the live states is bit-identical to solve_fields.
  /// This is the multi-target stage override solve_fields_from_blocks needs: a field-coupled multi-
  /// species step re-solves phi from the SIMULTANEOUS stage states of all coupled blocks (no operator
  /// observes a partially committed group, the IR commit_many guarantee). @throws std::invalid_argument
  /// if @p U_stages is not sized to the block count (a stale binding cannot silently mis-route).
  void assemble_poisson_rhs_from_blocks(MultiFab& rhs,
                                        const std::vector<const MultiFab*>& U_stages) {
    if (U_stages.size() != owner_->sp.size())
      throw std::invalid_argument(
          "assemble_poisson_rhs_from_blocks: U_stages size " + std::to_string(U_stages.size()) +
          " != block count " + std::to_string(owner_->sp.size()) +
          " (index U_stages by block index; nullptr = use the block's live state)");
    rhs.set_val(Real(0));
    for (std::size_t b = 0; b < owner_->sp.size(); ++b) {
      auto& s = owner_->sp[b];
      s.add_poisson_rhs(U_stages[b] != nullptr ? *U_stages[b] : s.U, rhs);
    }
  }

  /// POLAR solve_fields: assembles f = Sum_s elliptic_rhs_s(U_s) on Kokkos, solves the
  /// polar Poisson, then DERIVES the aux in the local basis (e_r, e_theta):
  ///   aux[0] = phi;  aux[1] = grad_r = d phi/dr;  aux[2] = grad_theta = (1/r) d phi/d theta.
  /// This is the layout expected by ExBVelocityPolar (v_r = -grad_theta/B, v_theta = grad_r/B).
  /// @p target_block / @p U_stage: per-stage state override for the target block (default -1: every
  /// block from s.U, bit-identical to the historical path).
  SolveReport solve_fields_polar(int target_block = -1, const MultiFab* U_stage = nullptr) {
    ensure_elliptic_polar();
    MultiFab& rhs = pell_->rhs();
    assemble_poisson_rhs(rhs, target_block, U_stage);
    // CONSTANT permittivity eps != 1: lap phi = f/eps (1/eps scaling of the rhs), like the
    // cartesian. (variable/aniso eps(x) is refused by ensure_elliptic_polar.)
    if (p_eps_ != Real(1)) {
      scale(rhs, Real(1) / p_eps_);
    }
    pell_->solve();
    device_fence();
    // Derivation (phi, grad_r, grad_theta) in the local basis (e_r, e_theta) via the SAME helper as the C++
    // test (derive_aux_polar of block_builder_polar.hpp). phi is WITHOUT ghost (direct single-box solver):
    // the helper thus never reads an out-of-domain index (radial OFFSET at the walls, theta WRAPPED in
    // periodic) -- that was the bug: the centered difference read phi(lo-1)/phi(hi+1)/phi(.,jlo-1) out of
    // allocation -> spurious gradient -> divergent velocity -> nan.
    derive_aux_polar(pell_->phi(), owner_->aux, owner_->pgeom_);
    apply_te();  // inert in polar ExB (no fluid block source of T_e), kept by symmetry
    // Aux ghosts: theta PERIODIC (joint 0/2pi), r PHYSICAL (extrapolation at the boundary). fill_ghosts
    // already routes through bc_ (xlo/xhi Foextrap, ylo/yhi Periodic) -> correct periodic azimuthal halo.
    fill_ghosts(owner_->aux, owner_->dom, owner_->bc_);
    apply_named_aux_bc();  // ADC-369: per-field halo override on the RADIAL faces (theta stays periodic)
    SolveReport report;
    report.mark_solved();
    return report;
  }

  /// Solves the system Poisson then DERIVES the aux = (phi, grad phi[, B_z, T_e]). Routes to
  /// solve_fields_polar() in polar geometry. device INVARIANT: the device_fence() between ell_solve()
  /// and the derivation of grad phi MUST stay atomic (without it the GPU V-cycle is not finished when
  /// phi is read); the derivation / population loops iterate over the LOCAL fabs (MPI-safe).
  ///
  /// @p target_block / @p U_stage: when set (target_block >= 0, U_stage != nullptr), the target
  /// block's Poisson RHS is assembled from @p U_stage INSTEAD of its live s.U -- the seam
  /// solve_fields_from_state uses so a field-coupled multi-stage scheme re-solves phi from each STAGE
  /// state (the compiled Program runs stages sequentially: stage k's solve overwrites the shared aux
  /// before stage k's RHS reads it). The default (-1 / nullptr) keeps every block at s.U: the
  /// historical solve_fields(), BIT-IDENTICAL.
  SolveReport solve_fields(int target_block = -1, const MultiFab* U_stage = nullptr) {
    if (owner_->polar_)
      // ring: polar Poisson + aux in local basis (e_r, e_theta)
      return solve_fields_polar(target_block, U_stage);
    trace_mark("solve_fields: start");
    ensure_elliptic();
    MultiFab& phi_before = ell_phi();
    MultiFab& published_phi = reusable_scratch_(published_phi_scratch_, phi_before,
                                                phi_before.ncomp(), phi_before.n_grow());
    PureFieldAlgebra::copy(published_phi, phi_before);
    trace_mark("solve_fields: after ensure_elliptic");
    MultiFab& rhs = ell_rhs();
    // f = Sum_s elliptic_rhs_s(U_s). By default the CURRENT state of each block; with a
    // target_block / U_stage override the target block reads U_stage (per-stage field solve, ADC-409).
    assemble_poisson_rhs(rhs, target_block, U_stage);
    p_nullspace_workspace_->require_compatible(rhs);
    trace_mark("solve_fields: after add_poisson_rhs");
    // CONSTANT permittivity: div(eps grad phi) = f <=> lap phi = f/eps, so we scale the rhs by
    // 1/eps. With a VARIABLE or ANISOTROPIC eps(x) field we DO NOT do it: the GeometricMG
    // operator carries eps directly (apply_epsilon_field / apply_epsilon_anisotropic_field), the
    // rhs stays f as is.
    if (!has_variable_diffusion_coefficient() && p_eps_ != Real(1)) {
      scale(rhs, Real(1) / p_eps_);
    }
    trace_mark("solve_fields: before ell_solve");
    SolveReport report = ell_solve();
    if (!report.solved_value_available()) {
      PureFieldAlgebra::copy(phi_before, published_phi);
      return report;
    }
    trace_mark("solve_fields: after ell_solve, before device_fence");
    device_fence();
    trace_mark("solve_fields: after device_fence (aux derivation)");
    // Per-cell derivation (phi, grad phi) -> aux channel: LOCAL to the owner rank. System
    // distributes ONE box (round-robin DistributionMapping(1, n_ranks())), so at np>1 a single rank
    // owns it; the others have local_size()==0 and HAVE NO fab to derive. We iterate over the LOCAL
    // fabs (never fab(0) hardcoded): no-op on an empty rank, identical to before on the owner
    // (loop executed once, bit-identical to np=1). ell_phi() and aux share the same
    // DistributionMapping -> same local indexing.
    MultiFab& phi_mf = ell_phi();
    p_nullspace_workspace_->apply_gauge(phi_mf);
    device_fence();
    derive_cartesian_aux(phi_mf);
    trace_mark("solve_fields: after aux derivation (phi, grad phi)");
    apply_te();  // T_e = p/rho of the fluid block source, recomputed on each solve (B_z, comp 3, preserved)
    trace_mark("solve_fields: after apply_te");
    fill_ghosts(owner_->aux, owner_->dom, owner_->bc_);
    apply_named_aux_bc();  // ADC-369: per-field halo override (after the shared fill; no-op if none)
    trace_mark("solve_fields: end (fill ghosts aux)");
    return report;
  }

  /// Per-stage field solve (ADC-409): SAME solve + derive-aux as solve_fields(), but the target
  /// block @p block_idx assembles its Poisson RHS from @p U_stage instead of its live s.U (the other
  /// blocks keep s.U). This re-fills the SHARED aux (phi, grad phi) with phi(U_stage); a compiled
  /// Program's stage-k RHS, called right after this, then reads phi solved from stage k's own state --
  /// removing the "solve from current state only" limitation for sequential multi-stage schemes. The
  /// next stage's solve overwrites the aux, so no distinct per-stage buffer is needed. With
  /// block_idx == 0 and U_stage == U^n (the first stage) this is identical to solve_fields().
  /// @throws std::out_of_range if @p block_idx is not a valid block index.
  SolveReport solve_fields_from_state(int block_idx, const MultiFab& U_stage) {
    if (block_idx < 0 || block_idx >= static_cast<int>(owner_->sp.size()))
      throw std::out_of_range("solve_fields_from_state: block index " + std::to_string(block_idx) +
                              " out of range (" + std::to_string(owner_->sp.size()) + " blocks)");
    return solve_fields(block_idx, &U_stage);
  }

  /// POLAR coupled multi-block solve (Spec 3 criterion 24, ADC-457): same solve + aux derivation as
  /// solve_fields_polar(), but the Poisson RHS is assembled from the SIMULTANEOUS stage states of all
  /// blocks (assemble_poisson_rhs_from_blocks) instead of a single-target override. @p U_stages is
  /// indexed by block index (nullptr = the block's live state). Mirrors solve_fields_polar() step for
  /// step (eps scaling, pell_->solve, device_fence, derive_aux_polar, apply_te, fill_ghosts, named-aux
  /// halo) -- only the RHS assembly differs, so a coupled solve from the live states is bit-identical.
  SolveReport solve_fields_polar_from_blocks(const std::vector<const MultiFab*>& U_stages) {
    ensure_elliptic_polar();
    MultiFab& rhs = pell_->rhs();
    assemble_poisson_rhs_from_blocks(rhs, U_stages);
    if (p_eps_ != Real(1)) {
      scale(rhs, Real(1) / p_eps_);
    }
    pell_->solve();
    device_fence();
    derive_aux_polar(pell_->phi(), owner_->aux, owner_->pgeom_);
    apply_te();
    fill_ghosts(owner_->aux, owner_->dom, owner_->bc_);
    apply_named_aux_bc();
    SolveReport report;
    report.mark_solved();
    return report;
  }

  /// Coupled multi-block field solve (Spec 3 criterion 24, ADC-457): SAME elliptic solve + aux
  /// derivation as solve_fields(), but the system Poisson RHS is assembled from the SIMULTANEOUS stage
  /// states of MULTIPLE blocks (assemble_poisson_rhs_from_blocks) -- every coupled block reads its OWN
  /// stage state at once, not a single-target override. @p U_stages is indexed by block index
  /// (nullptr = the block's live state). Routes to solve_fields_polar_from_blocks() in polar geometry.
  /// Mirrors solve_fields() step for step (the device_fence between ell_solve and the grad derivation,
  /// the LOCAL-fab loops, the order of fill_ghosts/fill_boundary); only the RHS assembly differs, so a
  /// coupled solve from the live states is bit-identical to solve_fields(). The codegen lowers
  /// P.solve_fields_from_blocks([...]) to this; the seam a multi-species field-coupled step uses to
  /// re-solve phi from all coupled blocks' stage states simultaneously. @throws (via
  /// assemble_poisson_rhs_from_blocks) if @p U_stages is not sized to the block count.
  SolveReport solve_fields_from_blocks(const std::vector<const MultiFab*>& U_stages) {
    if (owner_->polar_)
      return solve_fields_polar_from_blocks(U_stages);
    ensure_elliptic();
    MultiFab& phi_before = ell_phi();
    MultiFab& published_phi = reusable_scratch_(published_phi_scratch_, phi_before,
                                                phi_before.ncomp(), phi_before.n_grow());
    PureFieldAlgebra::copy(published_phi, phi_before);
    // Coupled multi-block solve always re-solves the Poisson from the requested stage states.
    SolveReport report;
    {
      MultiFab& rhs = ell_rhs();
      assemble_poisson_rhs_from_blocks(rhs, U_stages);
      p_nullspace_workspace_->require_compatible(rhs);
      if (!has_variable_diffusion_coefficient() && p_eps_ != Real(1)) {
        scale(rhs, Real(1) / p_eps_);
      }
      report = ell_solve();
      if (!report.solved_value_available()) {
        PureFieldAlgebra::copy(phi_before, published_phi);
        return report;
      }
    }
    device_fence();
    MultiFab& phi_mf = ell_phi();
    p_nullspace_workspace_->apply_gauge(phi_mf);
    device_fence();
    derive_cartesian_aux(phi_mf);
    apply_te();
    fill_ghosts(owner_->aux, owner_->dom, owner_->bc_);
    apply_named_aux_bc();
    return report;
  }

  // --- NAMED multi-elliptic field (ADC-428) ----------------------------------
  /// Builds the dedicated cartesian backend for named field @p nf through the same provider protocol
  /// as the default field. The core materializes one complete physical request; the selected
  /// provider owns its compatibility decision, implementation, exact contract and configuration. The variable
  /// or anisotropic permittivity of the default field is not inherited by a named field. Built
  /// lazily; no-op if already built.
  void ensure_named_backend(NamedField& nf, const std::string& slot) {
    if (nf.backend) {
      prepare_named_nullspace_(nf);
      return;
    }
    require_field_plan_consensus();
    const std::string& provider_identity =
        nf.has_plan ? nf.plan.backend_provider_identity : p_solver;
    const BCRec pbc = nf.has_plan ? named_field_bc(nf.plan) : poisson_bc();
    EllipticBackendBuildRequest request;
    request.elliptic = {owner_->geom,
                        owner_->ba,
                        owner_->dm,
                        pbc,
                        nf.has_plan ? ActiveRegionProvider2D{} : wall_active(),
                        FieldDistribution::Distributed,
                        0,
                        1};
    request.topology_digest = nf.plan.topology_digest;
    request.topology_provenance = nf.plan.topology_provenance;
    request.material_points = static_cast<std::size_t>(owner_->dom.num_cells());
    // External components authenticate their own pair of prepared component manifests.  They do not
    // publish the optional native EllipticOperatorContract; builtin providers still expose it.
    request.require_exact_operator_contract = false;
    ExactContractBuilder configuration;
    configuration.text("pops.runtime.named-elliptic-configuration")
        .scalar(std::uint32_t{1})
        .text(nf.plan.plan_identity)
        .presence(nf.plan.has_reaction);
    if (nf.plan.has_reaction)
      configuration.scalar(nf.plan.reaction);
    configuration.presence(nf.plan.has_boundary_kernel).presence(nf.plan.has_newton);
    request.exact_configuration_contract = std::move(configuration).release();
    if (nf.plan.has_reaction) {
      request.reaction_coefficient.emplace(owner_->ba, owner_->dm, 1, 0);
      request.reaction_coefficient->set_val(nf.plan.reaction);
    }
    if (nf.plan.has_boundary_kernel) {
      request.dynamic_boundary = nf.plan.boundary_kernel;
      request.boundary_context = nf.plan.boundary_context;
    }
    if (nf.plan.has_newton)
      request.nonlinear_boundary = nf.plan.newton;

    nf.backend = elliptic_registry_.prepare(provider_identity, std::move(request));
    prepare_named_nullspace_(nf);
  }

  /// Assembles the RIGHT-HAND SIDE of named field @p field into @p rhs: f = Sum_s
  /// named_poisson_rhs_s[field](U_s), the per-field elliptic brick of EACH block that declares it
  /// (ADC-428). When @p target_block >= 0 and @p U_stage != nullptr, the target block reads @p U_stage
  /// instead of its live s.U (per-stage field solve, like assemble_poisson_rhs). @throws if no block
  /// declares this field (a named field with no contributing block would solve a zero RHS silently).
  void assemble_named_poisson_rhs(const std::string& field, MultiFab& rhs, int target_block,
                                  const MultiFab* U_stage) {
    rhs.set_val(Real(0));
    const auto route = named_fields_.find(field);
    if (route == named_fields_.end() || !route->second.has_plan)
      throw std::runtime_error("System: field provider slot '" + field + "' is not installed");
    prepare_field_providers(route->second);
    MultiFab& contribution = reusable_scratch_(route->second.contribution_scratch, rhs, 1, 0);
    for (const auto& binding : route->second.prepared_providers) {
      const int b = binding.block;
      auto& state = owner_->sp[static_cast<std::size_t>(b)];
      const bool override_here = (U_stage != nullptr && b == target_block);
      if (binding.coefficient == Real(1)) {
        binding.rhs(override_here ? *U_stage : state.U, rhs);
      } else {
        contribution.set_val(Real(0));
        binding.rhs(override_here ? *U_stage : state.U, contribution);
        saxpy(rhs, binding.coefficient, contribution);
      }
    }
  }

  /// Solves named field @p field's SECOND elliptic problem from block @p block_idx's stage state
  /// @p U_stage and writes phi (+ centered grad) into the field's OWN aux components (ADC-428):
  /// assemble f = Sum_s named_poisson_rhs_s[field] -> backend.solve() ->
  /// aux[phi_comp] = phi, aux[gx_comp]/aux[gy_comp] = centered grad. The SHARED phi/grad (components
  /// 0..2) and the default Poisson (ell_) are NOT touched. The named aux components are then ghost-
  /// filled (the shared aux fill + the per-field halo override). CARTESIAN only (the polar named path is
  /// a future extension); @throws on the polar geometry or an unknown field.
  SolveReport solve_named_field_from_state(const std::string& field, int block_idx,
                                           const MultiFab& U_stage) {
    require_field_plan_consensus();
    if (block_idx < 0 || block_idx >= static_cast<int>(owner_->sp.size()))
      throw std::out_of_range("solve_fields_from_state (named): block index " +
                              std::to_string(block_idx) + " out of range (" +
                              std::to_string(owner_->sp.size()) + " blocks)");
    auto it = named_fields_.find(field);
    if (it == named_fields_.end())
      throw std::runtime_error("System: unknown named elliptic field '" + field +
                               "' (register it via m.elliptic_field + the compiled block)");
    if (owner_->polar_)
      throw std::runtime_error("System: named elliptic field '" + field +
                               "' on a polar (ring) grid is not supported yet (cartesian only)");
    NamedField& nf = it->second;
    if (nf.phi_comp < 0 || nf.phi_comp >= owner_->aux_ncomp_)
      throw std::runtime_error(
          "System: named elliptic field '" + field +
          "' aux component out of the channel width (add the block that declares "
          "its aux fields before solving)");
    if (nf.gradient_sign != -1 && nf.gradient_sign != 1)
      throw std::runtime_error("System: named elliptic field has no valid gradient sign");
    const bool has_gradient = nf.gx_comp >= 0 && nf.gy_comp >= 0;
    if (nf.phi_comp >= owner_->aux_ncomp_ ||
        (has_gradient && (nf.gx_comp >= owner_->aux_ncomp_ || nf.gy_comp >= owner_->aux_ncomp_)))
      throw std::runtime_error(
          "System: named elliptic field output components exceed the aux channel width");
    prepare_boundary_dependencies(nf, block_idx, &U_stage);
    ensure_named_backend(nf, field);
    nf.backend->configure_boundary(nf.plan);
    MultiFab& rhs = nf.backend->rhs();
    MultiFab& phi_mf = nf.backend->phi();
    MultiFab& published_phi =
        reusable_scratch_(nf.published_phi_scratch, phi_mf, phi_mf.ncomp(), phi_mf.n_grow());
    PureFieldAlgebra::copy(published_phi, phi_mf);
    auto restore_published = [&]() { PureFieldAlgebra::copy(phi_mf, published_phi); };
    assemble_named_poisson_rhs(field, rhs, block_idx, &U_stage);
    nf.backend->prepare_rhs(*this, rhs, nf.plan, *nf.nullspace_workspace);
    SolveReport report;
    try {
      report = nf.backend->solve(*this);
      if (!report.solved_value_available()) {
        restore_published();
        return report;
      }
    } catch (...) {
      restore_published();
      throw;
    }
    device_fence();  // CRITICAL: the V-cycle must finish before phi is read (same invariant as ell_)
    nf.backend->finalize(*this, nf.plan, *nf.nullspace_workspace);
    device_fence();
    const Real dx = owner_->geom.dx(), dy = owner_->geom.dy();
    const int cphi = nf.phi_comp, cgx = nf.gx_comp, cgy = nf.gy_comp;
    const Real gradient_scale = static_cast<Real>(nf.gradient_sign);
    const bool grad = has_gradient;
    for (int li = 0; li < owner_->aux.local_size(); ++li) {
      const ConstArray4 p = phi_mf.fab(li).const_array();
      Array4 a = owner_->aux.fab(li).array();
      const Box2D v = owner_->aux.box(li);
      for_each_cell(v, detail::SystemNamedFieldPostprocessKernel{a, p, cphi, cgx, cgy,
                                                                 gradient_scale, dx, dy, grad});
    }
    // Ghost-fill the named field's aux components: the shared aux fill (same routing as solve_fields)
    // then the per-field halo override (ADC-369). This re-fills ALL components -- the shared phi/grad
    // were last written by the default solve_fields, so their valid cells are unchanged and only the
    // halos are recomputed (idempotent for those components).
    fill_ghosts(owner_->aux, owner_->dom, owner_->bc_);
    apply_named_aux_bc();
    return report;
  }

 private:
  static MultiFab& reusable_scratch_(std::optional<MultiFab>& storage, const MultiFab& prototype,
                                     int ncomp, int ghosts) {
    const bool compatible = storage &&
                            storage->box_array().boxes() == prototype.box_array().boxes() &&
                            storage->dmap().ranks() == prototype.dmap().ranks() &&
                            storage->ncomp() == ncomp && storage->n_grow() == ghosts;
    if (!compatible)
      storage.emplace(prototype.box_array(), prototype.dmap(), ncomp, ghosts);
    return *storage;
  }

  Impl* owner_;
  bool field_plan_consensus_verified_ = false;
};

}  // namespace field_solver
}  // namespace pops
