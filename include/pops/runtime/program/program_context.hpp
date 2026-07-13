#pragma once

#include <chrono>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <pops/core/foundation/types.hpp>      // Real, POPS_HD
#include <pops/runtime/program/profiler.hpp>   // Profiler / ProfileScope (per-node timing, ADC-459)
#include <pops/runtime/program/wire_ids.hpp>   // stable compiled-Program numeric protocol
#include <pops/mesh/boundary/physical_bc.hpp>  // fill_ghosts (periodic / physical halo exchange)
#include <pops/mesh/execution/for_each.hpp>  // for_each_cell (per-cell coeff / reconstruct kernels + negated divergence copy)
#include <pops/mesh/geometry/geometry.hpp>  // Geometry (mesh metric of the Laplacian / gradient)
#include <pops/mesh/storage/fab2d.hpp>      // Array4 / ConstArray4 (per-cell handles)
#include <pops/mesh/storage/mf_arith.hpp>   // saxpy (linear combine over a MultiFab)
#include <pops/mesh/storage/multifab.hpp>   // MultiFab
#include <pops/numerics/elliptic/interface/elliptic_problem.hpp>  // field_postprocess (centered gradient)
#include <pops/numerics/elliptic/linear/generic_krylov.hpp>  // ApplyFn / cg / bicgstab / gmres / richardson (solve_linear_matfree seam)
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>  // apply_laplacian (shared 5-point matvec)
#include <pops/numerics/elliptic/polar/polar_tensor_operator.hpp>  // metric-aware generated tensor solve
#include <pops/runtime/config/runtime_params.hpp>  // RuntimeParams (compiled-Program runtime params, ADC-510)
#include <pops/runtime/context/grid_context.hpp>   // GridContext (System aux seam)
#include <pops/runtime/program/cache_manager.hpp>  // CacheManager (held-node value cache, ADC-458)
#include <pops/runtime/system.hpp>                 // System (the runtime this facade forwards to)

/// @file
/// @brief ProgramContext -- the C++-side facade a generated problem.so calls to run a compiled time
///        Program during sim.step(dt) (epic ADC-399, ADC-401 Phase 2b).
///
/// It REIMPLEMENTS NOTHING. Each method forwards to an existing pops::System primitive:
///   install(fn)          -> System::install_program_step(fn)   (registers the macro-step body)
///   solve_fields()       -> System::solve_fields()             (elliptic solve + aux at current U)
///   solve_fields_from_state(b, U) -> System::solve_fields_from_state(b, U) (aux at a stage state)
///   n_blocks()           -> System::n_blocks()
///   state(b)             -> System::block_state(b)             (the block's live MultiFab, zero-copy)
///   rhs_into(b, U, R)    -> System::block_rhs_into(b, U, R)    (R <- -div F + S, Poisson frozen)
///   neg_div_flux_default_into(b, U, R) -> System::block_neg_div_flux_into (R <- -div F, NO source)
///   axpy(U, a, R)        -> pops::saxpy(U, a, R)                (U <- U + a R, device-dispatched)
///
/// The Program composes the chain (e.g. Forward Euler = solve_fields(); for each block:
/// rhs_into(b, U, R); axpy(U, dt, R)) and installs it via install(...). The .so NEVER touches
/// System::Impl / Array4 / fill_boundary / the elliptic solver / Kokkos / MPI / CFL / substeps.
///
/// IDIOM: ProgramContext is a plain (non-template) class holding a System*. A generated .so receives
/// the System as a flat void* across the dlopen boundary (like the native loader's `void* self`) and
/// wraps it here; it reaches per-block storage through the System's public accessors because
/// System::Impl is private to the _pops translation unit.
namespace pops {
namespace runtime {
namespace program {

class ProgramContext {
 public:
  explicit ProgramContext(System* sys) : sys_(sys) {}
  /// Wraps a System passed as a flat void* (what pops_install_program(void* sys) receives).
  explicit ProgramContext(void* sys) : sys_(static_cast<System*>(sys)) {}

  /// Register the macro-step body. @p step advances ONE macro-step over dt (it owns solve_fields,
  /// the RHS, the linear combine and the commit). Empty std::function clears it.
  void install(std::function<void(double)> step) const {
    sys_->install_program_step(std::move(step));
  }

  /// Translate a PROGRAM block index @p b (P.state declaration order, what the codegen emits) to the
  /// SYSTEM block index it names (Spec 3 criterion 23, ADC-457). install_program stores the explicit
  /// name-matched map before the generated entry point constructs this context. Direct C++ users must
  /// install the same explicit map themselves: an empty, incomplete or invalid map is never interpreted
  /// positionally. Every seam method taking a block index routes through here, so the System blocks may
  /// be added in ANY order vs the Program's P.state declarations.
  int sys_block(int b) const {
    const std::vector<int>& m = sys_->program_block_map();
    if (m.empty())
      throw block_map_error_(
          "ProgramContext::sys_block: no explicit program-to-system block map is installed; "
          "positional block identity is not supported");
    if (b < 0 || b >= static_cast<int>(m.size()))
      throw block_map_error_(
          "ProgramContext::sys_block: program block index " + std::to_string(b) +
          " is outside the explicit block map [0, " + std::to_string(m.size()) + ")");
    const int mapped = m[static_cast<std::size_t>(b)];
    const int count = sys_->n_blocks();
    if (mapped < 0 || mapped >= count)
      throw block_map_error_(
          "ProgramContext::sys_block: program block index " + std::to_string(b) +
          " maps to invalid system block index " + std::to_string(mapped) +
          " for a System with " + std::to_string(count) + " blocks");
    return mapped;
  }

  void solve_fields() const {
    // No count_kernel() here: this forwards to the PUBLIC System::solve_fields() -> Impl::solve_fields(),
    // which already counts the kernel. (The from_state/from_blocks/named seams below DO count, because
    // their Impl paths do not.) Counting here too would double-count this one op.
    sys_->solve_fields();
  }
  /// Per-stage field solve (ADC-409): re-solve the elliptic fields and re-fill the shared aux from
  /// block @p b's STAGE state @p u_stage (not its live state), so a field-coupled multi-stage
  /// Program's stage k reads phi solved from stage k's own state. Forwards to
  /// System::solve_fields_from_state. With b = 0 and u_stage = U^n (the first stage) it matches
  /// solve_fields(); the codegen lowers every solve_fields op to this, passing the stage's state var.
  void solve_fields_from_state(int b, MultiFab& u_stage) const {
    count_kernel();
    sys_->solve_fields_from_state(sys_block(b), u_stage);
  }
  /// Named multi-elliptic field solve (ADC-428): re-solve the SECOND elliptic field @p field from block
  /// @p b's stage state @p u_stage and write its phi (+ centered grad) into the field's OWN aux
  /// components (distinct from the shared phi/grad the default solve_fields fills). Forwards to
  /// System::solve_fields_from_state(field, b, u_stage). The codegen lowers
  /// P.solve_fields(field=name, state=U) to this; a default (unnamed) solve_fields keeps the overload
  /// above, byte-identical.
  void solve_fields_from_state(const std::string& field, int b, MultiFab& u_stage) const {
    count_kernel();
    sys_->solve_fields_from_state(field, sys_block(b), u_stage);
  }
  /// Coupled multi-block field solve (Spec 3 criterion 24, ADC-457): re-solve the elliptic fields and
  /// re-fill the shared aux from the SIMULTANEOUS stage states of MULTIPLE blocks at once -- the system
  /// Poisson RHS is Sum_s elliptic_rhs_s(U_s), every coupled block reading its OWN stage state (not a
  /// single-target override). @p u_stages is indexed BY BLOCK INDEX (size == n_blocks()); a nullptr
  /// entry uses that block's live state. Forwards to System::solve_fields_from_blocks. The codegen
  /// lowers P.solve_fields_from_blocks([U0, U1, ...]) to this, building the per-block pointer vector
  /// from the listed stage-state vars (their declaration order == the block index order, asserted at
  /// emit time). This is the multi-target counterpart of solve_fields_from_state.
  void solve_fields_from_blocks(const std::vector<const MultiFab*>& u_stages) const {
    count_kernel();
    // The codegen builds @p u_stages indexed BY PROGRAM block index (a stage state slotted at its own
    // Program index, the rest nullptr). The System solver expects it indexed by SYSTEM block index, so
    // re-slot each Program entry p at its name-matched System index sys_block(p) (Spec 3 criterion 23,
    // ADC-457). Even an order-matching Program carries an explicit identity map.
    const std::vector<int>& m = sys_->program_block_map();
    if (m.empty())
      throw block_map_error_(
          "ProgramContext::solve_fields_from_blocks: no explicit program-to-system block map is "
          "installed; positional block identity is not supported");
    if (u_stages.size() < m.size())
      throw block_map_error_(
          "ProgramContext::solve_fields_from_blocks: received " +
          std::to_string(u_stages.size()) + " Program stage slots for an explicit block map with " +
          std::to_string(m.size()) + " entries");
    std::vector<const MultiFab*> remapped(static_cast<std::size_t>(sys_->n_blocks()), nullptr);
    // Iterate the PROGRAM block indices [0, m.size()) -- NOT u_stages.size(), which is the larger
    // SYSTEM block count. The codegen sizes u_stages to ctx.n_blocks() but only fills Program slots
    // [0, n_program_blocks); when the System has MORE blocks than the Program declares (a subset
    // install), walking the System-sized range would re-map the nullptr padding through the identity
    // fallthrough and clobber real entries. m[p] is Program block p's System index (install-validated
    // in range); the unlisted System slots stay nullptr = their live state. sys_block validates every
    // mapped value before it is used as a vector index.
    for (std::size_t p = 0; p < m.size(); ++p) {
      const int mapped = sys_block(static_cast<int>(p));
      remapped[static_cast<std::size_t>(mapped)] = u_stages[p];
    }
    sys_->solve_fields_from_blocks(remapped);
  }

  void solve_fields_from_blocks(const std::string& field,
                                const std::vector<const MultiFab*>& u_stages) const {
    count_kernel();
    const std::vector<int>& map = sys_->program_block_map();
    if (u_stages.size() != map.size())
      throw std::runtime_error(
          "ProgramContext::solve_fields_from_blocks(field): stage vector size mismatch");
    std::vector<MultiFab*> live;
    std::vector<MultiFab> published;
    live.reserve(u_stages.size());
    published.reserve(u_stages.size());
    int representative = -1;
    for (std::size_t p = 0; p < u_stages.size(); ++p) {
      if (u_stages[p] == nullptr)
        continue;
      const int system_block = sys_block(static_cast<int>(p));
      MultiFab& state_value = sys_->block_state(system_block);
      if (representative < 0)
        representative = system_block;
      live.push_back(&state_value);
      published.push_back(state_value);
      state_value = *u_stages[p];
    }
    if (representative < 0)
      throw std::runtime_error(
          "ProgramContext::solve_fields_from_blocks(field): no stage override was supplied");
    auto restore = [&]() {
      for (std::size_t i = 0; i < live.size(); ++i)
        *live[i] = std::move(published[i]);
    };
    try {
      sys_->solve_fields_from_state(field, representative,
                                    sys_->block_state(representative));
      restore();
    } catch (...) {
      restore();
      throw;
    }
  }
  int n_blocks() const { return sys_->n_blocks(); }
  Real physical_time() const { return static_cast<Real>(sys_->time()); }
  void set_field_logical_timepoint(const std::string& field,
                                   const FieldLogicalTimePoint& point) const {
    sys_->set_field_logical_timepoint(field, point);
  }
  void set_field_boundary_parameters(const std::string& field,
                                     const std::vector<double>& parameters) const {
    sys_->set_field_boundary_parameters(field, parameters);
  }
  void set_field_boundary_kernel(const std::string& field,
                                 const CompiledFieldBoundaryKernel& kernel) const {
    sys_->set_field_boundary_kernel(field, kernel);
  }
  MultiFab& state(int b) const { return sys_->block_state(sys_block(b)); }
  void rhs_into(int b, MultiFab& u, MultiFab& r, int rate_id = -1) const {
    (void)rate_id;
    count_kernel();
    sys_->block_rhs_into(sys_block(b), u, r);
  }

  /// r <- -div F(u) for block @p b -- the SAME flux divergence as @ref rhs_into but WITHOUT the model's
  /// default/composite source (Poisson frozen). Forwards to System::block_neg_div_flux_into (the block's
  /// SourceFreeModel<Model> rhs path, bit-identical to rhs_into minus the source). The codegen lowers a
  /// hyperbolic stage that excludes the default source (P.rhs(flux=True, sources without "default"),
  /// incl. the empty list) to this, so a Lie/Strang split assembles "flux but no source" without the
  /// default source leaking in (epic ADC-399 / ADC-425, spec criterion 17). Header-inline forwarder,
  /// like @ref rhs_into.
  void neg_div_flux_default_into(int b, MultiFab& u, MultiFab& r, int rate_id = -1) const {
    (void)rate_id;
    count_kernel();
    sys_->block_neg_div_flux_into(sys_block(b), u, r);
  }

  /// r <- S(u, aux) for block @p b -- the model's default/composite SOURCE only, WITHOUT the flux
  /// divergence (the exact MIRROR of @ref neg_div_flux_default_into). Forwards to
  /// System::block_source_into (the block's SourceInto path, bit-identical to the source half of
  /// rhs_into). The codegen lowers a SOURCE stage (P.rhs(flux=False, sources with "default")) to this, so
  /// a Lie/Strang split assembles "the default source but no flux" without the -div F base leaking in
  /// (epic ADC-399 / ADC-430, spec: rhs flux=False is source-only). Header-inline forwarder, like @ref
  /// neg_div_flux_default_into.
  void source_default_into(int b, MultiFab& u, MultiFab& r) const {
    count_kernel();
    sys_->block_source_into(sys_block(b), u, r);
  }

  /// The MIN physical cell size of the grid (Cartesian min(dx, dy); polar min(dr, r_min*dtheta)) -- the
  /// SAME hmin the native CFL uses. Forwards to System::cfl_min_dx. A compiled time Program's dt bound
  /// (epic ADC-399 / ADC-417, spec s18) reads it to express e.g. cfl * hmin / max_wave_speed.
  Real hmin() const { return sys_->cfl_min_dx(); }

  /// The maximum |wave speed| of block @p b on the state @p u: the SAME per-block reduction step_cfl
  /// reads (BlockState::max_speed). Forwards to System::block_max_speed -- it REUSES the block's
  /// wave-speed closure, it does not recompute the speed. @p u is the state the bound is evaluated on
  /// (the block's current state for a CFL bound). The dt_bound expression uses it as the denominator of
  /// cfl * hmin / max_wave_speed (epic ADC-399 / ADC-417, spec s18).
  Real max_wave_speed(int b, const MultiFab& u) const {
    return sys_->block_max_speed(sys_block(b), u);
  }

  /// The System aux MultiFab (phi=0, grad_x=1, grad_y=2, B_z=3, T_e=4, named fields from
  /// kAuxNamedBase). NOT owned by the context: it is the live System aux (stable address), the same
  /// channel solve_fields() fills. A generated local-linear-solve kernel reads the operator
  /// coefficients (e.g. B_z) from it. Forwards to System::grid_context().aux.
  MultiFab& aux() const { return *sys_->grid_context().aux; }

  /// The System grid context (transport BC + mesh geometry + the live aux pointer). BY VALUE:
  /// grid_context() returns a temporary. A generic seam accessor forwarding to
  /// System::grid_context(), used by out-of-line runtime operators (the coupled elliptic operator
  /// modules) that assemble coefficient / flux halos from the transport BC without reaching into
  /// System::Impl -- the SAME channel geom() / aux() expose, bundled.
  GridContext grid_context() const { return sys_->grid_context(); }

  /// The MultiFab a per-level coefficient / RHS assembly kernel should WRITE its field into (ADC-633).
  /// On the uniform System the answer is always the passed field itself -- an IDENTITY hook, so a
  /// templated assembly free function writes straight into the level-0-bound scratch the codegen
  /// allocated, byte-for-byte as before. The @p role tag (a field id defined by the assembly module) is
  /// ignored here; it exists so the AMR ProgramContext can, on a refined hierarchy, redirect the write
  /// to a per-level composite buffer instead. Kept trivial + inline so the uniform .so is unchanged.
  MultiFab& assembly_target(MultiFab& field, int role) const {
    validate_assembly_write_role(role, "ProgramContext::assembly_target");
    if (sys_->program_is_polar() && role >= 0 && role < 4)
      polar_coeffs_[static_cast<std::size_t>(role)] = &field;
    return field;
  }

  /// The MultiFab a per-level reconstruction should READ its solved field from (ADC-633). Identity on
  /// the uniform System (the field passed is the level-0 solution the emitted solve wrote); the AMR
  /// ProgramContext redirects the READ to the current level's published composite field on a refined
  /// hierarchy. Trivial + inline so the uniform .so is byte-for-byte unchanged.
  MultiFab& assembly_source(MultiFab& field, int role) const {
    validate_assembly_read_role(role, "ProgramContext::assembly_source");
    return field;
  }
  /// Uniform counterpart of AmrProgramContext::linear_solution: one grid has one solve field.
  MultiFab& linear_solution(MultiFab& field) const { return field; }

  /// Solve the matrix-free linear system A(phi) = rhs of a compiled Program (ADC-633). On the uniform
  /// System this dispatches by @p method to the SAME matrix-free Krylov call the codegen used to emit
  /// INLINE (cg / bicgstab / gmres / richardson) with byte-identical arguments, so a uniform Program's
  /// trajectory is unchanged; only the emission indirects through this seam. The AMR ProgramContext
  /// OVERRIDES this method to route a refined hierarchy through its composite elliptic. @p method:
  /// 0 = cg, 1 = bicgstab, 2 = gmres, 3 = richardson (LinearSolveMethod). @p precond is the
  /// preconditioner ApplyFn (empty = unpreconditioned); @p restart is the GMRES basis size (ignored by
  /// the others). The SolveReport is returned to generated code, which must consume it before
  /// publishing @p sol as a solved value.
  SolveReport solve_linear_matfree(MultiFab& sol, const MultiFab& rhs, const ApplyFn& apply,
                                   const ApplyFn& precond, int method, Real tol, int max_iter,
                                   int restart, Real omega) const {
    validate_linear_solve_method(method, "ProgramContext::solve_linear_matfree");
    if (sys_->program_is_polar()) {
      // The generated coefficient assembly registered the four tensor fields through
      // assembly_target.  Solve the same authored mathematical operator with the generic polar metric
      // backend; no source-stage/physics vocabulary enters this dispatch.
      for (std::size_t c = 0; c < polar_coeffs_.size(); ++c)
        if (!polar_coeffs_[c]) {
          SolveReport report;
          report.mark_failed(SolveStatus::kInvalidInput, SolveAction::kRejectAttempt);
          return report;
        }
      if (method != kLinearSolveBicgstab) {
        SolveReport report;
        report.mark_failed(SolveStatus::kCapabilityFailure);
        return report;
      }
      if (!polar_tensor_)
        polar_tensor_ = std::make_shared<PolarTensorKrylovSolver>(
            sys_->program_polar_geometry(), sol.box_array(), sys_->grid_context().bc,
            PolarPrecond::RadialLine);
      polar_tensor_->set_coefficients(polar_coeffs_[0], polar_coeffs_[1], polar_coeffs_[2],
                                      polar_coeffs_[3]);
      // The Program authors -div(A grad phi)=rhs. PolarTensorKrylovSolver owns the equivalent
      // +div(A grad phi) operator, hence its RHS is the exact negation.  Keep the authored initial
      // guess and do not publish a failed partial iterate.
      pops::lincomb(polar_tensor_->phi(), Real(1), sol, Real(0), sol);
      pops::lincomb(polar_tensor_->rhs(), Real(-1), rhs, Real(0), rhs);
      const PolarKrylovResult polar = polar_tensor_->solve(tol, max_iter);
      SolveReport report;
      report.iters = polar.iters;
      report.rel_residual = polar.rel_residual;
      if (polar.converged) {
        pops::lincomb(sol, Real(1), polar_tensor_->phi(), Real(0), polar_tensor_->phi());
        report.mark_solved();
      } else if (std::isfinite(static_cast<double>(polar.rel_residual))) {
        report.mark_failed(SolveStatus::kIterationLimit);
      } else {
        report.mark_failed(SolveStatus::kInvalidEvaluation);
      }
      return report;
    }
    switch (method) {
      case kLinearSolveCg:
        return pops::cg_solve(apply, sol, rhs, tol, max_iter);
      case kLinearSolveGmres:
        return pops::gmres_solve(apply, precond, sol, rhs, tol, max_iter, restart);
      case kLinearSolveRichardson:
        return pops::richardson_solve(apply, sol, rhs, omega, tol, max_iter);
      case kLinearSolveBicgstab:
        return pops::bicgstab_solve(apply, precond, sol, rhs, tol, max_iter);
      default:
        SolveReport report;
        report.mark_failed(SolveStatus::kInvalidInput, SolveAction::kRejectAttempt);
        return report;  // validated above; keeps exhaustive behavior explicit for defensive builds
    }
  }

  /// A fresh scalar field co-distributed with the System mesh (block 0's box array / distribution),
  /// @p n_comp components, @p n_ghost ghost layers, zero-initialized. Forwards to
  /// System::alloc_scalar_field. The scratch fields (residual, search direction, solution) a
  /// matrix-free Krylov solve allocates -- a 1-component field is distinct from the n_cons block state,
  /// but shares its (ba, dm) so laplacian / gradient pair it with the state and aux by local fab index.
  MultiFab alloc_scalar_field(int n_comp = 1, int n_ghost = 1) const {
    return sys_->alloc_scalar_field(n_comp, n_ghost);
  }

  /// The System mesh geometry (index domain + physical bounds, dx/dy). BY VALUE: grid_context()
  /// returns a temporary, so a reference to its @c geom member would dangle. The metric the matrix-free
  /// Laplacian / gradient read.
  Geometry geom() const { return sys_->grid_context().geom; }
  /// Metric facts captured by generated kernels before entering device lambdas.  Cartesian and polar
  /// Programs share one emitted body; only these geometry-level values select the coordinate metric.
  bool is_polar_geometry() const { return sys_->program_is_polar(); }
  Real radial_origin() const {
    return sys_->program_is_polar() ? sys_->program_polar_geometry().r_min : Real(0);
  }
  Real radial_spacing() const {
    return sys_->program_is_polar() ? sys_->program_polar_geometry().dr() : geom().dx();
  }

  /// out = Lap(in): fill @p in's ghosts (transport BC, periodic by default) then apply the SHARED
  /// discrete 5-point Laplacian (pops::apply_laplacian, all optional coefficients null -> the bare
  /// bit-identical Laplacian). @p in is non-const because the ghost fill WRITES its halos (the valid
  /// cells are unchanged); this is the same matvec idiom the matrix-free Krylov test
  /// (tests/test_generic_krylov.cpp) wraps in its ApplyFn. The compiled program forms an operator
  /// A(in) = in - alpha*Lap(in) by combining this with ctx.lincomb.
  void laplacian(MultiFab& out, MultiFab& in) const {
    count_kernel();
    const GridContext gc = sys_->grid_context();
    fill_ghosts(in, gc.geom.domain, gc.bc);
    if (sys_->program_is_polar()) {
      if (!polar_unit_rr_) {
        polar_unit_rr_ = std::make_shared<MultiFab>(
            in.box_array(), in.dmap(), 1, 1);
        polar_unit_tt_ = std::make_shared<MultiFab>(
            in.box_array(), in.dmap(), 1, 1);
        polar_unit_rr_->set_val(Real(1));
        polar_unit_tt_->set_val(Real(1));
      }
      apply_polar_tensor(in, sys_->program_polar_geometry(), out, polar_unit_rr_.get(),
                         polar_unit_tt_.get(), nullptr, nullptr);
    } else {
      apply_laplacian(in, gc.geom, out);  // all optional pointers null -> bare 5-point Laplacian
    }
  }

  /// out = grad(@p phi) by centered differences: out(.,0) = d phi/dx, out(.,1) = d phi/dy (@p out
  /// needs >= 2 components). Fills @p phi's ghosts then forwards to pops::field_postprocess with
  /// store_phi=false (the gradient lands in components 0/1) and the centered factors cx = 1/(2 dx),
  /// cy = 1/(2 dy) -- the same derivation the elliptic aux post-process uses (+grad sign).
  void gradient(MultiFab& out, MultiFab& phi) const {
    count_kernel();
    const GridContext gc = sys_->grid_context();
    fill_ghosts(phi, gc.geom.domain, gc.bc);
    const Real cx = Real(1) / (Real(2) * gc.geom.dx());
    const Real cy = Real(1) / (Real(2) * gc.geom.dy());
    field_postprocess(phi, out, cx, cy, FieldPostProcess{FieldPostProcess::GradSign::Plus, false});
  }

  /// out = div(@p fx, @p fy) by centered differences: out = d fx/dx + d fy/dy (component 0). The x-flux
  /// is read from component 0 of @p fx and the y-flux from component 1 of @p fy, the SAME layout
  /// @ref gradient writes (d/dx in component 0, d/dy in component 1) -- so chaining ctx.gradient(g, phi)
  /// then ctx.divergence(out, g, g) recovers the 5-point Laplacian. Fills the ghosts of @p fx and @p fy
  /// (transport BC, periodic by default) then forwards to pops::apply_divergence -- the exact inverse
  /// stencil of @ref gradient and the same centered FV divergence the coupled elliptic operator
  /// modules assemble. @p fx and @p fy are non-const because the ghost fill WRITES their halos (the
  /// valid cells are unchanged). A compiled Program forms a tensor flux operator
  /// A(phi) = phi - alpha*div(grad phi) by chaining ctx.gradient then ctx.divergence inside a
  /// matrix-free apply.
  void divergence(MultiFab& out, MultiFab& fx, MultiFab& fy) const {
    count_kernel();
    const GridContext gc = sys_->grid_context();
    fill_ghosts(fx, gc.geom.domain, gc.bc);
    if (&fy != &fx)
      fill_ghosts(fy, gc.geom.domain, gc.bc);  // skip the redundant halo fill when fy aliases fx
    apply_divergence(fx, fy, gc.geom, out, /*cx=*/0, /*cy=*/1);
  }

  /// r <- -div(fx, fy) per conservative component (ADC-419 named fluxes): r(.,c) = -(d fx(.,c)/dx +
  /// d fy(.,c)/dy), centered FV, for every component c of @p r. @p fx and @p fy hold the n_cons x- and
  /// y-flux fields a compiled Program's named-flux kernel wrote (component c = the flux of conservative
  /// component c). REUSES pops::apply_divergence component-by-component (the SAME centered stencil as
  /// @ref divergence, the inverse of @ref gradient -- no new differencing): the ghosts are filled once
  /// per field, then each component's divergence lands in a 1-component scratch and is copied with a
  /// sign flip into @p r. @p fx / @p fy are non-const because the ghost fill writes their halos (the
  /// valid cells are unchanged). This semi-discrete -div F is LINEAR in the flux, so the -div of a SUM
  /// of named fluxes equals the sum of their -div (the named-flux parity guarantee).
  void neg_div_flux_into(MultiFab& r, MultiFab& fx, MultiFab& fy) const {
    count_kernel();
    const GridContext gc = sys_->grid_context();
    fill_ghosts(fx, gc.geom.domain, gc.bc);
    fill_ghosts(fy, gc.geom.domain, gc.bc);
    MultiFab divc(r.box_array(), r.dmap(), 1,
                  0);  // 1-component divergence scratch (no ghosts needed)
    for (int c = 0; c < r.ncomp(); ++c) {
      apply_divergence(fx, fy, gc.geom, divc, /*cx=*/c, /*cy=*/c);  // divc(.,0) = div(fx_c, fy_c)
      for (int li = 0; li < r.local_size(); ++li) {
        const ConstArray4 d = divc.fab(li).const_array();
        Array4 rv = r.fab(li).array();
        const int comp = c;
        for_each_cell(r.box(li), [=] POPS_HD(int i, int j) { rv(i, j, comp) = -d(i, j, 0); });
      }
    }
  }

  /// A zero-initialized RHS scratch with the SAME layout (box array / distribution / ghosts) as @p u,
  /// so the subsequent axpy(u, ., r) combines identical layouts. Records the allocation into the
  /// scratch peak-memory counters (no-op when profiling is off); scratch_state_like forwards here, so
  /// every stage / rhs scratch is counted once at its single allocation site (ADC-459).
  MultiFab rhs_scratch_like(const MultiFab& u) const {
    MultiFab scratch(u.box_array(), u.dmap(), u.ncomp(), u.n_grow());
    count_scratch(scratch);
    return scratch;
  }

  /// A zero-initialized scratch STATE with the same layout as @p u: an intermediate stage state of a
  /// multi-stage scheme (SSPRK/RK). Same allocation as rhs_scratch_like; named for the codegen's
  /// intent. Starts at zero, so a stage `sum_i c_i V_i` is built by axpy-ing each term onto it.
  MultiFab scratch_state_like(const MultiFab& u) const { return rhs_scratch_like(u); }

  /// u <- u + a r over the valid cells (linear combine; forwards to pops::saxpy).
  void axpy(MultiFab& u, Real a, const MultiFab& r) const {
    count_kernel();
    pops::saxpy(u, a, r);
  }

  /// z <- a x + b y over the valid cells (assignment, not accumulation; z may alias x or y).
  /// Forwards to pops::lincomb. The codegen uses it for the committed stage: the block state becomes
  /// z = c_base * z + 1 * acc, where acc holds the non-base terms (self-alias z==x is safe).
  void lincomb(MultiFab& z, Real a, const MultiFab& x, Real b, const MultiFab& y) const {
    count_kernel();
    pops::lincomb(z, a, x, b, y);
  }

  /// Register (idempotent) the history @p name with maximum lag @p lag, allocating the ring buffer
  /// WITHOUT reading it. The codegen emits this ONCE at the top of the step body for each declared
  /// history, so the ring depth is locked before the first store (the cold-start fill then broadcasts
  /// the first stored value into every -- already allocated -- slot). @p ncomp is the slot component
  /// count: the default -1 resolves to block 0's ncomp (the multistep ring, byte-identical), and an
  /// explicit @p ncomp >= 1 sizes a narrower ring (ADC-427: a 1-component cross-step potential carry).
  /// Forwards to System::register_history. A read-only counterpart of @ref history.
  void register_history(const std::string& name, int lag, int ncomp = -1) const {
    sys_->register_history(name, lag, ncomp);
  }

  /// The history slot @p lag macro-steps back (the SYSTEM-OWNED ring buffer, ADC-406a): lag 1 = the
  /// previous step's stored value (e.g. R_{n-1} for Adams-Bashforth), lag 0 = the current slot. The
  /// codegen emits ``ctx.history("<name>", <lag>)``; the read registers the ring on first use
  /// (idempotent) and forwards to System::read_history, which throws if the history was never stored
  /// (spec error 17). The register uses the DEFAULT ncomp (block 0's ncomp) so a bare read never
  /// changes an already-declared ring's width; a narrower ring (ADC-427) is declared by the prelude
  /// register_history(name, lag, ncomp) the codegen emits before any read. @p lag defaults to 1.
  MultiFab& history(const std::string& name, int lag = 1) const {
    sys_->register_history(name, lag);  // idempotent: allocate the ring on first use
    return sys_->read_history(name, lag);
  }

  /// ZERO COLD-START history read (ADC-427): like @ref history, but a read BEFORE the first store
  /// returns the zero-filled slot instead of failing loud. A read-first carry (the cross-step
  /// potential: read the previous step's value at the TOP of the step, store the new one at the END)
  /// has no store before its very first read; its declared step-0 value IS zero (the slots are
  /// zero-initialized at registration), so the first read marks the ring initialized and reads it.
  /// The multistep store-first pattern keeps the fail-loud @ref history read unchanged. @p ncomp
  /// mirrors register_history (binds the slot width at the first register; -1 = block 0's ncomp).
  MultiFab& history_zero_start(const std::string& name, int lag, int ncomp = -1) const {
    sys_->register_history(name, lag, ncomp);  // idempotent; ncomp binds at the first register
    if (!sys_->history_initialized(name))
      sys_->set_history_initialized(name, true);  // the zero-filled slots ARE the declared cold start
    return sys_->read_history(name, lag);
  }

  /// Store @p value into the CURRENT slot of history @p name (ADC-406a). Registers the ring on first
  /// use (at least a current slot; the lag the program reads via @ref history sets the real depth) and
  /// forwards to System::store_history (which fills every slot on the first store -- the cold start).
  /// The codegen emits ``ctx.store_history("<name>", <value>)`` near the end of the step body. Uses the
  /// default ncomp on register (the width is fixed by the prelude register_history the codegen emits).
  void store_history(const std::string& name, const MultiFab& value) const {
    sys_->register_history(name, 1);  // idempotent: at least a current slot exists before the store
    sys_->store_history(name, value);
  }

  /// Shift every history ring one macro-step (slot k <- slot k-1). Forwards to
  /// System::rotate_histories. The codegen emits ``ctx.rotate_histories()`` as the LAST statement of
  /// the step body (after the commit), so the next step reads lag k as the value k stores ago.
  void rotate_histories() const { sys_->rotate_histories(); }

  /// @name Reductions (spec op 16)
  /// COLLECTIVE all_reduce over one component of a field (sum / signed max / signed min). The codegen
  /// lowers P.sum / P.sum_component / P.max / P.min DIRECTLY to the pops:: free functions (like norm2 ->
  /// pops::dot), but these wrappers expose them on the context for hand-rolled C++ stages and mirror
  /// norm2 / dot above. MANDATORY UNDER MPI: called on EVERY rank (empty ranks included), like dot.
  /// @{
  Real sum_component(const MultiFab& u, int comp) const { return pops::reduce_sum(u, comp); }
  Real max_component(const MultiFab& u, int comp) const { return pops::reduce_max(u, comp); }
  Real min_component(const MultiFab& u, int comp) const { return pops::reduce_min(u, comp); }
  /// L1 (absolute-sum) reduction Sum_cells |u(.,.,comp)| over one component -- P.norm1 / Norm(L1).
  Real abs_sum_component(const MultiFab& u, int comp) const { return pops::reduce_abs_sum(u, comp); }
  Real sum(const MultiFab& u) const { return pops::reduce_sum(u, 0); }
  Real max(const MultiFab& u) const { return pops::reduce_max(u, 0); }
  Real min(const MultiFab& u) const { return pops::reduce_min(u, 0); }
  Real abs_sum(const MultiFab& u) const { return pops::reduce_abs_sum(u, 0); }
  /// @}

  /// Fill the ghost cells (halos) of @p x in place: the transport BC (periodic by default), the SAME
  /// exchange laplacian / gradient / divergence run internally before differencing (spec op 22). The
  /// valid cells are untouched; only the halos change. Forwards to the shared pops::fill_ghosts.
  void fill_boundary(MultiFab& x) const {
    const GridContext gc = sys_->grid_context();
    fill_ghosts(x, gc.geom.domain, gc.bc);
  }

  /// Apply block @p b's post-step positivity projection to @p u in place: U <- project(U, aux) over the
  /// valid cells, the SAME Zhang-Shu / floor projection the native per-step path runs (ADC-177, spec
  /// op 21). REUSES the block's own projection closure (set at add_block time); a block WITHOUT a
  /// projection is a no-op. Forwards to System::block_project -- it reimplements no positivity.
  void apply_projection(int b, MultiFab& u) const { sys_->block_project(sys_block(b), u); }

  /// Store a runtime Scalar @p value into the System diagnostics map under @p name (spec op 23),
  /// retrievable after the step via System::program_diagnostic / program_diagnostics (exposed to
  /// Python as sim.program_diagnostic / sim.program_diagnostics). A pure side effect: the scalar is
  /// recorded for inspection / logging, it does not feed the numerics. Forwards to
  /// System::record_program_diagnostic.
  void record_scalar(const std::string& name, Real value) const {
    sys_->record_program_diagnostic(name, value);
  }

  /// The CURRENT RuntimeParams of PROGRAM block @p b (epic ADC-479 / ADC-510, Spec 5 C5): the
  /// per-block runtime-parameter values a compiled Program's lowered source / linear-source kernel
  /// reads via ``params.get(<index>)``. The codegen binds ``const pops::RuntimeParams params =
  /// ctx.program_params(<b>);`` ONCE per fab (outside the per-cell loop), then the device lambda
  /// captures it by value -- trivially copyable, device-clean, ``get()`` is POPS_HD. @p b is the
  /// PROGRAM block index (P.state declaration order, the index install_program seeded), NOT routed
  /// through sys_block: the System keys the store by program index, the same index Python's params
  /// route writes via set_program_params. A block with no runtime param returns a default
  /// RuntimeParams (count 0). Forwards to System::program_params; the value reflects the LATEST
  /// set_program_params (no recompile), since the store lives on the System the captured ctx points at.
  RuntimeParams program_params(int b) const { return sys_->program_params(b); }

  /// @name Per-node profiling (Spec 3 section 29, ADC-459)
  /// Time a single Program node into the System Profiler, so sim.profile_report() shows per-node
  /// times ("node:rhs2", "node:solve_fields1", ...) alongside the coarse "step" / "field_solve"
  /// phases. The Profiler is disabled by default; both calls are ~free when off (a ProfileScope still
  /// reads the clock twice -- wrap a per-node scope, the intended granularity, not the inner loops).
  /// @{
  /// The System Profiler (non-owning). A hand-written C++ stage can construct its own ProfileScope on
  /// it; the codegen uses profile_record below (which preserves the step body's C++ variable scope).
  runtime::program::Profiler& profiler() const { return sys_->profiler(); }
  /// RAII timer for one node: ``pops::runtime::program::ProfileScope s = ctx.profile_node("node:x");``
  /// times its own lifetime into the System Profiler. For a hand-rolled C++ stage that can wrap a whole
  /// block; the generated step body cannot use it (a node's emitted C++ declarations must outlive the
  /// node), so the codegen pairs a steady_clock now() with profile_record instead.
  runtime::program::ProfileScope profile_node(const std::string& name) const {
    return runtime::program::ProfileScope(sys_->profiler(), name);
  }
  /// Record one node's elapsed time (now() - @p t0) under @p name into the System Profiler. The
  /// generated step body captures @p t0 = std::chrono::steady_clock::now() BEFORE the node's
  /// statements and calls this AFTER them, so the node's C++ declarations stay at body scope (a
  /// surrounding RAII block would hide them from later nodes). No-op contribution when profiling is
  /// off (Profiler::record early-returns); the only cost is one extra clock read per node.
  void profile_record(const std::string& name, std::chrono::steady_clock::time_point t0) const {
    const auto t1 = std::chrono::steady_clock::now();
    sys_->profiler().record(name, std::chrono::duration<double>(t1 - t0).count());
  }
  /// @}

  /// @name Profiling counters (Spec 3 section 29, ADC-459)
  /// The named integer counters sim.profile_report() surfaces alongside the per-node timings: how many
  /// kernel launches a step issued, how the held-node scheduler hit/missed its cache, and the scratch
  /// peak memory. Each helper is a single predictable branch when profiling is off (Profiler::count /
  /// count_max early-return), so the hot path pays nothing unless sim.enable_profiling() ran. These move
  /// only on the COMPILED-PROGRAM path (a problem.so step body calling these seam ops); the native step
  /// counts "kernels" at its own elliptic-solve chokepoint instead (System::Impl::solve_fields).
  /// @{
  /// One per kernel-dispatching seam op (a -div F / source / matvec / solve). The compiled step body
  /// reaches the seam through these methods, so counting at this op granularity counts the per-node
  /// kernel LAUNCHES (the device dispatch in mesh/execution/for_each.hpp is a shared free function with
  /// no profiler handle -- instrumenting it would touch every numerics TU and add a hidden hot-path
  /// argument, so the op-granularity count is the deliberate, labeled choice, Spec 3 section 29).
  void count_kernel(std::int64_t by = 1) const { sys_->profiler().count("kernels", by); }
  /// Record one scratch MultiFab allocation: bumps the allocation count and updates the byte peak with
  /// THIS buffer's footprint. The peak is the largest SINGLE scratch (a deep allocation); a running
  /// "live total" is not tracked because the seam hands the buffer to the caller (no free hook here),
  /// so we report what is exactly knowable -- the allocation count and the largest one -- never a faked
  /// live-bytes figure (Spec 3 section 29 scratch peak memory).
  void count_scratch(const MultiFab& mf) const {
    runtime::program::Profiler& prof = sys_->profiler();
    if (!prof.enabled()) {
      return;  // skip the byte-summing loop entirely when profiling is off (zero hot-path cost)
    }
    prof.count("scratch_allocs");
    std::int64_t bytes = 0;
    for (int li = 0; li < mf.local_size(); ++li) {
      bytes += mf.fab(li).size() * static_cast<std::int64_t>(sizeof(Real));
    }
    prof.count_max("scratch_peak_bytes", bytes);
  }
  /// @}

  /// @name Scheduler value cache (Spec 3 section 17-18, ADC-458)
  /// A held field-solve node recomputes only when DUE (every N macro-steps) and reuses the cached
  /// System aux (phi / grad / E) in between. The cache is owned by the System (one CacheManager per
  /// installed Program, keyed by the Program node id) so the checkpoint can reach it (Spec 3 section
  /// 30); every ProgramContext copy forwards to that single manager via sys_->program_cache(). The
  /// codegen wraps a held solve_fields in
  /// ``if (cache_should_update(id, N)) { solve_fields_from_state(...); cache_store_aux(id); }
  ///  else { cache_restore_aux(id); }``. The runtime cadence/checkpoint is exercised in a compiled
  /// .so step loop (validated on ROMEO; not buildable on a host-only Mac).
  /// @{
  /// True if node @p node_id is due to recompute at the current macro step: cold start (never stored),
  /// then every @p every_n macro steps. Wraps CacheManager::is_due with System::macro_step().
  ///
  /// PROFILER scheduler counters (ADC-459, Spec 3 section 29): a DUE step recomputes the node (a cache
  /// "miss" + a "due" scheduled node); a NOT-due step reuses the held value (a cache "hit" + a
  /// "skipped" scheduled node). Counted here at the one decision point every scheduled node routes
  /// through, gated on the profiler (zero cost when off). These move only under the compiled .so step
  /// loop that exercises a held schedule (validated on Kokkos/ROMEO, not buildable host-only).
  bool cache_should_update(int node_id, int every_n) const {
    const bool due = sys_->program_cache().is_due(node_id, sys_->macro_step(), every_n);
    if (due) {
      sys_->profiler().count("cache_misses");
      sys_->profiler().count("nodes_due");
    } else {
      sys_->profiler().count("cache_hits");
      sys_->profiler().count("nodes_skipped");
    }
    return due;
  }
  /// Store a copy of the System aux (the field solve's output) as node @p node_id's cached value,
  /// stamped at the current macro step (resets its accumulated dt).
  void cache_store_aux(int node_id) const {
    sys_->program_cache().store(node_id, *sys_->grid_context().aux, sys_->macro_step());
  }
  /// Restore node @p node_id's cached aux into the System aux (a held step: no elliptic solve).
  void cache_restore_aux(int node_id) const {
    *sys_->grid_context().aux = sys_->program_cache().retrieve(node_id);
  }

  /// Store a copy of a NAMED scratch MultiFab (a held rhs / source / linear_combine output) as node
  /// @p node_id's cached value, stamped at the current macro step. The aux variants cache the System
  /// aux; this caches an arbitrary step-body scratch so ANY schedulable node can hold, not only a
  /// field solve.
  void cache_store_scratch(int node_id, const MultiFab& scratch) const {
    sys_->program_cache().store(node_id, scratch, sys_->macro_step());
  }
  /// Restore node @p node_id's cached scratch into @p scratch (a held step: no recompute).
  void cache_restore_scratch(int node_id, MultiFab& scratch) const {
    scratch = sys_->program_cache().retrieve(node_id);
  }
  /// The current macro step (0-based). Mirrors System::macro_step(); the codegen lowers on_start() to
  /// ``ctx.macro_step() == 0`` and reads it for any step-indexed predicate.
  int macro_step() const { return sys_->macro_step(); }
  /// Add a skipped step's @p dt to node @p node_id's accumulator (accumulate_dt policy): on a NOT-due
  /// step the held node does not recompute but records the dt so the next due step sees the full
  /// skipped interval. Variable step_cfl safe (the actual skipped dt, not N * dt_current).
  void cache_accumulate_dt(int node_id, Real dt) const {
    sys_->program_cache().accumulate_dt(node_id, dt);
  }
  /// The effective dt a due accumulate_dt step applies: @p dt_now plus the summed skipped dt since the
  /// last recompute (resets the accumulator). The codegen feeds this as the step's dt into the held
  /// node's recompute so it advances over the whole skipped interval at once.
  Real cache_effective_dt(int node_id, Real dt_now) const {
    return sys_->program_cache().effective_dt(node_id, dt_now);
  }
  /// Fail loud: a node with an `error` policy was reached off its schedule cadence (a stale value would
  /// be read). The codegen emits this on the not-due branch of an `error`-policy node.
  [[noreturn]] void scheduler_error(const std::string& what) const {
    throw std::runtime_error("pops Program scheduler: " + what);
  }
  /// @}

 private:
  static std::runtime_error block_map_error_(std::string message) {
    return std::runtime_error(std::move(message));
  }

  mutable std::array<MultiFab*, 4> polar_coeffs_{{nullptr, nullptr, nullptr, nullptr}};
  mutable std::shared_ptr<PolarTensorKrylovSolver> polar_tensor_;
  mutable std::shared_ptr<MultiFab> polar_unit_rr_;
  mutable std::shared_ptr<MultiFab> polar_unit_tt_;
  System* sys_;
};

}  // namespace program
}  // namespace runtime
}  // namespace pops
