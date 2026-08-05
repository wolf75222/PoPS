/// @file
/// @brief Exact compile-time-ranked AMR facade over runtime::amr::AmrRuntime<Dim>.

#include <pops/runtime/amr_system.hpp>

#include <pops/amr/hierarchy/amr_hierarchy.hpp>
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/output_piece_collective.hpp>
#include <pops/runtime/program/program_runtime_state.hpp>
#include <pops/runtime/program/profiler.hpp>
#include <pops/runtime/system/system_boundary_registry.hpp>
#include <pops/runtime/system/system_lifecycle.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops {
namespace {

inline void require_amr_assembling(
    const runtime::system::SystemLifecycle& lifecycle, const char* operation) {
  if (lifecycle.frozen())
    throw std::runtime_error(std::string("AmrSystem::") + operation +
                             ": composition is frozen after bind");
}

template <int Dim>
void validate_amr_config(const AmrSystemConfig<Dim>& config) {
  config.validate_spatial_domain();
  if (config.coordinate_system != runtime_config_detail::cartesian_coordinate_system<Dim>())
    throw std::invalid_argument(
        "AmrSystem Cartesian core requires a dimension-qualified Cartesian provider");
  if (config.level_count < 1)
    throw std::invalid_argument("AmrSystem level_count must be positive");
  if (config.regrid_every < 0)
    throw std::invalid_argument("AmrSystem regrid_every must be non-negative");

  const std::size_t transitions = static_cast<std::size_t>(config.level_count - 1);
  if (config.transition_ratios.size() != transitions ||
      config.transition_buffers.size() != transitions ||
      config.transition_lookaheads.size() != transitions)
    throw std::invalid_argument(
        "AmrSystem transition tables must contain exactly level_count - 1 ranked rows");
  for (std::size_t transition = 0; transition < transitions; ++transition)
    for (int axis = 0; axis < Dim; ++axis) {
      if (config.transition_ratios[transition][axis] < 2 ||
          config.transition_ratios[transition][axis] > std::numeric_limits<int>::max())
        throw std::invalid_argument(
            "AmrSystem transition refinement ratios must be at least two on every axis");
      if (config.transition_buffers[transition][axis] < 0 ||
          config.transition_lookaheads[transition][axis] < 0)
        throw std::invalid_argument(
            "AmrSystem transition buffers and lookaheads must be non-negative");
    }
  for (int axis = 0; axis < Dim; ++axis)
    if (config.coarse_max_grid[axis] < 0)
      throw std::invalid_argument("AmrSystem coarse_max_grid must be non-negative");
}

template <int Dim>
mesh::RankSpace<Dim> process_rank_space() {
  Extent<Dim> shape = runtime_config_detail::filled_extent<Dim>(1);
  shape[0] = n_ranks();
  return mesh::RankSpace<Dim>(Index<Dim>{}, shape);
}

template <int Dim>
std::int64_t checked_layout_cells(const mesh::BoxArray<Dim>& layout) {
  std::int64_t total = 0;
  for (const Box<Dim>& box : layout.boxes()) {
    const std::int64_t cells = box.numPts();
    if (cells < 1 || cells > std::numeric_limits<std::int64_t>::max() - total)
      throw std::overflow_error("AmrSystem layout cell budget exceeds int64_t");
    total += cells;
  }
  return total;
}

inline std::size_t checked_pair_count(std::size_t count) {
  if (count > 1 && count - 1 > std::numeric_limits<std::size_t>::max() / count)
    throw std::length_error("AmrSystem layout pair budget exceeds size_t");
  return count < 2 ? 0 : count * (count - 1) / 2;
}

inline std::size_t checked_square_count(std::size_t count) {
  if (count != 0 && count > std::numeric_limits<std::size_t>::max() / count)
    throw std::length_error("AmrSystem hierarchy pair budget exceeds size_t");
  return count * count;
}

template <int Dim>
std::size_t checked_cells(const Box<Dim>& box) {
  const std::int64_t cells = box.numPts();
  if (cells < 0 || static_cast<std::uint64_t>(cells) >
                       static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    throw std::overflow_error("AmrSystem ranked field exceeds size_t");
  return static_cast<std::size_t>(cells);
}

template <int Dim>
Index<Dim> unflatten(const Box<Dim>& box, std::size_t linear) {
  Index<Dim> index{};
  for (int axis = 0; axis < Dim; ++axis) {
    const std::size_t extent = static_cast<std::size_t>(box.length(axis));
    index[axis] = box.lo[axis] + static_cast<int>(linear % extent);
    linear /= extent;
  }
  return index;
}

template <int Dim>
std::size_t offset(const Index<Dim>& index, const Box<Dim>& box) {
  std::size_t result = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    result += static_cast<std::size_t>(index[axis] - box.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(box.length(axis));
  }
  return result;
}

template <int Dim>
std::vector<double> gather_field(const MultiFab<Dim>& field, const Box<Dim>& domain,
                                 int components) {
  if (components < 1 || components > field.ncomp())
    throw std::invalid_argument("AmrSystem gather component count is invalid");
  const std::size_t domain_cells = checked_cells(domain);
  if (static_cast<std::size_t>(components) >
      std::numeric_limits<std::size_t>::max() / domain_cells)
    throw std::overflow_error("AmrSystem gather buffer exceeds size_t");
  std::vector<double> result(static_cast<std::size_t>(components) * domain_cells, 0.0);
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const Fab<Dim>& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& valid = fab.box();
    const Box<Dim>& grown = fab.grown_box();
    const std::size_t local_cells = checked_cells(valid);
    const std::size_t component_stride = checked_cells(grown);
    for (int component = 0; component < components; ++component)
      for (std::size_t linear = 0; linear < local_cells; ++linear) {
        const Index<Dim> index = unflatten(valid, linear);
        result[static_cast<std::size_t>(component) * domain_cells + offset(index, domain)] =
            static_cast<double>(host(static_cast<std::size_t>(component) * component_stride +
                                     offset(index, grown)));
      }
  }
  all_reduce_sum_inplace(result.data(), result.size());
  return result;
}

template <int Dim>
void write_field(MultiFab<Dim>& field, const Box<Dim>& domain,
                 const std::vector<double>& values, int components) {
  const std::size_t domain_cells = checked_cells(domain);
  if (components < 1 || components > field.ncomp() ||
      values.size() != static_cast<std::size_t>(components) * domain_cells)
    throw std::invalid_argument("AmrSystem field input differs from its exact ranked shape");
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    Fab<Dim>& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& valid = fab.box();
    const Box<Dim>& grown = fab.grown_box();
    const std::size_t local_cells = checked_cells(valid);
    const std::size_t component_stride = checked_cells(grown);
    for (int component = 0; component < components; ++component)
      for (std::size_t linear = 0; linear < local_cells; ++linear) {
        const Index<Dim> index = unflatten(valid, linear);
        host(static_cast<std::size_t>(component) * component_stride + offset(index, grown)) =
            static_cast<Real>(values[static_cast<std::size_t>(component) * domain_cells +
                                     offset(index, domain)]);
      }
    fab.copy_from_host(host);
  }
}

template <int Dim>
void write_component(MultiFab<Dim>& field, const Box<Dim>& domain,
                     const std::vector<double>& values, int component) {
  const std::size_t domain_cells = checked_cells(domain);
  if (component < 0 || component >= field.ncomp() || values.size() != domain_cells)
    throw std::invalid_argument("AmrSystem component input differs from its exact ranked shape");
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    Fab<Dim>& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& valid = fab.box();
    const Box<Dim>& grown = fab.grown_box();
    const std::size_t local_cells = checked_cells(valid);
    const std::size_t component_stride = checked_cells(grown);
    for (std::size_t linear = 0; linear < local_cells; ++linear) {
      const Index<Dim> index = unflatten(valid, linear);
      host(static_cast<std::size_t>(component) * component_stride + offset(index, grown)) =
          static_cast<Real>(values[offset(index, domain)]);
    }
    fab.copy_from_host(host);
  }
}

template <int Dim>
double cell_measure(const AmrSystemConfig<Dim>& config, const Box<Dim>& domain) {
  double measure = 1.0;
  for (int axis = 0; axis < Dim; ++axis)
    measure *= static_cast<double>(config.upper[axis] - config.lower[axis]) /
               static_cast<double>(domain.length(axis));
  return measure;
}

}  // namespace

template <int Dim>
struct AmrSystem<Dim>::Impl {
  using engine_type = runtime::amr::AmrRuntime<Dim>;
  using field_type = MultiFab<Dim>;
  using boundary_registry_type = runtime::system::SystemBoundaryRegistry<Dim>;

  struct BlockSpec {
    std::string name;
    int ncomp = 0;
    double gamma = static_cast<double>(kPhysicalDefaultGamma);
    int substeps = 1;
    int stride = 1;
    int required_ghost_depth = 1;
    std::string time = "euler";
    bool has_density = false;
    std::vector<double> density;
    bool has_state = false;
    std::vector<double> state;
  };

  struct GlobalDtBound {
    std::string label;
    std::function<double()> evaluate;
  };

  AmrSystemConfig<Dim> cfg;
  std::shared_ptr<const PreparedLoadBalanceAuthority<Dim>> load_balance;
  std::vector<BlockSpec> blocks;
  boundary_registry_type boundary_registry;
  runtime::program::ProgramRuntimeState<Dim> program;
  runtime::system::SystemLifecycle lifecycle;
  mutable std::unique_ptr<engine_type> engine;
  std::vector<GlobalDtBound> dt_bounds;
  std::vector<CouplingOperatorView> coupling_views;
  double accepted_time = 0.0;
  int macro_step = 0;
  std::string last_dt_reason;
  std::vector<std::uint8_t> program_accepted_bytes;
  std::uint64_t program_accepted_revision = 0;

  struct AcceptedSnapshot {
    std::optional<typename engine_type::Snapshot> engine;
    runtime::program::ProgramRuntimeState<Dim> program;
    double accepted_time = 0.0;
    int macro_step = 0;
    std::vector<std::uint8_t> program_accepted_bytes;
    std::uint64_t program_accepted_revision = 0;

    explicit AcceptedSnapshot(const Impl& owner)
        : engine(owner.engine ? std::optional<typename engine_type::Snapshot>(
                                   owner.engine->snapshot())
                              : std::nullopt),
          program(owner.program),
          accepted_time(owner.accepted_time),
          macro_step(owner.macro_step),
          program_accepted_bytes(owner.program_accepted_bytes),
          program_accepted_revision(owner.program_accepted_revision) {}

    void restore(Impl& owner) {
      if (engine.has_value() != static_cast<bool>(owner.engine))
        throw std::logic_error("AmrSystem transaction changed engine materialization");
      if (engine)
        owner.engine->restore(*engine);
      owner.program = program;
      owner.accepted_time = accepted_time;
      owner.macro_step = macro_step;
      owner.program_accepted_bytes = program_accepted_bytes;
      owner.program_accepted_revision = program_accepted_revision;
    }
  };

  std::unique_ptr<AcceptedSnapshot> external_step_transaction;
  bool external_step_committed = false;

  explicit Impl(const AmrSystemConfig<Dim>& config)
      : cfg(config),
        load_balance(std::make_shared<const PreparedLoadBalanceAuthority<Dim>>(
            prepare_load_balance_authority<Dim>(cfg.load_balance_route, cfg.load_balance_identity,
                                                cfg.load_balance_options))) {}

  BlockSpec& block(const std::string& name) {
    for (BlockSpec& candidate : blocks)
      if (candidate.name == name || (blocks.size() == 1 && name.empty()))
        return candidate;
    throw std::runtime_error("AmrSystem has no block named '" + name + "'");
  }

  const BlockSpec& block(const std::string& name) const {
    return const_cast<Impl*>(this)->block(name);
  }

  void ensure_engine() const {
    if (engine)
      return;
    if (blocks.empty())
      throw std::logic_error("AmrSystem requires a dimension-qualified block before materialization");
    if (blocks.size() != 1)
      throw std::logic_error(
          "AmrSystem exact-ranked core requires a prepared multi-block hierarchy provider");

    const BlockSpec& block = blocks.front();
    const Box<Dim> domain = cfg.index_domain();
    const mesh::BoxArray<Dim> patches(cfg.materialized_boxes());
    const mesh::RankSpace<Dim> ranks = process_rank_space<Dim>();
    const Index<Dim> local_rank = ranks.coordinate(static_cast<std::size_t>(my_rank()));
    mesh::Distribution<Dim> distribution;
    if (cfg.distribute_coarse) {
      const parallel::LoadBalancePreparationBudget budget{
          patches.size(), ranks.size(), checked_layout_cells(patches)};
      distribution = load_balance->prepare(patches, ranks, budget).plan().distribution();
    } else {
      distribution = mesh::Distribution<Dim>::replicated(patches, ranks);
    }

    const mesh::BoxArrayValidationBudget layout_budget{patches.size(),
                                                       checked_pair_count(patches.size())};
    const amr::hierarchy::LevelLayout<Dim> coarse(
        0, domain, patches, distribution, amr::RefinementRatio<Dim>{}, layout_budget);
    const Extent<Dim> ghosts =
        runtime_config_detail::filled_extent<Dim>(block.required_ghost_depth);
    field_type state(patches, distribution, local_rank, block.ncomp, ghosts);
    if (block.has_state)
      write_field(state, domain, block.state, block.ncomp);
    else if (block.has_density)
      write_component(state, domain, block.density, 0);

    const std::size_t hierarchy_pairs =
        cfg.level_count == 1
            ? 0
            : std::max(checked_square_count(patches.size()), patches.size());
    auto hierarchy = amr::hierarchy::AmrHierarchy<Dim>::from_coarse(
        coarse, std::move(state),
        amr::hierarchy::HierarchyValidationBudget{static_cast<std::size_t>(cfg.level_count),
                                                  hierarchy_pairs});
    engine = std::make_unique<engine_type>(std::move(hierarchy), load_balance,
                                           "pops.amr-system.exact-ranked");
  }

  template <class Function>
  decltype(auto) execute_transaction(Function&& function) {
    ensure_engine();
    AcceptedSnapshot snapshot(*this);
    try {
      return std::forward<Function>(function)();
    } catch (...) {
      snapshot.restore(*this);
      throw;
    }
  }
};

template <int Dim>
AmrSystem<Dim>::AmrSystem(const AmrSystemConfig<Dim>& config) {
  validate_amr_config(config);
  p_ = std::make_unique<Impl>(config);
}

template <int Dim>
AmrSystem<Dim>::~AmrSystem() = default;

template <int Dim>
AmrSystem<Dim>::AmrSystem(AmrSystem&&) noexcept = default;

template <int Dim>
AmrSystem<Dim>& AmrSystem<Dim>::operator=(AmrSystem&&) noexcept = default;

template <int Dim>
void AmrSystem<Dim>::add_block(const std::string&, const ModelSpec&, const std::string&,
                               const std::string&, const std::string&, const std::string&, int, int,
                               const std::vector<std::string>&,
                               const std::vector<std::string>&, const NewtonOptions&, bool, double,
                               double, bool) {
  throw std::runtime_error(
      "AmrSystem ModelSpec installation requires a dimension-qualified compiled block provider");
}

template <int Dim>
void AmrSystem<Dim>::set_compiled_block(
    int ncomp, double gamma, int substeps, AmrCompiledBlockBuilder<Dim> runtime_builder,
    const std::string& name, bool, const std::string& time, int stride,
    const std::vector<std::string>& implicit_vars,
    const std::vector<std::string>& implicit_roles, double, double, bool) {
  require_amr_assembling(p_->lifecycle, "set_compiled_block");
  if (p_->engine)
    throw std::runtime_error("AmrSystem cannot install a block after hierarchy materialization");
  if (!runtime_builder || ncomp < 1 || substeps < 1 || stride < 1 || name.empty())
    throw std::invalid_argument("AmrSystem compiled block contract is incomplete");
  if (!implicit_vars.empty() || !implicit_roles.empty())
    throw std::invalid_argument(
        "AmrSystem compiled block has no dimension-qualified partial-implicit provider");
  if (!p_->blocks.empty())
    throw std::runtime_error(
        "AmrSystem multi-block installation requires a prepared exact-ranked hierarchy provider");
  (void)runtime_builder;
  p_->blocks.push_back(typename Impl::BlockSpec{name, ncomp, gamma, substeps, stride, 1, time});
}

template <int Dim>
void AmrSystem<Dim>::install_block_state_route(const std::string& name,
                                               const std::string& state_identity) {
  require_amr_assembling(p_->lifecycle, "install_block_state_route");
  p_->boundary_registry.install_state_route(name, state_identity);
}

template <int Dim>
void AmrSystem<Dim>::install_field_storage_route(const std::string& field_identity,
                                                 const std::string& provider_slot) {
  require_amr_assembling(p_->lifecycle, "install_field_storage_route");
  p_->boundary_registry.install_field_storage_route(field_identity, provider_slot);
}

template <int Dim>
void AmrSystem<Dim>::install_hyperbolic_boundary(
    const std::string& name, const std::string& identity, int required_depth,
    const std::vector<std::string>& face_types, const std::vector<double>& face_values,
    const std::vector<std::string>& face_identities,
    const std::vector<std::string>& component_roles, const std::string& state_identity,
    const std::vector<std::string>& face_representations,
    const std::vector<std::string>& face_converter_identities,
    const std::vector<std::vector<std::string>>& face_analytic_opcodes,
    const std::vector<std::vector<double>>& face_analytic_literals,
    const std::vector<std::string>& face_analytic_clocks) {
  require_amr_assembling(p_->lifecycle, "install_hyperbolic_boundary");
  p_->boundary_registry.install_boundary(
      name, identity, required_depth, face_types, face_values, face_identities, component_roles,
      state_identity, face_representations, face_converter_identities, face_analytic_opcodes,
      face_analytic_literals, face_analytic_clocks);
  typename Impl::BlockSpec& block = p_->block(name);
  if (p_->engine && required_depth > block.required_ghost_depth)
    throw std::runtime_error(
        "AmrSystem cannot widen boundary storage after hierarchy materialization");
  block.required_ghost_depth = std::max(block.required_ghost_depth, required_depth);
}

template <int Dim>
void AmrSystem<Dim>::install_prepared_hyperbolic_boundary(
    const std::string& name, const std::string& identity, int required_depth,
    const std::string& state_identity, std::shared_ptr<const HyperbolicBoundary> boundary) {
  require_amr_assembling(p_->lifecycle, "install_prepared_hyperbolic_boundary");
  p_->boundary_registry.install_boundary(name, identity, required_depth, state_identity,
                                         std::move(boundary));
  typename Impl::BlockSpec& block = p_->block(name);
  if (p_->engine && required_depth > block.required_ghost_depth)
    throw std::runtime_error(
        "AmrSystem cannot widen boundary storage after hierarchy materialization");
  block.required_ghost_depth = std::max(block.required_ghost_depth, required_depth);
}

template <int Dim>
void AmrSystem<Dim>::discard_hyperbolic_boundaries() {
  require_amr_assembling(p_->lifecycle, "discard_hyperbolic_boundaries");
  p_->boundary_registry.discard_transaction();
}

template <int Dim>
void AmrSystem<Dim>::set_density(const std::string& name, const std::vector<double>& density) {
  typename Impl::BlockSpec& block = p_->block(name);
  if (density.size() != checked_cells(p_->cfg.index_domain()))
    throw std::invalid_argument("AmrSystem density differs from the exact coarse shape");
  if (p_->engine)
    write_component(p_->engine->hierarchy().state(0), p_->cfg.index_domain(), density, 0);
  block.density = density;
  block.has_density = true;
}

template <int Dim>
void AmrSystem<Dim>::set_conservative_state(const std::string& name,
                                            const std::vector<double>& state) {
  typename Impl::BlockSpec& block = p_->block(name);
  const std::size_t cells = checked_cells(p_->cfg.index_domain());
  if (state.size() != static_cast<std::size_t>(block.ncomp) * cells)
    throw std::invalid_argument("AmrSystem state differs from the exact coarse shape");
  if (p_->engine)
    write_field(p_->engine->hierarchy().state(0), p_->cfg.index_domain(), state, block.ncomp);
  block.state = state;
  block.has_state = true;
}

template <int Dim>
void AmrSystem<Dim>::add_dt_bound(const std::string& label, std::function<double()> evaluate) {
  require_amr_assembling(p_->lifecycle, "add_dt_bound");
  if (label.empty() || !evaluate)
    throw std::invalid_argument("AmrSystem dt bound requires a label and provider");
  p_->dt_bounds.push_back(typename Impl::GlobalDtBound{label, std::move(evaluate)});
}

template <int Dim>
std::string AmrSystem<Dim>::last_dt_bound() const {
  return p_->last_dt_reason;
}

template <int Dim>
void AmrSystem<Dim>::step(double dt) {
  p_->program.require_step_installed("AmrSystem::step");
  runtime::program::ProfileScope scope(p_->program.profiler_, "step");
  p_->program.profiler_.count("steps");
  p_->execute_transaction([&] {
    p_->program.dispatch_cadence_step(p_->accepted_time, p_->macro_step, dt, "AmrSystem");
  });
}

template <int Dim>
void AmrSystem<Dim>::advance(double dt, int nsteps) {
  p_->program.require_step_installed("AmrSystem::advance");
  if (nsteps < 0)
    throw std::invalid_argument("AmrSystem::advance requires a non-negative step count");
  for (int step_index = 0; step_index < nsteps; ++step_index)
    step(dt);
}

template <int Dim>
double AmrSystem<Dim>::step_cfl(double cfl, double speed_floor, double max_dt, double min_dt) {
  p_->program.require_step_installed("AmrSystem::step_cfl");
  if (!std::isfinite(cfl) || cfl <= 0.0 || !std::isfinite(speed_floor) || speed_floor <= 0.0)
    throw std::invalid_argument("AmrSystem::step_cfl requires positive finite CFL inputs");
  if (std::isnan(max_dt) || max_dt <= 0.0 || !std::isfinite(min_dt) || min_dt < 0.0)
    throw std::invalid_argument("AmrSystem::step_cfl received invalid strategy bounds");

  double spacing = std::numeric_limits<double>::infinity();
  for (int axis = 0; axis < Dim; ++axis)
    spacing = std::min(spacing, static_cast<double>(p_->cfg.upper[axis] - p_->cfg.lower[axis]) /
                                    static_cast<double>(p_->cfg.shape[axis]));
  double selected = cfl * spacing / speed_floor;
  std::string reason = "degenerate";
  for (const typename Impl::GlobalDtBound& bound : p_->dt_bounds) {
    double candidate = bound.evaluate();
    if (!(candidate > 0.0) || !std::isfinite(candidate))
      candidate = std::numeric_limits<double>::infinity();
    candidate = all_reduce_min(candidate);
    if (candidate < selected) {
      selected = candidate;
      reason = "global:" + bound.label;
    }
  }
  if (p_->program.dt_bound_) {
    const double candidate =
        static_cast<double>(p_->program.dt_bound_(static_cast<Real>(cfl)));
    if (std::isfinite(candidate) && candidate > 0.0 && candidate < selected) {
      selected = candidate;
      reason = "program:dt_bound";
    }
  }
  if (max_dt < selected) {
    selected = max_dt;
    reason = "strategy:max_dt";
  }
  if (selected < min_dt)
    throw std::runtime_error("AmrSystem::step_cfl stability bound is below declared min_dt");
  p_->last_dt_reason = std::move(reason);
  step(selected);
  return selected;
}

template <int Dim>
void AmrSystem<Dim>::begin_step_transaction() {
  p_->ensure_engine();
  if (p_->external_step_transaction)
    throw std::runtime_error("AmrSystem step transaction is already active");
  p_->external_step_transaction = std::make_unique<typename Impl::AcceptedSnapshot>(*p_);
  p_->external_step_committed = false;
}

template <int Dim>
void AmrSystem<Dim>::commit_step_transaction() {
  if (!p_->external_step_transaction || p_->external_step_committed)
    throw std::runtime_error("AmrSystem has no active uncommitted step transaction");
  p_->external_step_committed = true;
}

template <int Dim>
void AmrSystem<Dim>::finalize_step_transaction() {
  if (!p_->external_step_transaction || !p_->external_step_committed)
    throw std::runtime_error("AmrSystem has no committed step transaction");
  p_->external_step_transaction.reset();
  p_->external_step_committed = false;
}

template <int Dim>
void AmrSystem<Dim>::rollback_step_transaction() {
  if (!p_->external_step_transaction)
    throw std::runtime_error("AmrSystem has no active step transaction");
  p_->external_step_transaction->restore(*p_);
  p_->external_step_transaction.reset();
  p_->external_step_committed = false;
}

template <int Dim>
bool AmrSystem<Dim>::has_active_step_transaction() const noexcept {
  return static_cast<bool>(p_->external_step_transaction);
}

template <int Dim>
void AmrSystem<Dim>::restore_active_step_transaction_for_program() {
  if (!p_->external_step_transaction)
    throw std::runtime_error("AmrSystem has no outer Program rollback image");
  p_->external_step_transaction->restore(*p_);
}

template <int Dim>
std::map<std::string, double> AmrSystem<Dim>::step_change_l2() const {
  if (!p_->external_step_transaction || !p_->engine ||
      !p_->external_step_transaction->engine)
    throw std::runtime_error("AmrSystem::step_change_l2 requires an active transaction");
  if (p_->engine->hierarchy().num_levels() != 1)
    throw std::runtime_error(
        "AmrSystem::step_change_l2 requires a prepared composite coverage provider");
  const MultiFab<Dim>& current = p_->engine->hierarchy().state(0);
  const MultiFab<Dim>& previous = p_->external_step_transaction->engine->hierarchy.state(0);
  const double value = std::sqrt(
      cell_measure(p_->cfg, p_->cfg.index_domain()) *
      static_cast<double>(difference_sum_sq_all(current, previous)));
  return {{p_->blocks.front().name, value}};
}

template <int Dim>
void AmrSystem<Dim>::install_program_step(std::function<void(double)> step) {
  p_->program.install_unverified_step(std::move(step));
}

template <int Dim>
void AmrSystem<Dim>::install_program_hierarchy_refresh(std::function<void()> refresh) {
  p_->program.install_hierarchy_refresh(std::move(refresh), "AmrSystem");
}

template <int Dim>
void AmrSystem<Dim>::install_program_restart_hooks(std::function<void()> preflight,
                                                   std::function<void()> regrid,
                                                   std::function<void()> resync) {
  p_->program.install_restart_hooks(std::move(preflight), std::move(regrid), std::move(resync),
                                    "AmrSystem");
}

template <int Dim>
void AmrSystem<Dim>::set_program_cadence(int substeps, int stride) {
  require_amr_assembling(p_->lifecycle, "set_program_cadence");
  p_->program.set_cadence(substeps, stride, "AmrSystem");
}

template <int Dim>
int AmrSystem<Dim>::program_substeps() const {
  return p_->program.substeps_;
}

template <int Dim>
int AmrSystem<Dim>::program_stride() const {
  return p_->program.stride_;
}

template <int Dim>
double AmrSystem<Dim>::program_cadence_window_dt() const {
  return p_->program.cadence_window_dt_;
}

template <int Dim>
int AmrSystem<Dim>::program_cadence_window_steps() const {
  return p_->program.cadence_window_steps_;
}

template <int Dim>
double AmrSystem<Dim>::program_cadence_window_start_time() const {
  return p_->program.cadence_window_start_time_;
}

template <int Dim>
double AmrSystem<Dim>::program_last_dt() const {
  return static_cast<double>(p_->program.last_dt_);
}

template <int Dim>
void AmrSystem<Dim>::restore_program_cadence_window(double accumulated_dt, int held_steps,
                                                    double window_start_time,
                                                    double accepted_last_dt,
                                                    double accepted_time, int macro_step) {
  p_->program.restore_cadence_window(accumulated_dt, held_steps, window_start_time,
                                     accepted_last_dt, accepted_time, macro_step, "AmrSystem");
}

template <int Dim>
void AmrSystem<Dim>::set_program_block_map(const std::vector<int>& program_to_runtime) {
  for (std::size_t program = 0; program < program_to_runtime.size(); ++program) {
    const int block = program_to_runtime[program];
    if (block < 0 || block >= static_cast<int>(p_->blocks.size()))
      throw std::out_of_range("AmrSystem Program block map is out of range");
    for (std::size_t previous = 0; previous < program; ++previous)
      if (program_to_runtime[previous] == block)
        throw std::invalid_argument("AmrSystem Program block map contains duplicate routes");
  }
  p_->program.block_map_ = program_to_runtime;
}

template <int Dim>
const std::vector<int>& AmrSystem<Dim>::program_block_map() const {
  return p_->program.block_map_;
}

template <int Dim>
std::string AmrSystem<Dim>::installed_program_hash() const {
  return p_->program.installed_hash_;
}

template <int Dim>
void AmrSystem<Dim>::seed_program_params(int block, const std::vector<double>& defaults) {
  p_->program.seed_params(block, defaults);
}

template <int Dim>
void AmrSystem<Dim>::set_program_params(int block, const std::vector<double>& values) {
  p_->program.set_params(block, values, "AmrSystem");
}

template <int Dim>
RuntimeParams AmrSystem<Dim>::program_params(int block) const {
  return p_->program.params(block);
}

template <int Dim>
runtime::amr::AmrRuntime<Dim>* AmrSystem<Dim>::engine() const {
  p_->ensure_engine();
  return p_->engine.get();
}

template <int Dim>
bool AmrSystem<Dim>::uses_runtime_engine() const {
  return static_cast<bool>(p_->engine);
}

template <int Dim>
runtime::program::Profiler& AmrSystem<Dim>::profiler_handle() {
  return p_->program.profiler_;
}

template <int Dim>
runtime::program::ProgramRuntimeState<Dim>& AmrSystem<Dim>::program_runtime_state_() {
  return p_->program;
}

template <int Dim>
void AmrSystem<Dim>::record_program_diagnostic(const std::string& name, double value) {
  p_->program.record_diagnostic(name, static_cast<Real>(value));
}

template <int Dim>
void AmrSystem<Dim>::record_program_balance_term(const std::string& route,
                                                 const std::string& term, double value) {
  p_->program.record_balance_term(route, term, static_cast<Real>(value), "AmrSystem");
}

template <int Dim>
bool AmrSystem<Dim>::program_balance_consumer_is_due(const std::string& contract,
                                                     const std::string& route,
                                                     int every_n) const {
  return p_->program.balance_consumer_is_due(contract, route, every_n, "AmrSystem");
}

template <int Dim>
double AmrSystem<Dim>::program_diagnostic(const std::string& name) const {
  return static_cast<double>(p_->program.diagnostic(name, "AmrSystem"));
}

template <int Dim>
std::map<std::string, double> AmrSystem<Dim>::program_diagnostics() const {
  std::map<std::string, double> result;
  for (const auto& [name, value] : p_->program.diagnostics())
    result.emplace(name, static_cast<double>(value));
  return result;
}

template <int Dim>
void AmrSystem<Dim>::begin_step_projection_report() {
  p_->program.begin_step_projection_report();
}

template <int Dim>
void AmrSystem<Dim>::note_step_projection(const std::string& name) {
  p_->program.note_step_projection(name);
}

template <int Dim>
std::vector<std::string> AmrSystem<Dim>::consume_step_projections() {
  return p_->program.consume_step_projections();
}

template <int Dim>
void AmrSystem<Dim>::mark_bound() {
  if (p_->lifecycle.frozen())
    p_->lifecycle.to_bound();
  p_->ensure_engine();
  const auto& routes = p_->boundary_registry.state_routes();
  if (!routes.empty() && routes.size() != p_->blocks.size())
    throw std::runtime_error("AmrSystem state routes do not exactly cover its blocks");
  for (const typename Impl::BlockSpec& block : p_->blocks) {
    if (!routes.empty() && !routes.contains(block.name))
      throw std::runtime_error("AmrSystem block lacks its exact state route");
    const auto* boundary = p_->boundary_registry.find_boundary(block.name);
    if (boundary == nullptr)
      continue;
    if (boundary->authority->ncomp() != block.ncomp ||
        boundary->authority->periodic_axes() != p_->cfg.periodicity)
      throw std::runtime_error("AmrSystem boundary differs from its block/domain contract");
    for (std::size_t level = 0; level < p_->engine->hierarchy().num_levels(); ++level)
      for (int axis = 0; axis < Dim; ++axis)
        if (p_->engine->hierarchy().state(level).ghosts()[axis] < boundary->required_depth)
          throw std::runtime_error("AmrSystem boundary depth exceeds level storage");
  }
  p_->lifecycle.to_bound();
}

template <int Dim>
std::string AmrSystem<Dim>::lifecycle_state() const {
  return p_->lifecycle.state(p_->macro_step);
}

template <int Dim>
Extent<Dim> AmrSystem<Dim>::spatial_shape() const {
  return p_->cfg.shape;
}

template <int Dim>
double AmrSystem<Dim>::time() const {
  return p_->accepted_time;
}

template <int Dim>
int AmrSystem<Dim>::macro_step() const {
  return p_->macro_step;
}

template <int Dim>
void AmrSystem<Dim>::set_clock(double accepted_time, int macro_step) {
  if (!std::isfinite(accepted_time) || macro_step < 0)
    throw std::invalid_argument("AmrSystem clock requires finite time and non-negative step");
  p_->program.consume_cadence_clock_restore(accepted_time, macro_step, "AmrSystem");
  p_->accepted_time = accepted_time;
  p_->macro_step = macro_step;
}

template <int Dim>
void AmrSystem<Dim>::enable_profiling() {
  p_->program.profiler_.enable();
}

template <int Dim>
void AmrSystem<Dim>::disable_profiling() {
  p_->program.profiler_.disable();
}

template <int Dim>
bool AmrSystem<Dim>::is_profiling() const {
  return p_->program.profiler_.enabled();
}

template <int Dim>
void AmrSystem<Dim>::reset_profiling() {
  p_->program.profiler_.reset();
}

template <int Dim>
std::string AmrSystem<Dim>::profile_report() const {
  return p_->program.profiler_.report();
}

template <int Dim>
int AmrSystem<Dim>::n_blocks() const {
  return static_cast<int>(p_->blocks.size());
}

template <int Dim>
std::vector<std::string> AmrSystem<Dim>::block_names() const {
  std::vector<std::string> names;
  names.reserve(p_->blocks.size());
  for (const typename Impl::BlockSpec& block : p_->blocks)
    names.push_back(block.name);
  return names;
}

template <int Dim>
EffectiveOptionsReport AmrSystem<Dim>::effective_options_report() const {
  EffectiveOptionsReport report;
  report.runtime = "amr_system";
  report.has_amr = true;
  report.topology.dimension = Dim;
  report.topology.periodicity.reserve(Dim);
  for (int axis = 0; axis < Dim; ++axis)
    report.topology.periodicity.push_back(p_->cfg.periodicity[axis]);
  report.blocks.reserve(p_->blocks.size());
  for (const typename Impl::BlockSpec& block : p_->blocks) {
    EffectiveBlockOptions row;
    row.name = block.name;
    row.ncomp = block.ncomp;
    row.n_ghost = block.required_ghost_depth;
    row.gamma = block.gamma;
    row.substeps = block.substeps;
    row.stride = block.stride;
    row.time = block.time;
    report.blocks.push_back(std::move(row));
  }
  return report;
}

template <int Dim>
int AmrSystem<Dim>::n_levels() {
  p_->ensure_engine();
  return static_cast<int>(p_->engine->hierarchy().num_levels());
}

template <int Dim>
int AmrSystem<Dim>::max_levels() {
  return p_->cfg.level_count;
}

template <int Dim>
int AmrSystem<Dim>::configured_n_levels() {
  return p_->cfg.level_count;
}

template <int Dim>
int AmrSystem<Dim>::n_vars() {
  if (p_->blocks.size() != 1)
    throw std::runtime_error("AmrSystem::n_vars requires exactly one block");
  return p_->blocks.front().ncomp;
}

template <int Dim>
int AmrSystem<Dim>::block_n_vars(const std::string& name) {
  return p_->block(name).ncomp;
}

template <int Dim>
int AmrSystem<Dim>::n_patches() {
  p_->ensure_engine();
  const auto& hierarchy = p_->engine->hierarchy();
  return static_cast<int>(hierarchy.layout(hierarchy.num_levels() - 1).patches().size());
}

template <int Dim>
std::vector<AmrPatch<Dim>> AmrSystem<Dim>::patch_boxes() {
  p_->ensure_engine();
  std::vector<AmrPatch<Dim>> result;
  for (std::size_t level = 1; level < p_->engine->hierarchy().num_levels(); ++level)
    for (const Box<Dim>& box : p_->engine->hierarchy().layout(level).patches().boxes())
      result.push_back({static_cast<int>(level), box});
  return result;
}

template <int Dim>
std::vector<AmrPatch<Dim>> AmrSystem<Dim>::output_geometry_boxes() {
  p_->ensure_engine();
  std::vector<AmrPatch<Dim>> result;
  for (std::size_t level = 0; level < p_->engine->hierarchy().num_levels(); ++level)
    for (const Box<Dim>& box : p_->engine->hierarchy().layout(level).patches().boxes())
      result.push_back({static_cast<int>(level), box});
  return result;
}

template <int Dim>
int AmrSystem<Dim>::coarse_local_boxes() {
  p_->ensure_engine();
  return static_cast<int>(p_->engine->hierarchy().state(0).local_size());
}

template <int Dim>
int AmrSystem<Dim>::coarse_total_boxes() {
  p_->ensure_engine();
  return static_cast<int>(p_->engine->hierarchy().layout(0).patches().size());
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::level_state(int level) {
  return level_state_global(level);
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::level_state_global(int level) {
  p_->ensure_engine();
  if (level < 0 || static_cast<std::size_t>(level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("AmrSystem level is out of range");
  const auto& hierarchy = p_->engine->hierarchy();
  return gather_field(hierarchy.state(static_cast<std::size_t>(level)),
                      hierarchy.layout(static_cast<std::size_t>(level)).domain(),
                      hierarchy.state(static_cast<std::size_t>(level)).ncomp());
}

template <int Dim>
void AmrSystem<Dim>::set_level_state(int level, const std::vector<double>& state) {
  p_->ensure_engine();
  if (level < 0 || static_cast<std::size_t>(level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("AmrSystem level is out of range");
  auto& hierarchy = p_->engine->hierarchy();
  write_field(hierarchy.state(static_cast<std::size_t>(level)),
              hierarchy.layout(static_cast<std::size_t>(level)).domain(), state,
              hierarchy.state(static_cast<std::size_t>(level)).ncomp());
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::block_level_state(const std::string& name, int level) {
  (void)p_->block(name);
  return level_state(level);
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::block_level_state_global(const std::string& name, int level) {
  (void)p_->block(name);
  return level_state_global(level);
}

template <int Dim>
void AmrSystem<Dim>::set_block_level_state(const std::string& name, int level,
                                           const std::vector<double>& state) {
  (void)p_->block(name);
  set_level_state(level, state);
}

template <int Dim>
std::vector<OutputPiece<Dim>> AmrSystem<Dim>::output_state_local_pieces(
    const std::string& name, int level) {
  (void)p_->block(name);
  p_->ensure_engine();
  if (level < 0 || static_cast<std::size_t>(level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("AmrSystem output level is out of range");
  const MultiFab<Dim>& state = p_->engine->hierarchy().state(static_cast<std::size_t>(level));
  return output_local_pieces(state, level, state.distribution().replicated());
}

template <int Dim>
std::vector<OutputPiece<Dim>> AmrSystem<Dim>::output_state_root_pieces(
    const ObserverMpiLane& lane, const std::string& name, int level) {
  return output_pieces_to_root(
      lane, detail::output_collective_identity("amr_system", "state", name, level),
      [&] { return output_state_local_pieces(name, level); });
}

template <int Dim>
std::vector<int> AmrSystem<Dim>::level_owner_ranks(int level) {
  p_->ensure_engine();
  if (level < 0 || static_cast<std::size_t>(level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("AmrSystem owner level is out of range");
  const auto& distribution =
      p_->engine->hierarchy().layout(static_cast<std::size_t>(level)).distribution();
  if (distribution.replicated())
    throw std::runtime_error("replicated AMR levels have no unique owner ranks");
  std::vector<int> result;
  result.reserve(distribution.owners().size());
  for (const Index<Dim>& owner : distribution.owners())
    result.push_back(static_cast<int>(distribution.rank_space().linear_rank(owner)));
  return result;
}

template <int Dim>
double AmrSystem<Dim>::mass() {
  if (p_->blocks.empty())
    throw std::logic_error("AmrSystem has no block");
  return mass(p_->blocks.front().name);
}

template <int Dim>
double AmrSystem<Dim>::mass(const std::string& name) {
  (void)p_->block(name);
  p_->ensure_engine();
  if (p_->engine->hierarchy().num_levels() != 1)
    throw std::runtime_error("AmrSystem mass requires a prepared composite coverage provider");
  return static_cast<double>(reduce_sum(p_->engine->hierarchy().state(0), 0)) *
         cell_measure(p_->cfg, p_->cfg.index_domain());
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::density() {
  if (p_->blocks.empty())
    throw std::logic_error("AmrSystem has no block");
  return density(p_->blocks.front().name);
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::density(const std::string& name) {
  (void)p_->block(name);
  p_->ensure_engine();
  return gather_field(p_->engine->hierarchy().state(0), p_->cfg.index_domain(), 1);
}

template <int Dim>
std::vector<std::uint8_t> AmrSystem<Dim>::program_accepted_state() const {
  return p_->program_accepted_bytes;
}

template <int Dim>
void AmrSystem<Dim>::copy_program_accepted_state_into(std::vector<std::uint8_t>& state) const {
  state = p_->program_accepted_bytes;
}

template <int Dim>
void AmrSystem<Dim>::restore_program_accepted_state(const std::vector<std::uint8_t>& state) {
  if (p_->program_accepted_revision == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error("AmrSystem Program accepted-state revision overflow");
  p_->program_accepted_bytes = state;
  ++p_->program_accepted_revision;
}

template <int Dim>
void AmrSystem<Dim>::restore_checkpoint_accepted_state(const std::vector<std::uint8_t>& state) {
  restore_program_accepted_state(state);
}

template <int Dim>
std::uint64_t AmrSystem<Dim>::program_accepted_state_revision() const {
  return p_->program_accepted_revision;
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::program_accepted_state_manifest() const {
  return {};
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::program_clock_manifest() const {
  return {};
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::program_temporal_partition_manifest() const {
  return {};
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::program_flux_ledger_manifest() const {
  return {};
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::program_interface_flux_ledger_manifest() const {
  return {};
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::program_sync_manifest() const {
  return {};
}

template <int Dim>
const std::vector<CouplingOperatorView>& AmrSystem<Dim>::coupled_operators() const {
  return p_->coupling_views;
}

template AmrSystem<kNativeDimension>::AmrSystem(
    const AmrSystemConfig<kNativeDimension>&);
template AmrSystem<kNativeDimension>::~AmrSystem();
template AmrSystem<kNativeDimension>::AmrSystem(AmrSystem&&) noexcept;
template AmrSystem<kNativeDimension>& AmrSystem<kNativeDimension>::operator=(AmrSystem&&) noexcept;
template void AmrSystem<kNativeDimension>::add_block(
    const std::string&, const ModelSpec&, const std::string&, const std::string&,
    const std::string&, const std::string&, int, int, const std::vector<std::string>&,
    const std::vector<std::string>&, const NewtonOptions&, bool, double, double, bool);
template void AmrSystem<kNativeDimension>::set_compiled_block(
    int, double, int, AmrCompiledBlockBuilder<kNativeDimension>, const std::string&, bool,
    const std::string&, int, const std::vector<std::string>&, const std::vector<std::string>&,
    double, double, bool);
template void AmrSystem<kNativeDimension>::install_block_state_route(const std::string&,
                                                                      const std::string&);
template void AmrSystem<kNativeDimension>::install_field_storage_route(const std::string&,
                                                                        const std::string&);
template void AmrSystem<kNativeDimension>::install_hyperbolic_boundary(
    const std::string&, const std::string&, int, const std::vector<std::string>&,
    const std::vector<double>&, const std::vector<std::string>&,
    const std::vector<std::string>&, const std::string&, const std::vector<std::string>&,
    const std::vector<std::string>&, const std::vector<std::vector<std::string>>&,
    const std::vector<std::vector<double>>&, const std::vector<std::string>&);
template void AmrSystem<kNativeDimension>::install_prepared_hyperbolic_boundary(
    const std::string&, const std::string&, int, const std::string&,
    std::shared_ptr<const HyperbolicBoundary>);
template void AmrSystem<kNativeDimension>::discard_hyperbolic_boundaries();
template void AmrSystem<kNativeDimension>::set_density(const std::string&,
                                                       const std::vector<double>&);
template void AmrSystem<kNativeDimension>::set_conservative_state(
    const std::string&, const std::vector<double>&);
template void AmrSystem<kNativeDimension>::add_dt_bound(const std::string&,
                                                        std::function<double()>);
template std::string AmrSystem<kNativeDimension>::last_dt_bound() const;
template void AmrSystem<kNativeDimension>::step(double);
template void AmrSystem<kNativeDimension>::advance(double, int);
template double AmrSystem<kNativeDimension>::step_cfl(double, double, double, double);
template void AmrSystem<kNativeDimension>::begin_step_transaction();
template void AmrSystem<kNativeDimension>::commit_step_transaction();
template void AmrSystem<kNativeDimension>::finalize_step_transaction();
template void AmrSystem<kNativeDimension>::rollback_step_transaction();
template bool AmrSystem<kNativeDimension>::has_active_step_transaction() const noexcept;
template void AmrSystem<kNativeDimension>::restore_active_step_transaction_for_program();
template std::map<std::string, double> AmrSystem<kNativeDimension>::step_change_l2() const;
template void AmrSystem<kNativeDimension>::install_program_step(std::function<void(double)>);
template void AmrSystem<kNativeDimension>::install_program_hierarchy_refresh(
    std::function<void()>);
template void AmrSystem<kNativeDimension>::install_program_restart_hooks(
    std::function<void()>, std::function<void()>, std::function<void()>);
template void AmrSystem<kNativeDimension>::set_program_cadence(int, int);
template int AmrSystem<kNativeDimension>::program_substeps() const;
template int AmrSystem<kNativeDimension>::program_stride() const;
template double AmrSystem<kNativeDimension>::program_cadence_window_dt() const;
template int AmrSystem<kNativeDimension>::program_cadence_window_steps() const;
template double AmrSystem<kNativeDimension>::program_cadence_window_start_time() const;
template double AmrSystem<kNativeDimension>::program_last_dt() const;
template void AmrSystem<kNativeDimension>::restore_program_cadence_window(double, int, double,
                                                                          double, double, int);
template void AmrSystem<kNativeDimension>::set_program_block_map(const std::vector<int>&);
template const std::vector<int>& AmrSystem<kNativeDimension>::program_block_map() const;
template std::string AmrSystem<kNativeDimension>::installed_program_hash() const;
template void AmrSystem<kNativeDimension>::seed_program_params(int, const std::vector<double>&);
template void AmrSystem<kNativeDimension>::set_program_params(int, const std::vector<double>&);
template RuntimeParams AmrSystem<kNativeDimension>::program_params(int) const;
template runtime::amr::AmrRuntime<kNativeDimension>* AmrSystem<kNativeDimension>::engine() const;
template bool AmrSystem<kNativeDimension>::uses_runtime_engine() const;
template runtime::program::Profiler& AmrSystem<kNativeDimension>::profiler_handle();
template runtime::program::ProgramRuntimeState<kNativeDimension>&
AmrSystem<kNativeDimension>::program_runtime_state_();
template void AmrSystem<kNativeDimension>::record_program_diagnostic(const std::string&, double);
template void AmrSystem<kNativeDimension>::record_program_balance_term(const std::string&,
                                                                        const std::string&, double);
template bool AmrSystem<kNativeDimension>::program_balance_consumer_is_due(
    const std::string&, const std::string&, int) const;
template double AmrSystem<kNativeDimension>::program_diagnostic(const std::string&) const;
template std::map<std::string, double> AmrSystem<kNativeDimension>::program_diagnostics() const;
template void AmrSystem<kNativeDimension>::begin_step_projection_report();
template void AmrSystem<kNativeDimension>::note_step_projection(const std::string&);
template std::vector<std::string> AmrSystem<kNativeDimension>::consume_step_projections();
template void AmrSystem<kNativeDimension>::mark_bound();
template std::string AmrSystem<kNativeDimension>::lifecycle_state() const;
template Extent<kNativeDimension> AmrSystem<kNativeDimension>::spatial_shape() const;
template double AmrSystem<kNativeDimension>::time() const;
template int AmrSystem<kNativeDimension>::macro_step() const;
template void AmrSystem<kNativeDimension>::set_clock(double, int);
template void AmrSystem<kNativeDimension>::enable_profiling();
template void AmrSystem<kNativeDimension>::disable_profiling();
template bool AmrSystem<kNativeDimension>::is_profiling() const;
template void AmrSystem<kNativeDimension>::reset_profiling();
template std::string AmrSystem<kNativeDimension>::profile_report() const;
template int AmrSystem<kNativeDimension>::n_blocks() const;
template std::vector<std::string> AmrSystem<kNativeDimension>::block_names() const;
template EffectiveOptionsReport AmrSystem<kNativeDimension>::effective_options_report() const;
template int AmrSystem<kNativeDimension>::n_levels();
template int AmrSystem<kNativeDimension>::max_levels();
template int AmrSystem<kNativeDimension>::configured_n_levels();
template int AmrSystem<kNativeDimension>::n_vars();
template int AmrSystem<kNativeDimension>::block_n_vars(const std::string&);
template int AmrSystem<kNativeDimension>::n_patches();
template std::vector<AmrPatch<kNativeDimension>> AmrSystem<kNativeDimension>::patch_boxes();
template std::vector<AmrPatch<kNativeDimension>>
AmrSystem<kNativeDimension>::output_geometry_boxes();
template int AmrSystem<kNativeDimension>::coarse_local_boxes();
template int AmrSystem<kNativeDimension>::coarse_total_boxes();
template std::vector<double> AmrSystem<kNativeDimension>::level_state(int);
template std::vector<double> AmrSystem<kNativeDimension>::level_state_global(int);
template void AmrSystem<kNativeDimension>::set_level_state(int, const std::vector<double>&);
template std::vector<double> AmrSystem<kNativeDimension>::block_level_state(const std::string&, int);
template std::vector<double> AmrSystem<kNativeDimension>::block_level_state_global(
    const std::string&, int);
template void AmrSystem<kNativeDimension>::set_block_level_state(const std::string&, int,
                                                                  const std::vector<double>&);
template std::vector<OutputPiece<kNativeDimension>>
AmrSystem<kNativeDimension>::output_state_local_pieces(const std::string&, int);
template std::vector<OutputPiece<kNativeDimension>>
AmrSystem<kNativeDimension>::output_state_root_pieces(const ObserverMpiLane&, const std::string&,
                                                       int);
template std::vector<int> AmrSystem<kNativeDimension>::level_owner_ranks(int);
template double AmrSystem<kNativeDimension>::mass();
template double AmrSystem<kNativeDimension>::mass(const std::string&);
template std::vector<double> AmrSystem<kNativeDimension>::density();
template std::vector<double> AmrSystem<kNativeDimension>::density(const std::string&);
template std::vector<std::uint8_t> AmrSystem<kNativeDimension>::program_accepted_state() const;
template void AmrSystem<kNativeDimension>::copy_program_accepted_state_into(
    std::vector<std::uint8_t>&) const;
template void AmrSystem<kNativeDimension>::restore_program_accepted_state(
    const std::vector<std::uint8_t>&);
template void AmrSystem<kNativeDimension>::restore_checkpoint_accepted_state(
    const std::vector<std::uint8_t>&);
template std::uint64_t AmrSystem<kNativeDimension>::program_accepted_state_revision() const;
template std::vector<std::vector<std::string>>
AmrSystem<kNativeDimension>::program_accepted_state_manifest() const;
template std::vector<std::vector<std::string>>
AmrSystem<kNativeDimension>::program_clock_manifest() const;
template std::vector<std::vector<std::string>>
AmrSystem<kNativeDimension>::program_temporal_partition_manifest() const;
template std::vector<std::vector<std::string>>
AmrSystem<kNativeDimension>::program_flux_ledger_manifest() const;
template std::vector<std::vector<std::string>>
AmrSystem<kNativeDimension>::program_interface_flux_ledger_manifest() const;
template std::vector<std::vector<std::string>>
AmrSystem<kNativeDimension>::program_sync_manifest() const;
template const std::vector<CouplingOperatorView>&
AmrSystem<kNativeDimension>::coupled_operators() const;

}  // namespace pops
