#include <pops/core/foundation/native_dimension.hpp>
#include <pops/runtime/amr/component_field_solver_provider.hpp>

namespace pops::runtime::amr {
namespace {

template <int Dim>
class ComponentExactAmrFieldSolver final : public ExactAmrFieldSolver<Dim> {
 public:
  using field_type = MultiFab<Dim>;
  using request_type = ExactAmrFieldSolverBuildRequest<Dim>;
  using component_type = field::PreparedFieldSolverComponent<Dim>;

  ComponentExactAmrFieldSolver(const request_type& request, std::string contract,
                               field::PreparedFieldSolverSpec spec,
                               std::shared_ptr<component::LoadedComponent> topology,
                               std::shared_ptr<component::LoadedComponent> solver,
                               const ExecutionLane& lane)
      : lane_(&lane),
        lane_borrow_(lane.borrow_immutably()),
        contract_(std::move(contract)),
        spec_(std::move(spec)),
        component_(spec_, std::move(topology), std::move(solver), true) {
    PopsFieldGlobalTopologyV1 global{};
    std::vector<component::FieldTopologyLevelGeometryV2> level_geometry;
    std::exception_ptr preparation_error;
    try {
      const auto& levels = request.hierarchy.levels;
      level_geometry.resize(levels.size());
      rhs_.reserve(levels.size());
      candidates_.reserve(levels.size());
      std::size_t total = 0;
      for (const auto& level : levels) {
        rhs_.emplace_back(level.boxes, level.distribution, level.local_rank, 1, level.rhs_ghosts);
        candidates_.emplace_back(level.boxes, level.distribution, level.local_rank, 1,
                                 level.phi_ghosts);
        if (level.boxes.size() > std::numeric_limits<std::size_t>::max() - total)
          throw std::length_error("external field hierarchy patch count exceeds size_t");
        total += level.boxes.size();
      }
      // The prepared contract includes the exact hierarchy, geometry, ownership and lane.
      const std::vector<std::uint8_t> bytes(contract_.begin(), contract_.end());
      layout_identity_ = "pops.runtime-field-hierarchy.v1:sha256:" + identity::sha256_hex(bytes);
      patch_identities_.reserve(total);
      for (std::size_t index = 0; index < total; ++index)
        patch_identities_.push_back(layout_identity_ + ":patch:" + std::to_string(index));
      global_.reserve(total);
      std::size_t first = 0;
      for (std::size_t l = 0; l < levels.size(); ++l) {
        level_offsets_.push_back(first);
        const auto& level = levels[l];
        const auto& geometry = level.geometry;
        auto& exact_geometry = level_geometry[l];
        for (int axis = 0; axis < Dim; ++axis) {
          exact_geometry.lower[axis] = geometry.domain().lo[axis];
          exact_geometry.upper[axis] = geometry.domain().hi[axis];
          exact_geometry.physical_lower[axis] = geometry.lower()[axis];
          exact_geometry.cell_spacing[axis] = geometry.spacing(axis);
        }
        for (std::size_t index = 0; index < level.boxes.size(); ++index) {
          const auto& box = level.boxes[index];
          PopsFieldPatchMetadataV1 row{};
          row.struct_size = sizeof(row);
          row.global_patch_index = first + index;
          row.level = static_cast<std::int32_t>(l);
          row.owner_rank = level.distribution.replicated()
                               ? 0
                               : static_cast<std::int32_t>(rhs_[l].rank_space().linear_rank(
                                     level.distribution.owner(index)));
          row.dimension = Dim;
          row.centering = POPS_FIELD_CENTERING_CELL_V1;
          row.layout_identity = spec_.source_layout_identity.c_str();
          row.patch_identity = patch_identities_[first + index].c_str();
          for (int axis = 0; axis < Dim; ++axis) {
            row.lower[axis] = box.lo[axis];
            row.upper[axis] = box.hi[axis];
            row.cell_spacing[axis] = geometry.spacing(axis);
            row.physical_lower[axis] = geometry.lower()[axis] + (static_cast<double>(box.lo[axis]) -
                                                                 geometry.domain().lo[axis]) *
                                                                    geometry.spacing(axis);
          }
          global_.push_back(row);
        }
        for (std::size_t local = 0; local < rhs_[l].local_size(); ++local) {
          const auto& box = rhs_[l].box(local);
          coverage_.emplace_back(static_cast<std::size_t>(box.numPts()), std::uint8_t{1});
          auto& mask = coverage_.back();
          if (l + 1 < levels.size()) {
            const auto& child = levels[l + 1];
            const auto& ratio = request.hierarchy.ratios[l];
            for (std::size_t ordinal = 0; ordinal < mask.size(); ++ordinal) {
              std::size_t remaining = ordinal;
              Index<Dim> index{};
              for (int axis = 0; axis < Dim; ++axis) {
                const auto extent = static_cast<std::size_t>(box.hi[axis] - box.lo[axis] + 1);
                index[axis] = box.lo[axis] + static_cast<int>(remaining % extent);
                remaining /= extent;
              }
              // Fine boxes are parent-cell aligned by the exact hierarchy contract.
              for (const auto& fine : child.boxes.boxes()) {
                bool covered = true;
                for (int axis = 0; axis < Dim; ++axis) {
                  const auto child_index =
                      static_cast<std::int64_t>(child.geometry.domain().lo[axis]) +
                      (static_cast<std::int64_t>(index[axis]) - geometry.domain().lo[axis]) *
                          ratio[axis];
                  covered = covered && child_index >= fine.lo[axis] &&
                            child_index + ratio[axis] - 1 <= fine.hi[axis];
                }
                if (covered) {
                  mask[ordinal] = 0;
                  break;
                }
              }
            }
          }
          const std::size_t metadata = first + rhs_[l].global_index(local);
          component::FieldTopologyPatchInputV2 input{};
          input.metadata_index = metadata;
          input.material_representation = POPS_FIELD_MATERIAL_BINARY_COVERAGE_V1;
          input.material_coverage.struct_size = sizeof(PopsConstByteViewV1);
          input.material_coverage.data = mask.data();
          input.material_coverage.size = mask.size();
          local_.push_back(input);
          bindings_.push_back(component_type::hierarchy_patch_binding(
              metadata, rhs_[l].fab(local), candidates_[l].fab(local), box,
              spec_.source_layout_identity.c_str(), patch_identities_[metadata].c_str()));
        }
        first += level.boxes.size();
      }
      global.struct_size = sizeof(global);
      global.topology_recipe_identity = spec_.topology_recipe_identity.c_str();
      global.source_layout_identity = spec_.source_layout_identity.c_str();
      global.materialized_layout_identity = layout_identity_.c_str();
      global.dimension = Dim;
      global.patch_count = global_.size();
      global.patches = global_.data();
      for (int axis = 0; axis < Dim; ++axis) {
        global.domain_lower[axis] = levels.front().geometry.domain().lo[axis];
        global.domain_upper[axis] = levels.front().geometry.domain().hi[axis];
        if (levels.front().boundary.topology().is_periodic(Face<Dim>{axis, BoundarySide::lower}))
          global.periodic_axes |= std::uint32_t{1} << axis;
      }
    } catch (...) {
      preparation_error = std::current_exception();
    }
    if (all_reduce_max(preparation_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && preparation_error)
        std::rethrow_exception(preparation_error);
      throw std::runtime_error("external AMR field hierarchy preparation failed collectively");
    }
    component_.bind_hierarchy(global, local_, bindings_, level_geometry);
  }

  std::string_view provider_identity() const noexcept override { return spec_.provider_slot; }
  std::string_view exact_prepared_contract() const noexcept override { return contract_; }
  bool couples_hierarchy_levels() const noexcept override { return true; }
  int level_count() const noexcept override { return static_cast<int>(rhs_.size()); }
  field_type& rhs_level(int level) override { return rhs_.at(level); }
  field_type& candidate_level(int level) override { return candidates_.at(level); }
  const field_type& candidate_level(int level) const override { return candidates_.at(level); }
  void install_newton(FieldNewtonOptions) override {
    throw std::invalid_argument("external AMR FieldSolver does not accept builtin Newton options");
  }
  void install_boundary_kernel(CompiledFieldBoundaryKernel<Dim>) override {
    throw std::invalid_argument(
        "external AMR FieldSolver requires its declared physical boundary contract");
  }
  void set_boundary_contexts(std::shared_ptr<const PreparedFieldBoundaryContextSet<Dim>>) override {
    throw std::invalid_argument("external AMR FieldSolver has no dynamic boundary-context ABI");
  }
  void install_nullspace(PreparedFieldNullspace<Dim> prepared,
                         std::vector<PreparedVectorDistribution<Dim>> distributions) override {
    if (!nullspace_contract_.empty() || prepared.provider_identity.empty() ||
        prepared.provider_version == 0 || prepared.exact_prepared_contract.empty() ||
        distributions.size() != rhs_.size())
      throw std::invalid_argument("external AMR FieldSolver requires exact nullspace authority");
    if (!prepared.plan.empty())
      throw std::invalid_argument(
          "external AMR FieldSolver cannot encode nonempty runtime nullspace plans");
    nullspace_contract_ = std::move(prepared.exact_prepared_contract);
  }
  int maximum_iterations() const noexcept override { return component_.maximum_iterations(); }
  std::vector<field::FieldTopologyReportRow> topology_report() const override {
    return component_.topology_report();
  }
  SolveReport solve(const ExecutionLane& lane) override {
    if (all_reduce_max(&lane == lane_ ? 0L : 1L, *lane_) != 0)
      throw std::invalid_argument("external AMR FieldSolver requires its prepared execution lane");
    if (nullspace_contract_.empty())
      throw std::logic_error("external AMR FieldSolver has no installed nullspace authority");
    bool exact = true;
    std::size_t cursor = 0;
    for (std::size_t l = 0; l < rhs_.size(); ++l) {
      exact = exact && rhs_[l].ncomp() == 1 && candidates_[l].ncomp() == 1 &&
              rhs_[l].local_size() == candidates_[l].local_size() &&
              rhs_[l].layout() == candidates_[l].layout() &&
              rhs_[l].distribution() == candidates_[l].distribution() &&
              rhs_[l].local_rank() == candidates_[l].local_rank();
      if (!exact)
        break;
      for (std::size_t local = 0; local < rhs_[l].local_size(); ++local, ++cursor) {
        const auto metadata = level_offsets_[l] + rhs_[l].global_index(local);
        if (metadata >= patch_identities_.size() || cursor >= bindings_.size() ||
            rhs_[l].global_index(local) != candidates_[l].global_index(local)) {
          exact = false;
          break;
        }
        exact = exact && component_.hierarchy_binding_matches(
                             cursor, component_type::hierarchy_patch_binding(
                                         metadata, rhs_[l].fab(local), candidates_[l].fab(local),
                                         rhs_[l].box(local), spec_.source_layout_identity.c_str(),
                                         patch_identities_[metadata].c_str()));
      }
    }
    if (all_reduce_max(exact && cursor == bindings_.size() ? 0L : 1L, *lane_) != 0)
      throw std::runtime_error("external AMR FieldSolver storage changed after hierarchy binding");
    return component_.solve_hierarchy();
  }

 private:
  const ExecutionLane* lane_;
  ExecutionLane::ImmutableBorrow lane_borrow_;
  std::string contract_;
  field::PreparedFieldSolverSpec spec_;
  std::string nullspace_contract_;
  std::string layout_identity_;
  std::vector<std::string> patch_identities_;
  std::vector<std::size_t> level_offsets_;
  std::vector<field_type> rhs_;
  std::vector<field_type> candidates_;
  std::vector<PopsFieldPatchMetadataV1> global_;
  std::vector<std::vector<std::uint8_t>> coverage_;
  std::vector<component::FieldTopologyPatchInputV2> local_;
  std::vector<component::FieldSolverPatchBindingV2> bindings_;
  // Destroy provider sessions before their borrowed hierarchy storage.
  component_type component_;
};

template <int Dim>
class ComponentExactAmrFieldSolverProvider final : public ExactAmrFieldSolverProvider<Dim> {
 public:
  using request_type = ExactAmrFieldSolverBuildRequest<Dim>;
  ComponentExactAmrFieldSolverProvider(field::PreparedFieldSolverSpec spec,
                                       std::shared_ptr<component::LoadedComponent> topology,
                                       std::shared_ptr<component::LoadedComponent> solver)
      : spec_(std::move(spec)), topology_(std::move(topology)), solver_(std::move(solver)) {
    field::PreparedFieldSolverComponent<Dim> validated(spec_, topology_, solver_);
    collective_contract_ = validated.collective_contract();
  }
  std::string_view identity() const noexcept override { return spec_.provider_slot; }
  std::string_view collective_contract() const noexcept override { return collective_contract_; }
  PreparedProviderSupport supports(const request_type& request,
                                   const ExecutionLane& lane) const noexcept override {
    try {
      if (request.mode != ExactFieldHierarchyMode::composite || request.hierarchy.levels.empty() ||
          request.hierarchy.ratios.size() + 1 != request.hierarchy.levels.size())
        return PreparedProviderSupport::reject(
            1, "external field component requires a complete composite hierarchy");
      if (request.reaction != Real(0))
        return PreparedProviderSupport::reject(
            2, "external field component has no runtime reaction-coefficient binding");
      const auto& levels = request.hierarchy.levels;
      if (levels.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
          !levels.front().boxes.tiles_exactly(levels.front().geometry.domain(),
                                              levels.front().layout_budget))
        return PreparedProviderSupport::reject(
            3, "external field hierarchy requires complete coarse coverage");
      for (std::size_t l = 0; l < levels.size(); ++l) {
        const auto& level = levels[l];
        if ((lane.size() > 1 && level.distribution.replicated()) ||
            !level.boxes.is_disjoint_within(level.geometry.domain(), level.layout_budget))
          return PreparedProviderSupport::reject(
              4, "external field component requires unique valid patch ownership");
        if (l == 0)
          continue;
        const auto& parent = levels[l - 1];
        const auto& ratio = request.hierarchy.ratios[l - 1];
        for (int axis = 0; axis < Dim; ++axis) {
          if (level.geometry.lower()[axis] != parent.geometry.lower()[axis] ||
              level.geometry.upper()[axis] != parent.geometry.upper()[axis] ||
              static_cast<std::int64_t>(level.geometry.domain().length(axis)) !=
                  static_cast<std::int64_t>(parent.geometry.domain().length(axis)) * ratio[axis])
            return PreparedProviderSupport::reject(
                5, "external field hierarchy geometry does not match its refinement ratio");
          for (const auto& box : level.boxes.boxes()) {
            if ((static_cast<std::int64_t>(box.lo[axis]) - level.geometry.domain().lo[axis]) %
                        ratio[axis] !=
                    0 ||
                (static_cast<std::int64_t>(box.hi[axis]) + 1 - level.geometry.domain().lo[axis]) %
                        ratio[axis] !=
                    0)
              return PreparedProviderSupport::reject(
                  6, "external field hierarchy fine patches must align to complete parent cells");
          }
        }
      }
      return PreparedProviderSupport::accept();
    } catch (...) {
      return PreparedProviderSupport::reject(
          7, "external field component hierarchy validation failed");
    }
  }
  std::string expected_prepared_contract(const request_type& request,
                                         const ExecutionLane& lane) const override {
    ExactContractBuilder contract;
    contract.text(make_exact_amr_field_solver_contract(identity(), request, lane))
        .text(collective_contract_);
    return std::move(contract).release();
  }
  std::unique_ptr<ExactAmrFieldSolver<Dim>> build(const request_type& request,
                                                  const ExecutionLane& lane) const override {
    const auto support = supports(request, lane);
    if (!support.accepted())
      throw std::invalid_argument(std::string(support.reason));
    return std::make_unique<ComponentExactAmrFieldSolver<Dim>>(
        request, expected_prepared_contract(request, lane), spec_, topology_, solver_, lane);
  }

 private:
  field::PreparedFieldSolverSpec spec_;
  std::shared_ptr<component::LoadedComponent> topology_;
  std::shared_ptr<component::LoadedComponent> solver_;
  std::string collective_contract_;
};

}  // namespace

template <int Dim>
std::shared_ptr<const ExactAmrFieldSolverProvider<Dim>>
make_component_exact_amr_field_solver_provider(field::PreparedFieldSolverSpec spec,
                                               std::shared_ptr<component::LoadedComponent> topology,
                                               std::shared_ptr<component::LoadedComponent> solver) {
  return std::make_shared<ComponentExactAmrFieldSolverProvider<Dim>>(
      std::move(spec), std::move(topology), std::move(solver));
}

template POPS_EXPORT std::shared_ptr<const ExactAmrFieldSolverProvider<kNativeDimension>>
    make_component_exact_amr_field_solver_provider<kNativeDimension>(
        field::PreparedFieldSolverSpec, std::shared_ptr<component::LoadedComponent>,
        std::shared_ptr<component::LoadedComponent>);

}  // namespace pops::runtime::amr
