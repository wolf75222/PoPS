void require_named_flux_execution_envelope_(int runtime_block) const {
  if (facade_->program_prepared_amr_block_level_active_mask_(runtime_block, active_level_) !=
      nullptr)
    throw std::invalid_argument(
        "AMR named flux currently requires a Cartesian level without embedded boundaries");
  const auto& hierarchy = facade_->prepared_amr_multiblock_hierarchy_();
  // The prepared interface scheduler exposes hierarchy-wide rather than per-block participation.
  // Until that authority publishes an exact participating-block set, accepting one block
  // selectively would claim a topological face route that the named cell-flux carrier cannot
  // authenticate. Ordinary prepared source couplings remain supported because they run after the
  // independently conservative transport expression and do not own a topological face route.
  if (hierarchy.has_interface_flux_provider())
    throw std::invalid_argument(
        "AMR named flux currently refuses the complete prepared carrier pack when shared "
        "topological interfaces are installed");
}

const field_type* staged_parent_for_block_(int runtime_block) const {
  if (runtime_block < 0)
    throw std::out_of_range("AMR Program staged-parent block is out of range");
  if (active_staged_parents_.empty())
    return nullptr;
  if (static_cast<std::size_t>(runtime_block) >= active_staged_parents_.size())
    throw std::logic_error("AMR Program staged-parent registry is incomplete");
  return active_staged_parents_[static_cast<std::size_t>(runtime_block)];
}

std::optional<int> authenticated_runtime_block_for_state_target_(const field_type& target) const {
  require_facade_execution_();
  const auto& map = facade_->program_block_map_();
  if (map.size() != static_cast<std::size_t>(n_blocks()))
    throw std::logic_error("AMR Program state target has no complete authenticated block map");
  std::optional<int> match;
  for (int runtime_block = 0; runtime_block < n_blocks(); ++runtime_block) {
    const field_type* candidate = live_attempt_state_(runtime_block, active_level_);
    const field_type* accepted =
        &facade_->program_prepared_amr_block_state_(runtime_block, active_level_);
    if (&target != candidate && &target != accepted)
      continue;
    if (match)
      throw std::logic_error("AMR Program state target aliases two runtime blocks");
    match = runtime_block;
  }
  return match;
}

static void copy_full_(const field_type& source, field_type& destination) {
  require_same_field_contract_(source, destination, "AMR Program full-field copy");
  if (source.ghosts() != destination.ghosts() || source.shares_storage_with(destination))
    throw std::invalid_argument(
        "AMR Program full-field copy requires detached exact ghost storage");
  for (std::size_t local = 0; local < destination.local_size(); ++local) {
    if (source.global_index(local) != destination.global_index(local) ||
        source.fab(local).box() != destination.fab(local).box() ||
        source.fab(local).grown_box() != destination.fab(local).grown_box() ||
        source.fab(local).size() != destination.fab(local).size())
      throw std::invalid_argument("AMR Program full-field copy patch storage changed");
  }
  for (std::size_t local = 0; local < destination.local_size(); ++local) {
    const auto& source_storage = source.fab(local).storage();
    auto& destination_storage = destination.fab(local).storage();
    if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemorySpace>::accessible)
      std::copy_n(source_storage.data(), source_storage.extent(0), destination_storage.data());
    else
      Kokkos::deep_copy(destination_storage, source_storage);
  }
}

static void copy_valid_(const field_type& source, field_type& destination) {
  require_same_field_contract_(source, destination, "AMR Program valid-field copy");
  for (std::size_t local = 0; local < destination.local_size(); ++local) {
    const auto input = source.fab(local).view();
    const auto output = destination.fab(local).view();
    const int components = destination.ncomp();
    for_each_cell(destination.box(local), [=] POPS_HD(const Index<Dim>& cell) {
      for (int component = 0; component < components; ++component)
        output(cell, component) = input(cell, component);
    });
  }
}

void laplacian_without_fill_(field_type& output, field_type& input,
                             const Geometry<Dim>& geom) const {
  require_scalar_stencil_(output, input, 1, "AMR Program Laplacian");
  for (std::size_t local = 0; local < output.local_size(); ++local) {
    const auto result = output.fab(local).view();
    const auto value = std::as_const(input).fab(local).view();
    for_each_cell(output.box(local), [=] POPS_HD(const Index<Dim>& cell) {
      Real image = Real(0);
      for (int axis = 0; axis < Dim; ++axis) {
        Index<Dim> lower = cell;
        Index<Dim> upper = cell;
        --lower[axis];
        ++upper[axis];
        const Real spacing = geom.spacing(axis);
        image +=
            (value(upper, 0) - Real(2) * value(cell, 0) + value(lower, 0)) / (spacing * spacing);
      }
      result(cell, 0) = image;
    });
  }
  count_kernel_();
}
void gradient_without_fill_(field_type& output, field_type& input,
                            const Geometry<Dim>& geom) const {
  require_scalar_stencil_(output, input, Dim, "AMR Program gradient");
  for (std::size_t local = 0; local < output.local_size(); ++local) {
    const auto result = output.fab(local).view();
    const auto value = std::as_const(input).fab(local).view();
    for_each_cell(output.box(local), [=] POPS_HD(const Index<Dim>& cell) {
      for (int axis = 0; axis < Dim; ++axis) {
        Index<Dim> lower = cell;
        Index<Dim> upper = cell;
        --lower[axis];
        ++upper[axis];
        result(cell, axis) = (value(upper, 0) - value(lower, 0)) / (Real(2) * geom.spacing(axis));
      }
    });
  }
  count_kernel_();
}
std::uint64_t next_boundary_generation_() const {
  if (boundary_generation_ == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error("AMR Program boundary generation exhausted uint64_t");
  return ++boundary_generation_;
}
void count_kernel_(std::int64_t count = 1) const {
  if (facade_ != nullptr)
    facade_->program_profiler_().count("kernel_launches", count);
}
