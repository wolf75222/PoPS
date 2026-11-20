// A generated predicate and its exact protected Field share this reduction. Backend admission is
// negotiated at installation. The System's prepared box tiling and embedded geometry are fixed
// on the lane, and every invocation checks its actual Field before exposing a view to the kernel.
bool prepare_owned_guard_host_lane() const {
  constexpr bool host =
      std::is_same_v<typename Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace>;
  return all_reduce_min(host ? 1L : 0L, prepared_execution_lane()) != 0;
}

bool owned_guard_unmasked() const noexcept {
  return system_->owned_guard_unmasked_();
}

/// Reduce cell-local codes 0 (valid), 1 (predicate false), or 2 (checked expression invalid).
/// A checked producer reads the same Field that the guard aliases. Its representation check
/// precedes evaluation; a raw reduction checks the alias after its reduction has succeeded.
template <bool CheckedExpression, class Check>
Real owned_guard_fold(int program_block, const field_type& field, int components,
                      Check check) const {
  const ExecutionLane& lane = prepared_execution_lane();
  ConsensusWords<kPointwiseRouteWords + runtime::system::kPreparedActiveMaskWords + 4> words;
  const PointwiseRouteSlots route = add_pointwise_route_consensus_(words, program_block);
  runtime::system::PreparedActiveMaskLocal<field_type> mask;
  if (route.runtime_block >= 0)
    mask = system_->prepared_program_block_active_mask_local_(route.runtime_block, field, lane);
  const auto active = runtime::system::add_active_mask_consensus(words, mask);
  const std::size_t layout = words.failure(mask.mode != -1 || field.ncomp() != components);

  std::exception_ptr representation_error;
  try {
    require_discharged_field_representation(field);
  } catch (...) {
    representation_error = std::current_exception();
  }
  const std::size_t representation = words.failure(static_cast<bool>(representation_error));

  Real local = Real(0);
  std::exception_ptr reduction_error;
  if (route.runtime_block >= 0 && !mask.foreign_lane && !mask.invalid && mask.mode == -1 &&
      field.ncomp() == components && (!CheckedExpression || !representation_error)) {
    try {
      for (std::size_t box = 0; box < field.local_size(); ++box) {
        const auto view = field.fab(box).view();
        const Real value = pops::for_each_cell_reduce_max(
            field.box(box), KOKKOS_LAMBDA(const Index<Dim>& index) { return check(view, index); });
        local = std::max(local, value);
      }
    } catch (...) {
      reduction_error = std::current_exception();
      local = Real(0);
    }
  }
  const std::size_t reduction = words.failure(static_cast<bool>(reduction_error));
  const std::size_t maximum = words.maximum(static_cast<double>(local));
  words.reduce(lane);
  if constexpr (CheckedExpression) {
    if (words.failed(representation))
      collectively_rethrow_exception(representation_error, lane,
                                     "numerical producer cannot discard a Field representation");
  }
  require_pointwise_route_consensus_(words, route);
  runtime::system::require_active_mask_consensus(words, active);
  if (words.failed(layout))
    throw std::invalid_argument("Program guard reduction requires its exact unmasked Field layout");
  if (words.failed(reduction)) {
    if (lane.size() == 1 && reduction_error)
      std::rethrow_exception(reduction_error);
    throw std::runtime_error(CheckedExpression
                                 ? "Program pointwise status reduction failed collectively"
                                 : "Program min reduction failed collectively");
  }
  if constexpr (!CheckedExpression) {
    if (words.failed(representation))
      collectively_rethrow_exception(representation_error, lane,
                                     "numerical producer cannot discard a Field representation");
  }
  return static_cast<Real>(words.value(maximum));
}
