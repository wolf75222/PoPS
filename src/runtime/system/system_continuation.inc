// Included in system.cpp, within namespace pops. No scientific-model dispatch.
template<int Dim>
std::string System<Dim>::checkpoint_continuation_schema() const {
  const auto& store = p_->program_.continuation_;
  return store.configured() ? store.configuration().identity : std::string{};
}

template<int Dim>
std::uint64_t System<Dim>::checkpoint_continuation_capacity() const {
  const auto& store = p_->program_.continuation_;
  return store.configured() ? store.configuration().byte_capacity : 0;
}

template<int Dim>
void System<Dim>::require_continuation_projection() const {
  std::exception_ptr error;
  try {
    const auto& store = p_->program_.continuation_;
    if (store.configured()) {
      store.require_frontier(p_->t, p_->macro_step_);
      for (const auto& binding : store.configuration().fields) {
        const int block = p_->program_.block_map().at(static_cast<std::size_t>(binding.program_block));
        runtime::program::require_continuation_projection<Dim>(
            runtime::program::continuation_at<Dim>(store.read(), binding.path)->field(),
            p_->sp.at(static_cast<std::size_t>(block)).U);
      }
    }
  } catch (...) { error = std::current_exception(); }
  try { Kokkos::fence(); } catch (...) { error = std::current_exception(); }
  collectively_rethrow_exception(error, prepared_boundary_execution_lane(), "Continuation accepted projection");
}

template<int Dim>
std::vector<std::uint8_t> System<Dim>::checkpoint_continuation_image() const {
  const auto& lane = prepared_boundary_execution_lane();
  std::vector<std::uint8_t> bytes;
  std::exception_ptr error;
  try {
    const auto& store = p_->program_.continuation_;
    if (!store.configured()) throw std::logic_error("Continuation capture requires an installed schema");
    if (p_->external_restart_transaction_ || step_transaction_depth() > 1)
      throw std::logic_error("Continuation capture requires a complete root accepted image");
    store.require_frontier(p_->t, p_->macro_step_);
    bytes = store.checkpoint([&](int owner) -> const MultiFab<Dim>& {
      const int block = p_->program_.block_map().at(static_cast<std::size_t>(owner));
      return p_->sp.at(static_cast<std::size_t>(block)).U;
    });
  } catch (...) { error = std::current_exception(); }
  try { Kokkos::fence(); } catch (...) { error = std::current_exception(); }
  collectively_rethrow_exception(error, lane, "Continuation checkpoint capture");
  require_continuation_projection();
  return bytes;
}

template<int Dim>
void System<Dim>::validate_checkpoint_continuation_image(std::span<const std::uint8_t> bytes,
                                                        double time, int step) const {
  // Local preflight only. The surrounding checkpoint prepare protocol converges every exception
  // before beginning its native restart transaction; no rank enters a collective from this body.
  const auto& store = p_->program_.continuation_;
  if (!store.configured()) throw std::logic_error("Continuation restart has no installed schema");
  const auto prepared = store.prepare_restart(bytes, [&](int owner) -> const MultiFab<Dim>& {
    const int block = p_->program_.block_map().at(static_cast<std::size_t>(owner));
    return p_->sp.at(static_cast<std::size_t>(block)).U;
  }, &prepared_boundary_execution_lane());
  const auto& configuration = store.configuration();
  if (std::bit_cast<std::uint64_t>(runtime::program::continuation_at<Dim>(prepared.value, configuration.time_path)->real()) !=
          std::bit_cast<std::uint64_t>(time) ||
      runtime::program::continuation_at<Dim>(prepared.value, configuration.step_path)->integer() != step)
    throw std::invalid_argument("Continuation checkpoint and enclosing accepted frontier differ");
  Kokkos::fence();
}

template<int Dim>
void System<Dim>::restore_checkpoint_continuation_image(std::span<const std::uint8_t> bytes) {
  using Store = runtime::program::ContinuationStore<Dim>;
  const auto& lane = prepared_boundary_execution_lane();
  std::optional<typename Store::Prepared> prepared;
  std::string identity;
  std::exception_ptr error;
  try {
    if (!p_->external_restart_transaction_ || p_->external_step_transaction_committed_)
      throw std::logic_error("Continuation restore requires an open native restart transaction");
    auto& store = p_->program_.continuation_;
    auto prototype = [&](int owner) -> const MultiFab<Dim>& {
      const int block = p_->program_.block_map().at(static_cast<std::size_t>(owner));
      return p_->sp.at(static_cast<std::size_t>(block)).U;
    };
    prepared.emplace(store.prepare_restart(bytes, prototype, &lane));
    const auto& configuration = store.configuration();
    if (std::bit_cast<std::uint64_t>(runtime::program::continuation_at<Dim>(prepared->value, configuration.time_path)->real()) !=
            std::bit_cast<std::uint64_t>(p_->t) ||
        runtime::program::continuation_at<Dim>(prepared->value, configuration.step_path)->integer() != p_->macro_step_)
      throw std::invalid_argument("Continuation restore and restored native frontier differ");
    // Ordinary checkpoint readers stage each carrier first. The structural image must match
    // those exact valid cells; its bytes never silently overwrite a conflicting second truth.
    for (const auto& binding : configuration.fields)
      runtime::program::require_continuation_projection<Dim>(
          runtime::program::continuation_at<Dim>(prepared->value, binding.path)->field(),
          prototype(binding.program_block));
    identity = store.codec(prototype).collective_identity(prepared->value, prepared->publication);
  } catch (...) { error = std::current_exception(); }
  try { Kokkos::fence(); } catch (...) { error = std::current_exception(); }
  collectively_rethrow_exception(error, lane, "Continuation restart preparation");
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"continuation-restart", identity}}, lane))
    throw std::runtime_error("Continuation restored replicated values differ between ranks");
  const auto closure = runtime::program::continuation_collective_closure<Dim>(prepared->value,
      prepared->distributed_registry.get(), lane, prepared->configuration->byte_capacity,prepared->realm);
  error = {};
  try {
    if (prepared->closure) {
      if (prepared->closure->domains != closure->domains || prepared->closure->arrays != closure->arrays)
        throw std::invalid_argument("checkpoint distributed tables are not the exact global dependency closure");
    } else if (!closure->domains.empty() || !closure->arrays.empty())
      throw std::invalid_argument("checkpoint omitted its distributed dependency closure");
  } catch (...) { error = std::current_exception(); }
  collectively_rethrow_exception(error, lane, "Continuation global restart closure");
  // Revoke attempts before the no-throw swap; decoded data is never a live capability.
  p_->program_.reject_resource_work();
  p_->program_.continuation_.publish(std::move(*prepared));
}

template std::string System<kNativeDimension>::checkpoint_continuation_schema() const;
template std::uint64_t System<kNativeDimension>::checkpoint_continuation_capacity() const;
template std::vector<std::uint8_t> System<kNativeDimension>::checkpoint_continuation_image() const;
template void System<kNativeDimension>::validate_checkpoint_continuation_image(std::span<const std::uint8_t>, double, int) const;
template void System<kNativeDimension>::restore_checkpoint_continuation_image(std::span<const std::uint8_t>);
template void System<kNativeDimension>::require_continuation_projection() const;
