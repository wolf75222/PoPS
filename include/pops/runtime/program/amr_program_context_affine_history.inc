// Included inside AmrProgramContext. The callback supplies only the temporal
// matrix; source values and publication identities remain native authorities.
private:
struct AffineHistoryTransfer {
  std::string identity;
  std::size_t slots = 0;
  bool state_endpoints = false;
  HistoryTransferCallback callback;
};
mutable std::map<std::string, AffineHistoryTransfer> affine_history_transfers_;
struct AffineHistoryCandidate {
  std::vector<field_type> fields;
  std::vector<Real> intervals;
  std::vector<HistorySampleIdentity> samples;
  std::vector<FluxExpression> expressions;
  int fill_count = 0;
};
using AffineHistories = std::map<std::string, AffineHistoryCandidate>;

std::string affine_history_invocation_(
    const std::string& key, const AffineHistoryTransfer& transfer,
    const ::pops::amr::reflux::AffineIntervalPublication& publication) const {
  ExactContractBuilder contract;
  contract.text("pops.affine-history-invocation.v1")
      .text(key)
      .bytes(transfer.identity)
      .scalar(runtime_->topology_epoch())
      .scalar(runtime_->materialization_generation())
      .scalar(accepted_subcycling_attempt_)
      .scalar(publication.source_start)
      .scalar(publication.source_duration)
      .scalar(publication.accepted_duration)
      .scalar(publication.increment_scale)
      .scalar(publication.source_start + publication.accepted_duration)
      .scalar(publication.source_duration / publication.accepted_duration);
  return std::move(contract).release();
}

AffineHistories prepare_affine_histories_(
    const ::pops::amr::reflux::AffineIntervalPublication& publication) const {
  AffineHistories result;
  const auto& frame = *affine_finalization_;
  const auto& entry = frame.entry_histories;
  const auto& candidate = runtime_state().hist_;
  for (const auto& [key, level] : history_levels_) {
    const auto decoded = decode_history_key_(key);
    if (!decoded || decoded->first != level)
      throw std::logic_error("affine history has a malformed level authority");
    const auto found = affine_history_transfers_.find(decoded->second);
    if (found == affine_history_transfers_.end())
      throw std::invalid_argument("affine acceptance lacks an authored retained-history transfer");
    const auto& transfer = found->second;
    const auto& old = entry.histories.at(key);
    const auto& provisional = candidate.histories.at(key);
    if (old.size() != transfer.slots || provisional.size() != transfer.slots ||
        entry.owner.at(key) != candidate.owner.at(key) ||
        entry.state_identity.at(key) != candidate.state_identity.at(key) ||
        entry.space_identity.at(key) != candidate.space_identity.at(key) ||
        entry.clock_identity.at(key) != candidate.clock_identity.at(key) ||
        entry.interpolation_identity.at(key) != candidate.interpolation_identity.at(key) ||
        entry.store_pending.at(key) || candidate.store_pending.at(key))
      throw std::invalid_argument("affine history source changed its exact ring authority");
    std::vector<field_type> fields;
    std::vector<TemporalSampleSource> sources;
    std::vector<FluxExpression> expressions;
    const auto append = [&](const auto& manager, const auto& flux) {
      const auto& ring = manager.histories.at(key);
      const auto& identities = manager.validated_samples(key);
      for (std::size_t slot = 0; slot < ring.size(); ++slot) {
        fields.emplace_back(ring[slot]);
        sources.push_back(TemporalSampleSource::history(
            identities[slot], manager.initialized.at(key), manager.slot_dt.at(key)[slot]));
        expressions.push_back(flux.at(key).at(slot));
      }
    };
    append(entry, frame.entry_history_flux);
    append(candidate, history_flux_expressions_);
    const std::string invocation = affine_history_invocation_(key, transfer, publication);
    if (transfer.state_endpoints) {
      const auto owner = static_cast<std::size_t>(candidate.owner.at(key));
      const auto& initial = frame.entry.at(owner).at(static_cast<std::size_t>(level));
      const auto& final = frame.provisional.at(owner).at(static_cast<std::size_t>(level));
      require_same_field_contract_(old.front(), initial, "affine history State endpoint");
      field_type accepted = final;
      pops::lincomb(accepted, Real(1 - publication.increment_scale), initial,
                    Real(publication.increment_scale), final);
      for (int endpoint = 0; endpoint < 2; ++endpoint) {
        ExactContractBuilder authority;
        authority.text("pops.native-state-temporal-point.v1")
            .bytes(invocation)
            .text(candidate.state_identity.at(key))
            .text(candidate.space_identity.at(key))
            .scalar(candidate.owner.at(key))
            .scalar(level)
            .scalar(endpoint);
        const double time =
            publication.source_start + (endpoint == 0 ? 0. : publication.accepted_duration);
        fields.emplace_back(endpoint == 0 ? initial : accepted);
        sources.push_back(TemporalSampleSource::point(time, std::move(authority).release()));
        // A newly accepted State point is a new base value. Its integrated
        // transport belongs to the separately authenticated accepted flux ledger.
        expressions.emplace_back();
      }
    }
    HistoryTransferInput input;
    input.exact_coefficient_bytes = max_exact_coefficient_bytes_;
    input.accepted_time = Real(publication.source_start + publication.accepted_duration);
    input.temporal_ratio = Real(publication.source_duration / publication.accepted_duration);
    input.parameters = {Real(publication.source_start), Real(publication.source_duration),
                        Real(publication.accepted_duration), Real(publication.increment_scale)};
    for (const auto& source : sources) {
      input.times.push_back(Real(source.time));
      input.intervals.push_back(Real(source.interval));
    }
    const auto matrix = transfer.callback(input);
    if (matrix.rows.size() != transfer.slots)
      throw std::invalid_argument("affine history transfer changed the target ring width");
    auto distinct = distinct_history_flux_sources(
        expressions, [](const auto& basis) { return std::make_shared<FluxBasis>(*basis); });
    AffineHistoryCandidate output;
    output.fill_count = std::min(static_cast<int>(transfer.slots), entry.fill_count.at(key) + 1);
    for (std::size_t slot = 0; slot < matrix.rows.size(); ++slot) {
      const auto& row = matrix.rows[slot];
      if (!std::isfinite(row.time) || row.time > input.accepted_time ||
          !std::isfinite(row.interval) || !(row.interval > Real(0)) ||
          row.weights.size() != fields.size())
        throw std::invalid_argument("affine history transfer returned an invalid target window");
      validate_history_transfer_weights(row);
      if (slot > 0) {
        const Real next = slot == 1 ? input.accepted_time : matrix.rows[slot - 1].time;
        const Real tolerance =
            Real(32) * std::numeric_limits<Real>::epsilon() *
            std::max({Real(1), std::abs(next), std::abs(row.time), row.interval});
        if (!(next > row.time) || std::abs(next - row.time - row.interval) > tolerance)
          throw std::invalid_argument(
              "affine history target intervals do not join accepted samples");
      }
      output.samples.push_back(
          reconstructed_temporal_sample(sources, row, invocation, key, runtime_->topology_epoch(),
                                        runtime_->materialization_generation()));
      output.intervals.push_back(row.interval);
      output.fields.emplace_back(provisional[slot]);
      auto& target = output.fields.back();
      target.set_val(Real(0));
      FluxExpression expression;
      for (std::size_t column = 0; column < fields.size(); ++column) {
        const auto weight =
            history_transfer_exact_weight(row, column, max_exact_coefficient_bytes_);
        if (weight == ::pops::ExactWeight{0, 1})
          continue;
        require_same_field_contract_(target, fields[column], "affine history source field");
        pops::saxpy(target, row.weights[column], fields[column]);
        if (!distinct[column].empty())
          add_flux_expression_(
              expression, scaled_flux_expression_(distinct[column], ExactPolynomial{{0, weight}}));
      }
      require_flux_expression_budget_(expression);
      output.expressions.push_back(std::move(expression));
      for (std::size_t local = 0; local < target.local_size(); ++local)
        for (int component = 0; component < target.ncomp(); ++component)
          if (pops::for_each_cell_reduce_max(
                  target.box(local),
                  pops::numerics::time::amr::detail::AffineMacroFiniteKernel<Dim>{
                      std::as_const(target).fab(local).view(), component}) > Real(0))
            throw std::overflow_error("affine history reconstruction produced nonfinite values");
    }
    result.emplace(key, std::move(output));
  }
  return result;
}

void publish_affine_histories_(AffineHistories& histories) const {
  std::exception_ptr error;
  try {
    auto& manager = runtime_state().hist_;
    for (auto& [key, candidate] : histories)
      for (std::size_t slot = 0; slot < candidate.fields.size(); ++slot) {
        program_values_.revoke_buffer_(manager.histories.at(key)[slot]);
        copy_valid_(candidate.fields[slot], manager.histories.at(key)[slot]);
      }
  } catch (...) {
    error = std::current_exception();
  }
  converge_owner_reduction_(error, prepared_execution_lane(), "affine history publication");
  auto& manager = runtime_state().hist_;
  for (auto& [key, candidate] : histories) {
    manager.slot_dt.at(key).swap(candidate.intervals);
    manager.slot_sample.at(key).swap(candidate.samples);
    history_flux_expressions_.at(key).swap(candidate.expressions);
    manager.initialized.at(key) = true;
    manager.fill_count.at(key) = candidate.fill_count;
    manager.store_pending.at(key) = false;
  }
}

public:
void register_affine_history(const std::string& name, const std::string& identity,
                             std::size_t slots, bool state_endpoints,
                             HistoryTransferCallback callback) const {
  if (name.empty() || identity.empty() || slots < 2 || !callback)
    throw std::invalid_argument("affine history requires a complete closed transfer graph");
  const auto found = affine_history_transfers_.find(name);
  if (found != affine_history_transfers_.end()) {
    if (found->second.identity != identity || found->second.slots != slots ||
        found->second.state_endpoints != state_endpoints)
      throw std::invalid_argument("affine history changed its installed source contract");
    return;
  }
  affine_history_transfers_.emplace(
      name, AffineHistoryTransfer{identity, slots, state_endpoints, std::move(callback)});
}
