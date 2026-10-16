// Included in history_field_detail: provenance of explicit temporal reconstruction.
inline HistoryFieldMetadata interpolated_metadata(const HistoryFieldMetadata& older,
                                                  const HistoryFieldMetadata& newer, double alpha,
                                                  double target_time,
                                                  const HistoryFieldMetadataBudget& budget) {
  if (!std::isfinite(alpha) || alpha < 0 || alpha > 1 || !std::isfinite(target_time))
    throw std::invalid_argument("history interpolation has an invalid exact bracket");
  if (alpha == 0)
    return older;
  if (alpha == 1)
    return newer;
  if (older.representation || newer.representation)
    throw std::invalid_argument(
        "history interpolation requires discharging Field representation before a nonidentity "
        "transfer");
  if (!older.support && !newer.support)
    return {};
  if (!older.support || !newer.support)
    throw std::invalid_argument(
        "history interpolation mixes qualified and unqualified Field support");
  const auto& a = *older.support;
  const auto& b = *newer.support;
  a.validate();
  b.validate();
  if (a.program_identity != b.program_identity || a.program_owner != b.program_owner ||
      a.producer_level != b.producer_level || a.topology_epoch != b.topology_epoch ||
      a.materialization_generation != b.materialization_generation ||
      a.field_provider != b.field_provider)
    throw std::invalid_argument("history interpolation mixes different physical Field supports");
  budget.validate();
  // ExactContractBuilder frames each scalar/string. This conservative requested
  // byte bound covers both contract copies and the produced immutable support.
  std::uint64_t bytes = 0;
  charge(bytes, 1024, 1, budget.workspace_bytes);
  for (const auto* support : {&a, &b}) {
    charge(bytes, support->program_identity.size(), 2, budget.workspace_bytes);
    charge(bytes, support->origin_point.size(), 2, budget.workspace_bytes);
    charge(bytes, support->field_provider.size(), 2, budget.workspace_bytes);
  }
  ExactContractBuilder contract;
  contract.text("pops.history.linear-interpolation@1")
      .scalar(std::bit_cast<std::uint64_t>(alpha))
      .scalar(std::bit_cast<std::uint64_t>(target_time));
  older.append_contract(contract);
  newer.append_contract(contract);
  const auto origin = std::string("pops.history.linear-interpolation@1:sha256:") +
                      identity::sha256_hex(std::vector<std::uint8_t>(contract.view().begin(),
                                                                     contract.view().end()));
  return {std::make_shared<const ContinuationFieldSupport>(ContinuationFieldSupport{
              a.program_identity, origin, a.program_owner, a.producer_level, a.topology_epoch,
              a.materialization_generation, a.field_provider}),
          {}};
}

inline HistoryFieldMetadata reconstructed_metadata(std::span<const HistoryFieldMetadata> sources,
                                                   std::string_view operation, int level,
                                                   std::uint64_t epoch, std::uint64_t generation,
                                                   const HistoryFieldMetadataBudget& budget) {
  const ContinuationFieldSupport* first = nullptr;
  bool unqualified = false;
  std::uint64_t bytes = 0;
  for (const auto& source : sources) {
    if (source.representation)
      throw std::invalid_argument(
          "history reconstruction requires discharging Field representation before a nonidentity "
          "transfer");
    if (!source.support) {
      unqualified = true;
      continue;
    }
    const auto& support = *source.support;
    support.validate();
    if (first && (first->program_identity != support.program_identity ||
                  first->program_owner != support.program_owner ||
                  first->field_provider != support.field_provider))
      throw std::invalid_argument("history reconstruction mixes different scientific Field owners");
    first = &support;
    charge(bytes, support.program_identity.size(), 2, budget.workspace_bytes);
    charge(bytes, support.origin_point.size(), 2, budget.workspace_bytes);
    charge(bytes, support.field_provider.size(), 2, budget.workspace_bytes);
    charge(bytes, 256, 1, budget.workspace_bytes);
  }
  if (!first)
    return {};
  if (unqualified || operation.empty() || level < 0)
    throw std::invalid_argument("history reconstruction lacks its complete source/target support");
  charge(bytes, operation.size(), 2, budget.workspace_bytes);
  charge(bytes, 1024, 1, budget.workspace_bytes);
  ExactContractBuilder contract;
  contract.text("pops.history.reconstruction@1")
      .bytes(operation)
      .scalar(level)
      .scalar(epoch)
      .scalar(generation);
  for (const auto& source : sources)
    source.append_contract(contract);
  const auto origin = std::string("pops.history.reconstruction@1:sha256:") +
                      identity::sha256_hex(std::vector<std::uint8_t>(contract.view().begin(),
                                                                     contract.view().end()));
  return {std::make_shared<const ContinuationFieldSupport>(
              ContinuationFieldSupport{first->program_identity, origin, first->program_owner, level,
                                       epoch, generation, first->field_provider}),
          {}};
}

struct LinearBracket {
  std::size_t older = 0, newer = 0;
  double alpha = 0, target_time = 0;
  bool endpoint() const noexcept { return older == newer || alpha == 0 || alpha == 1; }
  std::size_t selected() const noexcept { return alpha == 1 ? newer : older; }
};
inline LinearBracket linear_bracket(double coordinate, double accepted_time,
                                    std::span<const double> intervals, int max_lag) {
  if (max_lag < 1 || !std::isfinite(coordinate) || !std::isfinite(accepted_time) ||
      coordinate > 0 || coordinate < -static_cast<double>(max_lag) ||
      intervals.size() <= static_cast<std::size_t>(max_lag))
    throw std::invalid_argument("history interpolation target lies outside retained timestamps");
  if (coordinate == 0)
    return {0, 0, 0, accepted_time};
  const auto lag = static_cast<std::size_t>(std::ceil(-coordinate));
  double newer = accepted_time, older = accepted_time, interval = 0;
  for (std::size_t i = 1; i <= lag; ++i) {
    interval = intervals[i];
    if (!std::isfinite(interval) || interval <= 0)
      throw std::invalid_argument("history interpolation requires positive retained intervals");
    older = newer - interval;
    if (i != lag)
      newer = older;
  }
  const double fraction = coordinate + static_cast<double>(lag);
  // Preserve an exact endpoint without subtracting nearly equal timestamps.
  if (fraction == 0)
    return {lag, lag, 0, older};
  const double target = older + fraction * interval;
  const double alpha = (target - older) / (newer - older);
  if (!std::isfinite(alpha) || alpha < 0 || alpha > 1)
    throw std::invalid_argument("history interpolation cannot bracket retained timestamps");
  return {lag, lag - 1, alpha, target};
}
