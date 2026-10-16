// Authenticated temporal flux producers and legacy history migration.
void install_flux_temporal_families_(
    std::initializer_list<std::tuple<int, int, int, std::string_view>> rows) const {
  std::map<std::tuple<std::size_t, int, FluxBasisProvider>, std::string> candidate;
  std::string contract;
  std::exception_ptr local_error;
  try {
    ExactContractBuilder exact;
    exact.text("pops.program-flux-temporal-families")
        .scalar(std::uint32_t{1})
        .sequence(rows, [&](ExactContractBuilder& entry, const auto& row) {
          const auto& [program_block, rhs_identity, raw_provider, family] = row;
          if (program_block < 0 || rhs_identity < 0 || raw_provider < 0 ||
              raw_provider > static_cast<int>(FluxBasisProvider::GeneratedFace) || family.empty())
            throw std::invalid_argument(
                "AMR Program flux temporal-family declaration is incomplete");
          const auto provider = static_cast<FluxBasisProvider>(raw_provider);
          const int runtime_block = sys_block(program_block);
          if (!candidate
                   .emplace(
                       std::tuple{static_cast<std::size_t>(runtime_block), rhs_identity, provider},
                       std::string(family))
                   .second)
            throw std::invalid_argument(
                "AMR Program flux temporal-family declaration aliases one producer");
          entry.scalar(std::int32_t{program_block})
              .scalar(std::int32_t{runtime_block})
              .scalar(std::int32_t{rhs_identity})
              .scalar(static_cast<std::uint8_t>(provider))
              .text(family);
        });
    contract = std::move(exact).release();
  } catch (...) {
    local_error = std::current_exception();
  }
  const ExecutionLane& lane = prepared_execution_lane();
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR Program flux temporal-family declaration failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"amr-program-flux-temporal-families", contract}},
                                                lane))
    throw std::invalid_argument(
        "AMR Program flux temporal-family declarations differ between ranks");
  if (!declared_flux_temporal_families_.empty() && declared_flux_temporal_families_ != candidate)
    throw std::logic_error("AMR Program flux temporal-family declarations changed after install");
  declared_flux_temporal_families_.swap(candidate);
}

std::string restored_flux_temporal_family_(std::size_t block, int rhs_identity,
                                           FluxBasisProvider provider, std::string_view encoded,
                                           bool legacy) const {
  const auto declared =
      declared_flux_temporal_families_.find(std::tuple{block, rhs_identity, provider});
  if (declared != declared_flux_temporal_families_.end()) {
    if (!legacy && encoded != declared->second)
      throw std::invalid_argument(
          "history flux checkpoint differs from its declared producer family");
    return declared->second;
  }
  if (!declared_flux_temporal_families_.empty())
    throw std::invalid_argument("history flux checkpoint has no declared producer family");
  if (!legacy && !encoded.empty() && provider != FluxBasisProvider::ExactFace)
    throw std::invalid_argument("history flux checkpoint has an undeclared producer family");
  return std::string(encoded);
}

void restore_flux_temporal_family_(FluxBasis& basis, std::string_view encoded, bool legacy) const {
  basis.temporal_family = restored_flux_temporal_family_(basis.runtime_block, basis.rhs_identity,
                                                         basis.provider, encoded, legacy);
}

static bool same_flux_producer_(const FluxBasis& active, const FluxBasis& retained) noexcept {
  return active.runtime_block == retained.runtime_block && active.level == retained.level &&
         active.rhs_identity == retained.rhs_identity && active.provider == retained.provider &&
         (retained.temporal_family.empty() || active.temporal_family == retained.temporal_family);
}

// An installed producer registry is authoritative; a missing or foreign family
// cannot turn a modern producer into a legacy exact-pack comparison.
std::string_view declared_flux_temporal_family_(const FluxBasis& basis) const {
  if (declared_flux_temporal_families_.empty())
    return {};
  const auto declared = declared_flux_temporal_families_.find(
      std::tuple{basis.runtime_block, basis.rhs_identity, basis.provider});
  if (declared == declared_flux_temporal_families_.end() || declared->second.empty() ||
      basis.temporal_family != declared->second)
    throw std::invalid_argument("AMR Program flux basis lacks its declared producer family");
  return declared->second;
}

// Render every operator occurrence field and the exact coefficient together.
// Reflux may authenticate another level's own coefficient without rewriting it.
static std::string flux_expression_stage_identity_(const FluxBasis& basis,
                                                   const ::pops::ExactWeight& weight) {
  return "pops.program-flux-expression.v1/provider/" +
         std::to_string(static_cast<unsigned int>(basis.provider)) + "/rhs/" +
         std::to_string(basis.rhs_identity) + "/point-stage/" + std::to_string(basis.point.stage) +
         "/basis/" + std::to_string(basis.identity) + "/dt-power/1/weight/" + weight.text() +
         "/stage/" + std::to_string(basis.point.stage_fraction.numerator) + "/" +
         std::to_string(basis.point.stage_fraction.denominator);
}

// A coarse fragment retains its own quadrature descriptor. Authenticate that
// descriptor against its own declared spatial producer and stored measure;
// comparing it to a fine-level descriptor would forbid legitimate local methods.
template <class Entry>
void require_declared_flux_fragment_stage_(std::size_t runtime_block, const Entry& entry) const {
  const auto invalid = [] {
    throw std::invalid_argument(
        "AMR Program coarse flux stage differs from its declared producer or measure");
  };
  std::string_view remaining(entry.key.stage);
  const auto read = [&](std::string_view label, auto& value) {
    if (!remaining.starts_with(label))
      invalid();
    remaining.remove_prefix(label.size());
    const auto end = remaining.find('/');
    const auto token = remaining.substr(0, end);
    const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value);
    if (token.empty() || parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size())
      invalid();
    remaining.remove_prefix(token.size());
  };
  int provider = -1, rhs = -1, stage = -1, power = 0;
  std::uint64_t identity = 0;
  std::int64_t stage_numerator = 0, stage_denominator = 0;
  read("pops.program-flux-expression.v1/provider/", provider);
  read("/rhs/", rhs);
  read("/point-stage/", stage);
  read("/basis/", identity);
  read("/dt-power/", power);
  const auto& weight = entry.measure.stage_weight;
  const std::string weight_token = "/weight/" + weight.text();
  if (!remaining.starts_with(weight_token))
    invalid();
  remaining.remove_prefix(weight_token.size());
  read("/stage/", stage_numerator);
  read("/", stage_denominator);
  if (!remaining.empty() || provider < 0 ||
      provider > static_cast<int>(FluxBasisProvider::GeneratedFace) || rhs < 0 || stage < 0 ||
      power != 1 || weight.zero() || stage_denominator <= 0 || stage_numerator < 0 ||
      stage_numerator > stage_denominator)
    invalid();
  FluxBasis source;
  source.runtime_block = runtime_block;
  source.provider = static_cast<FluxBasisProvider>(provider);
  source.rhs_identity = rhs;
  source.identity = identity;
  source.point.stage = stage;
  source.point.stage_fraction = ::pops::amr::Rational(stage_numerator, stage_denominator);
  source.temporal_family = entry.key.temporal_family;
  if ((source.provider != FluxBasisProvider::ExactFace && source.point.stage != rhs) ||
      entry.key.stage != flux_expression_stage_identity_(source, weight) ||
      weight != entry.measure.stage_weight ||
      entry.key.clock.phase !=
          entry.measure.substep_begin + (entry.measure.substep_end - entry.measure.substep_begin) *
                                            source.point.stage_fraction)
    invalid();
  const auto declared =
      declared_flux_temporal_families_.find(std::tuple{runtime_block, rhs, source.provider});
  if (declared == declared_flux_temporal_families_.end() ||
      declared->second != source.temporal_family)
    invalid();
}
