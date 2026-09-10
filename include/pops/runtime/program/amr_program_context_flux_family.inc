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
              raw_provider > static_cast<int>(FluxBasisProvider::DiffusiveFace) || family.empty())
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
