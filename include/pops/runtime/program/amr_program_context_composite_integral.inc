// Included inside AmrProgramContext. Authored scalar algebra is integrated on
// authenticated immutable composite support, optionally restricted to one level.
public:
template <class Integrand>
Real composite_integral(int program_block, Integrand integrand,
                        std::optional<std::int64_t> selected_level = std::nullopt) const {
  const auto& lane = prepared_execution_lane();
  std::exception_ptr error;
  std::string domain_contract;
  int owner = -1;
  try {
    require_affine_finalization_("composite_integral");
    const auto& frame = *affine_finalization_;
    owner = sys_block(program_block);
    const auto levels = frame.entry.at(owner).size();
    if (selected_level &&
        (*selected_level < 0 || static_cast<std::uint64_t>(*selected_level) >= levels))
      throw std::invalid_argument("composite_integral selected level is outside its hierarchy");
    if (frame.provisional.at(owner).size() != levels || frame.measures.at(owner).size() != levels)
      throw std::invalid_argument("composite_integral has inconsistent immutable towers");
    ExactContractBuilder bytes;
    bytes.text("pops.composite-integral-domain@1")
        .scalar(program_block)
        .scalar(owner)
        .scalar(levels)
        .scalar(frame.topology_epoch)
        .scalar(frame.materialization_generation)
        .scalar(selected_level.has_value())
        .scalar(selected_level.value_or(0));
    domain_contract = std::move(bytes).release();
  } catch (...) {
    error = std::current_exception();
  }
  // Converge range/authority failures before a peer can reduce a different domain.
  converge_owner_reduction_(error, lane, "authored composite integration domain");
  const std::array<std::pair<std::string_view, std::string_view>, 1> domain_parts{
      {{"composite-integral-domain", domain_contract}}};
  if (!all_ranks_agree_exact_ordered_byte_pairs(domain_parts, lane))
    throw std::invalid_argument("composite_integral selected domain differs between ranks");
  FiniteCompensatedSum accumulated;
  try {
    const auto& frame = *affine_finalization_;
    const auto& entry = frame.entry.at(owner);
    const auto& current = frame.provisional.at(owner);
    for (std::size_t level = 0; level < entry.size(); ++level) {
      if (selected_level && level != static_cast<std::uint64_t>(*selected_level))
        continue;
      const auto& a = entry[level];
      const auto& b = current.at(level);
      require_same_field_contract_(a, b, "composite_integral");
      const auto& measure = frame.measures.at(owner).at(level);
      const auto* active = measure.active ? &*measure.active : nullptr;
      const auto* fraction = measure.fraction ? &*measure.fraction : nullptr;
      const auto& coverage = measure.coverage;
      mf_arith_detail::validate_measure(a, {active, nullptr}, "composite_integral");
      mf_arith_detail::validate_measure(a, {fraction, nullptr},
                                        "composite_integral volume fraction");
      mf_arith_detail::validate_measure(a, {&coverage, nullptr}, "composite_integral");
      const auto& geometry = measure.geometry;
      double volume = 1.;
      for (int axis = 0; axis < Dim; ++axis)
        volume *= geometry.spacing(axis);
      if (!(volume > 0.) || !std::isfinite(volume))
        throw std::invalid_argument("composite_integral requires finite positive cell volumes");
      for (std::size_t local = 0; local < a.local_size(); ++local) {
        integrand.entry = a.fab(local).view();
        integrand.current = b.fab(local).view();
        const auto sum = pops::for_each_cell_reduce_finite_sum(
            a.box(local),
            detail::CompositeIntegralKernel<Dim, Integrand>{
                integrand, active ? active->fab(local).view() : FieldView<const Real, Dim>{},
                coverage.fab(local).view(),
                fraction ? fraction->fab(local).view() : FieldView<const Real, Dim>{}, Real(volume),
                active != nullptr, fraction != nullptr});
        if (!sum.finite())
          throw std::overflow_error("composite_integral has nonfinite active values");
        if (!a.distribution().replicated() || lane.rank() == 0) {
          accumulated.add(sum.high);
          accumulated.add(sum.low);
        }
      }
    }
    if (!accumulated.finite())
      throw std::overflow_error("composite_integral has nonfinite accumulation");
  } catch (...) {
    error = std::current_exception();
  }
  const CollectiveCompensatedSum reduced = pops::collective_finite_compensated_sum_or_failure(
      accumulated, static_cast<bool>(error), lane.communicator());
  return reduced.or_vote(
      [&] { converge_owner_reduction_(error, lane, "authored physical composite integral"); });
}
