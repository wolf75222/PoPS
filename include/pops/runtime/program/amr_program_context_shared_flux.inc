using shared_flux_sample_type = runtime::multiblock::InterfaceFluxSample;
using shared_flux_samples_type = std::vector<std::shared_ptr<const shared_flux_sample_type>>;

shared_flux_samples_type capture_grouped_shared_flux_(
    int group_id, const std::vector<const RhsGroupRequest*>& requests,
    const std::vector<runtime::multiblock::BoundaryEvaluationPoint>& points,
    std::vector<field_type>& candidates) const {
  std::vector<field_type*> states, rhs;
  runtime::multiblock::BoundaryEvaluationPoint point;
  std::exception_ptr error;
  const ExecutionLane& lane = prepared_execution_lane();
  try {
    states.assign(static_cast<std::size_t>(n_blocks()), nullptr);
    rhs.assign(states.size(), nullptr);
    point = points.front();
    point.stage = group_id;
    point.graph_identity = facade_->installed_program_hash();
    point.rate_identity = "shared-rhs/" + std::to_string(group_id);
    point.application_identity = "program-rhs-group";
    for (std::size_t index = 0; index < requests.size(); ++index) {
      auto same = points[index];
      same.stage = group_id;
      same.graph_identity = point.graph_identity;
      same.rate_identity = point.rate_identity;
      same.application_identity = point.application_identity;
      if (same != point)
        throw std::invalid_argument("shared RHS endpoints require one exact StagePoint");
      const std::size_t block = static_cast<std::size_t>(requests[index]->block);
      if (block >= states.size())
        throw std::invalid_argument("shared RHS endpoint has no Program block");
      states[block] = requests[index]->state;
      rhs[block] = &candidates[index];
    }
  } catch (...) { error = std::current_exception(); }
  collectively_rethrow_exception(error, lane, "shared RHS capture preparation failed collectively");
  auto samples = facade_->capture_prepared_amr_interface_residual(point, states, rhs);
  shared_flux_samples_type retained;
  error = nullptr;
  try {
    retained.reserve(samples.size());
    for (auto& sample : samples)
      retained.push_back(std::make_shared<const shared_flux_sample_type>(std::move(sample)));
  } catch (...) { error = std::current_exception(); }
  collectively_rethrow_exception(error, lane, "shared RHS retention failed collectively");
  return retained;
}

void attach_grouped_shared_flux_(const std::vector<const RhsGroupRequest*>& requests,
                                 const std::vector<int>& runtime_blocks,
                                 const shared_flux_samples_type& samples,
                                 FluxExpressionRegistry& registry) const {
  for (std::size_t index = 0; index < requests.size(); ++index) {
    auto& expression = registry.at(requests[index]->rhs);
    if (expression.size() != 1 || !expression.begin()->second.basis)
      throw std::logic_error("shared RHS attachment requires its endpoint-owned rate basis");
    auto& term = expression.begin()->second;
    FluxBasis basis = *term.basis;
    for (const auto& sample : samples)
      if (sample->left_block == static_cast<std::size_t>(runtime_blocks[index]) ||
          sample->right_block == static_cast<std::size_t>(runtime_blocks[index]))
        basis.shared_samples.push_back(sample);
    term.basis = std::make_shared<const FluxBasis>(std::move(basis));
  }
}

// Source identity never contains a consumed-attempt ordinal. Copies and lag reads of the same
// earned physical sample aggregate under one identity, while distinct source stages stay distinct.
static std::string shared_flux_source_identity_(const shared_flux_sample_type& sample) {
  ExactContractBuilder exact;
  const auto& point = sample.source_point;
  exact.text(sample.route_contract).scalar(sample.source_topology_epoch).text(point.clock).scalar(point.tick).scalar(point.level)
      .scalar(point.substep).scalar(point.stage).scalar(point.stage_fraction.numerator)
      .scalar(point.stage_fraction.denominator).scalar(point.dt).scalar(point.physical_time)
      .text(point.graph_identity).text(point.rate_identity).text(point.application_identity);
  return std::move(exact).release();
}

static std::string shared_flux_stage_(const shared_flux_sample_type& sample,
                                      ::pops::amr::Rational weight,
                                      const std::string& source_identity, bool projected) {
  std::string stage = "shared-source/" + std::to_string(sample.source_point.tick) + "/" +
                      std::to_string(sample.source_point.substep) + "/" +
                      std::to_string(sample.source_point.stage) + "/weight/" +
                      std::to_string(weight.numerator) + "/" + std::to_string(weight.denominator);
  // A projected parent and a current child can share all rebased consumer coordinates. Retain
  // the complete physical source identity, including its level/epoch, without attempt ordinals.
  // The slash suffix cannot alias the historical format, whose final token is an integer.
  if (projected)
    stage += "/source/" + identity::sha256_hex(std::vector<std::uint8_t>(source_identity.begin(),
                                                                         source_identity.end()));
  return stage;
}

static std::string shared_flux_source_binding_(std::size_t block,
                                               const shared_flux_samples_type& samples) {
  ExactContractBuilder exact;
  exact.text("pops.amr.shared-history-source-association.v1").scalar(block);
  std::size_t count = 0;
  for (const auto& sample : samples) {
    if (!sample)
      throw std::invalid_argument("shared history source association contains an absent sample");
    if (sample->left_block == block || sample->right_block == block) {
      exact.bytes(shared_flux_source_identity_(*sample));
      ++count;
    }
  }
  if (count == 0)
    return {};
  exact.scalar(count);
  return std::move(exact).release();
}

std::optional<runtime::multiblock::InterfaceFluxSampleProjection<Dim>>
prepare_shared_flux_projection_(const FluxBasis& basis,
                                const shared_flux_sample_type& sample) const {
  if (sample.left_block != basis.runtime_block && sample.right_block != basis.runtime_block)
    throw std::invalid_argument("shared flux projection names a foreign basis endpoint");
  if (sample.level == basis.level) {
    facade_->authenticate_prepared_amr_interface_sample(sample);
    return {};
  }
  if (!basis.history_snapshot)
    throw std::invalid_argument("shared history projection lacks its physical face archive");
  history_flux::require_shared_source_binding(
      basis.history_snapshot->source_identity,
      shared_flux_source_binding_(basis.runtime_block, basis.shared_samples));
  const auto lineage = history_flux::require_projection_lineage(
      basis.history_snapshot, sample.level, basis.level, sample.component_count,
      facade_->configured_n_levels(),
      [&](int level) { return facade_->prepared_amr_level_geometry(level); });
  auto projection = facade_->prepare_prepared_amr_interface_sample_projection(sample, basis.level);
  ExactContractBuilder contract;
  contract.bytes(lineage).bytes(projection.exact_contract);
  projection.exact_contract = std::move(contract).release();
  return projection;
}

void materialize_active_shared_flux_() const {
  struct Consumption {
    std::shared_ptr<const shared_flux_sample_type> sample;
    std::optional<runtime::multiblock::InterfaceFluxSampleProjection<Dim>> projection;
    ::pops::amr::ClockWindow window;
    runtime::multiblock::BoundaryEvaluationPoint point;
    ExactPolynomial left, right;
  };
  std::map<std::string, Consumption> consumed;
  std::string route_contract;
  ExactContractBuilder routes;
  std::optional<typename interface_flux_ledger_type::PreparedAccumulation> publication;
  const ExecutionLane& lane = prepared_execution_lane();
  std::exception_ptr error;
  try {
    for (std::size_t block = 0; block < active_attempt_states_.size(); ++block) {
      const FluxExpression expression = active_flux_expression_(*active_attempt_states_[block]);
      require_flux_expression_budget_(expression);
      for (const auto& [identity, term] : expression) {
        if (!term.basis || term.basis->identity != identity || term.basis->runtime_block != block)
          throw std::invalid_argument("shared flux consumer lost endpoint-owned basis authority");
        for (const auto& sample : term.basis->shared_samples) {
          if (!sample || term.basis->level != active_level_ ||
              (sample->left_block != block && sample->right_block != block))
            throw std::invalid_argument("shared flux sample names a foreign consumer endpoint");
          auto projection = prepare_shared_flux_projection_(*term.basis, *sample);
          routes.bytes(projection ? projection->exact_contract
                                  : facade_->authenticate_prepared_amr_interface_sample(*sample));
          auto [entry, inserted] = consumed.try_emplace(shared_flux_source_identity_(*sample));
          auto& value = entry->second;
          auto point = term.basis->point;
          point.stage = sample->source_point.stage;
          point.graph_identity = sample->source_point.graph_identity;
          point.rate_identity = sample->source_point.rate_identity;
          point.application_identity = sample->source_point.application_identity;
          if (inserted) {
            value.sample = sample;
            value.projection = std::move(projection);
            value.window = term.basis->window;
            value.point = point;
          } else if (value.sample->flux_density != sample->flux_density ||
                     value.sample->face_measure != sample->face_measure ||
                     value.projection.has_value() != projection.has_value() ||
                     (projection &&
                      (value.projection->level != projection->level ||
                       value.projection->face_measure != projection->face_measure ||
                       value.projection->source_extents != projection->source_extents ||
                       value.projection->target_extents != projection->target_extents)) ||
                     (value.window.begin != term.basis->window.begin ||
                      value.window.end != term.basis->window.end) ||
                     value.point != point) {
            throw std::invalid_argument("shared source sample has inconsistent paired provenance");
          }
          add_exact_polynomial_(sample->left_block == block ? value.left : value.right,
                                term.coefficient);
        }
      }
    }
    std::vector<typename interface_flux_ledger_type::Entry> entries;
    for (const auto& [identity, value] : consumed) {
      if (value.left != value.right)
        throw std::invalid_argument("shared interface endpoint quadrature coefficients disagree");
      if (value.left.empty())
        continue;
      if (value.left.size() != 1 || value.left.begin()->first != 1)
        throw std::invalid_argument("shared interface quadrature is not an exact dt integral");
      const auto weight = value.left.begin()->second;
      const auto& sample = *value.sample;
      const auto& point = value.point;
      const auto projected_density =
          value.projection ? value.projection->apply(sample) : std::vector<Real>{};
      const auto& density = value.projection ? projected_density : sample.flux_density;
      const double measure =
          value.projection ? value.projection->face_measure : sample.face_measure;
      const double duration = active_flux_interval_duration_(value.window);
      if (point.level != active_level_ || point.substep != logical_substep_ ||
          point.dt != current_dt_ || duration != current_dt_ ||
          value.window.begin != active_subcycling_window_.begin ||
          value.window.end.level != active_subcycling_window_.end.level ||
          value.window.end.macro_step != active_subcycling_window_.end.macro_step ||
          value.window.end.phase != active_subcycling_window_.end.phase ||
          (value.window.end.physical_time != active_subcycling_window_.end.physical_time &&
           value.window.end.physical_time != current_interval_start_time_ + current_dt_))
        throw std::invalid_argument("shared interface consumption names a foreign attempt window");
      const auto clock = runtime::multiblock::clock_stamp_in_window(point, value.window);
      const std::string stage =
          shared_flux_stage_(sample, weight, identity, value.projection.has_value());
      const auto append = [&](int coarse, int fine, ::pops::amr::InterfaceFluxOrientation orientation) {
        ::pops::amr::InterfaceFluxFragmentKey key{
            sample.interface_identity, runtime_->topology_epoch(), coarse, fine, clock, stage,
            point.graph_identity, point.rate_identity, point.application_identity, value.window,
            orientation, sample.left_block, sample.right_block};
        entries.push_back({std::move(key), {weight, measure, duration, true}, density});
      };
      if (active_level_ > 0)
        append(active_level_ - 1, active_level_, ::pops::amr::InterfaceFluxOrientation::FineOutward);
      if (active_level_ + 1 < nlev())
        append(active_level_, active_level_ + 1, ::pops::amr::InterfaceFluxOrientation::CoarseOutward);
    }
    route_contract = std::move(routes).release();
    if (!entries.empty()) {
      if (!interface_flux_ledger_ || !interface_flux_ledger_->in_transaction())
        throw std::logic_error("shared flux publication has no active accepted-state transaction");
      publication.emplace(interface_flux_ledger_->prepare_accumulation(std::move(entries)));
    }
  } catch (...) { error = std::current_exception(); }
  collectively_rethrow_exception(error, lane, "shared quadrature preparation failed collectively");
  // Digest certificates retain a bounded wire identity; live route consensus remains exact bytes.
  if (!all_ranks_agree_exact_ordered_byte_pairs({{"shared-live-routes", route_contract}}, lane))
    throw std::invalid_argument("retained shared flux live routes differ across execution ranks");
  const long present = publication ? 1L : 0L;
  if (all_reduce_min(present, lane) != all_reduce_max(present, lane))
    throw std::invalid_argument("shared quadrature publication presence differs across ranks");
  if (publication) {
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"shared-quadrature", publication->exact_contract()}}, lane))
      throw std::invalid_argument("shared quadrature differs across execution ranks");
    interface_flux_ledger_->publish_prepared_accumulation(*publication);
  }
}

static void write_shared_flux_sample_(checkpoint_detail::Writer& out,
                                      const shared_flux_sample_type& sample) {
  out.string(sample.interface_identity);
  out.string(sample.route_contract);
  out.u64(sample.left_block);
  out.u64(sample.right_block);
  out.i32(sample.level);
  out.u64(sample.source_topology_epoch);
  out.real(sample.face_measure);
  out.i32(sample.left_axis); out.i32(sample.right_axis);
  out.u64(static_cast<std::uint64_t>(sample.left_side));
  out.u64(static_cast<std::uint64_t>(sample.right_side));
  out.real(sample.left_normal_spacing); out.real(sample.right_normal_spacing);
  out.size(sample.face_count); out.i32(sample.component_count);
  out.size(sample.right_component_for_left.size());
  for (int component : sample.right_component_for_left) out.i32(component);
  const auto& point = sample.source_point;
  out.string(point.clock);
  out.i64(point.tick);
  out.i32(point.level);
  out.i32(point.substep);
  out.i32(point.stage);
  out.i64(point.stage_fraction.numerator);
  out.i64(point.stage_fraction.denominator);
  out.real(point.dt);
  out.real(point.physical_time);
  out.string(point.graph_identity);
  out.string(point.rate_identity);
  out.string(point.application_identity);
  out.size(sample.flux_density.size());
  for (const Real value : sample.flux_density)
    out.real(static_cast<double>(value));
}

std::shared_ptr<const shared_flux_sample_type> read_shared_flux_sample_(
    checkpoint_detail::Reader& in, std::size_t endpoint) const {
  shared_flux_sample_type sample;
  sample.interface_identity = in.string();
  sample.route_contract = in.string();
  sample.left_block = in.u64();
  sample.right_block = in.u64();
  sample.level = in.i32();
  sample.source_topology_epoch = in.u64();
  sample.face_measure = in.real();
  sample.left_axis = in.i32(); sample.right_axis = in.i32();
  const auto left_side = in.u64(), right_side = in.u64();
  if (left_side > static_cast<std::uint64_t>(runtime::multiblock::InterfaceSide::High) ||
      right_side > static_cast<std::uint64_t>(runtime::multiblock::InterfaceSide::High))
    throw std::invalid_argument("retained shared endpoint side is invalid");
  sample.left_side = static_cast<runtime::multiblock::InterfaceSide>(left_side);
  sample.right_side = static_cast<runtime::multiblock::InterfaceSide>(right_side);
  sample.left_normal_spacing = in.real(); sample.right_normal_spacing = in.real();
  sample.face_count = in.u64(); sample.component_count = in.i32();
  sample.right_component_for_left.resize(in.size(checkpoint_detail::kEncodedScalarBytes));
  for (int& component : sample.right_component_for_left) component = in.i32();
  if (sample.route_contract.size() != 64 ||
      !std::all_of(sample.route_contract.begin(), sample.route_contract.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
      }))
    throw std::invalid_argument("retained shared route certificate is not canonical SHA256");
  auto& point = sample.source_point;
  point.clock = in.string();
  point.tick = in.i64();
  point.level = in.i32();
  point.substep = in.i32();
  point.stage = in.i32();
  point.stage_fraction = {in.i64(), in.i64()};
  point.dt = in.real();
  point.physical_time = in.real();
  point.graph_identity = in.string();
  point.rate_identity = in.string();
  point.application_identity = in.string();
  sample.flux_density.resize(in.size(checkpoint_detail::kEncodedScalarBytes));
  for (Real& value : sample.flux_density)
    value = static_cast<Real>(in.real());
  if ((sample.left_block != endpoint && sample.right_block != endpoint) ||
      point.graph_identity != facade_->installed_program_hash())
    throw std::invalid_argument("restored shared flux sample has a foreign endpoint or Program");
  facade_->authenticate_prepared_amr_interface_sample(sample);
  return std::make_shared<const shared_flux_sample_type>(std::move(sample));
}
