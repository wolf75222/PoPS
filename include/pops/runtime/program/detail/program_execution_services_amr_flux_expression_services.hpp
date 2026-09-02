void require_facade_execution_() const {
  if (facade_ == nullptr)
    throw std::logic_error("AMR Program execution requires its exact-ranked facade");
}
void require_prepared_lane_(const ExecutionLane& lane, std::string_view operation) const {
  const ExecutionLane& prepared = prepared_execution_lane();
  const long invalid = !lane.active() || !prepared.active() || lane.identity().empty() ||
                               lane.identity() != prepared.identity() ||
                               !lane.congruent_with(prepared)
                           ? 1L
                           : 0L;
  if (all_reduce_max(invalid, prepared) != 0)
    throw std::invalid_argument(std::string(operation) +
                                " requires the context's authenticated execution lane");
}

// Every mutation that couples numeric history, provenance, and temporal metadata stages locally
// first.  The exact prepared lane reports local preparation failures before any rank enters a
// later publication collective, then authenticates one operation contract for the candidates.
template <class Build, class Contract>
auto prepare_history_mutation_collectively_(Build&& build, Contract&& contract,
                                            std::string_view operation) const
    -> std::invoke_result_t<Build> {
  using result_type = std::invoke_result_t<Build>;
  const ExecutionLane& lane = prepared_execution_lane();
  require_prepared_lane_(lane, operation);
  std::optional<result_type> result;
  std::exception_ptr local_error;
  try {
    result.emplace(std::forward<Build>(build)());
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error(std::string(operation) + " preparation failed collectively");
  }
  const std::string_view exact_contract = std::invoke(contract, *result);
  if (!all_ranks_agree_exact_ordered_byte_pairs({{operation, exact_contract}}, lane))
    throw std::runtime_error(std::string(operation) +
                             " contract differs between communicator ranks");
  return std::move(*result);
}
void require_block_boundary_session_(const runtime::multiblock::BoundaryEvaluationPoint& point,
                                     int program_block, const block_boundary_session_type& boundary,
                                     std::string_view operation) const {
  require_boundary_point_(point, operation);
  const ExecutionLane& lane = prepared_execution_lane();
  if (boundary.facade_ != facade_ || boundary.runtime_block_ != sys_block(program_block) ||
      boundary.point_ != point || boundary.lane_ != &lane || !boundary.transport_)
    throw std::invalid_argument(std::string(operation) +
                                " received a foreign or stale prepared boundary session");
}
static void require_rate_identity_(int rate_id) {
  if (rate_id < 0)
    throw std::invalid_argument("AMR Program rate identity must be non-negative");
}

static void erase_zero_terms_(ExactPolynomial& polynomial) {
  for (auto term = polynomial.begin(); term != polynomial.end();) {
    if (term->second.numerator == 0)
      term = polynomial.erase(term);
    else
      ++term;
  }
}

static ExactPolynomial multiply_exact_polynomials_(const ExactPolynomial& left,
                                                   const ExactPolynomial& right) {
  ExactPolynomial result;
  for (const auto& [left_power, left_factor] : left)
    for (const auto& [right_power, right_factor] : right) {
      if (left_power > std::numeric_limits<int>::max() - right_power)
        throw std::overflow_error("AMR Program flux coefficient dt power exceeds int");
      const int power = left_power + right_power;
      const auto found = result.find(power);
      const ::pops::amr::Rational product = left_factor * right_factor;
      if (found == result.end())
        result.emplace(power, product);
      else
        found->second = found->second + product;
    }
  erase_zero_terms_(result);
  return result;
}

static void add_exact_polynomial_(ExactPolynomial& destination, const ExactPolynomial& source) {
  for (const auto& [power, factor] : source) {
    const auto found = destination.find(power);
    if (found == destination.end())
      destination.emplace(power, factor);
    else
      found->second = found->second + factor;
  }
  erase_zero_terms_(destination);
}

static Real evaluate_exact_polynomial_(const ExactPolynomial& polynomial, Real dt) {
  Real result = Real(0);
  for (const auto& [power, factor] : polynomial) {
    Real dt_power = Real(1);
    for (int exponent = 0; exponent < power; ++exponent)
      dt_power *= dt;
    result += static_cast<Real>(factor.value()) * dt_power;
  }
  return result;
}

ExactPolynomial exact_coefficient_unchecked_(
    Real factor, Real reference_dt, std::initializer_list<ExactCoefficientTerm> terms) const {
  if (!std::isfinite(static_cast<double>(factor)) ||
      !std::isfinite(static_cast<double>(reference_dt)) || reference_dt != current_dt_)
    throw std::invalid_argument("AMR Program flux coefficient does not name the active exact dt");
  ExactPolynomial polynomial;
  for (const ExactCoefficientTerm& term : terms) {
    if (term.dt_power < 0 || term.denominator <= 0)
      throw std::invalid_argument("AMR Program flux coefficient metadata is invalid");
    const ::pops::amr::Rational coefficient{term.numerator, term.denominator};
    if (coefficient.numerator != term.numerator || coefficient.denominator != term.denominator)
      throw std::invalid_argument("AMR Program flux coefficient metadata is not canonical");
    const auto found = polynomial.find(term.dt_power);
    if (found == polynomial.end())
      polynomial.emplace(term.dt_power, coefficient);
    else
      found->second = found->second + coefficient;
  }
  erase_zero_terms_(polynomial);
  if (evaluate_exact_polynomial_(polynomial, reference_dt) != factor)
    throw std::invalid_argument(
        "AMR Program numerical coefficient differs from its exact metadata");
  return polynomial;
}

static ::pops::amr::Rational exact_binary_rational_(Real value) {
  if (!std::isfinite(static_cast<double>(value)))
    throw std::invalid_argument("AMR Program coefficient is not finite");
  if (value == Real(0))
    return {0, 1};
  int exponent = 0;
  const double fraction = std::frexp(std::abs(static_cast<double>(value)), &exponent);
  constexpr int digits = std::numeric_limits<double>::digits;
  const auto mantissa = static_cast<std::uint64_t>(std::ldexp(fraction, digits));
  const int binary_power = exponent - digits;
  if (binary_power >= 0) {
    if (binary_power >= 63 ||
        mantissa >
            (static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) >> binary_power))
      throw std::overflow_error("AMR Program coefficient exceeds exact int64 metadata");
    const std::int64_t numerator = static_cast<std::int64_t>(mantissa << binary_power);
    return {value < Real(0) ? -numerator : numerator, 1};
  }
  if (-binary_power >= 63)
    throw std::overflow_error("AMR Program coefficient denominator exceeds exact int64 metadata");
  const std::int64_t numerator = static_cast<std::int64_t>(mantissa);
  const std::int64_t denominator = std::int64_t{1} << (-binary_power);
  return {value < Real(0) ? -numerator : numerator, denominator};
}

ExactPolynomial exact_runtime_coefficient_unchecked_(Real factor) const {
  if (active_attempt_states_.empty() || static_flux_tables_.bound)
    return {};
  return {{0, exact_binary_rational_(factor)}};
}

template <class Build>
auto prepare_flux_metadata_collectively_(Build&& build, std::string_view failure) const
    -> std::invoke_result_t<Build> {
  using result_type = std::invoke_result_t<Build>;
  if (active_attempt_states_.empty() || static_flux_tables_.bound)
    return std::forward<Build>(build)();
  std::optional<result_type> result;
  std::exception_ptr local_error;
  try {
    result.emplace(std::forward<Build>(build)());
  } catch (...) {
    local_error = std::current_exception();
  }
  const auto& lane = facade_->prepared_amr_multiblock_hierarchy_().lane();
  if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
    if (lane.size() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error(std::string(failure));
  }
  return std::move(*result);
}

ExactPolynomial exact_coefficient_(Real factor, Real reference_dt,
                                   std::initializer_list<ExactCoefficientTerm> terms) const {
  return prepare_flux_metadata_collectively_(
      [&] { return exact_coefficient_unchecked_(factor, reference_dt, terms); },
      "AMR Program exact flux coefficient failed collectively");
}

ExactPolynomial exact_runtime_coefficient_(Real factor) const {
  return prepare_flux_metadata_collectively_(
      [&] { return exact_runtime_coefficient_unchecked_(factor); },
      "AMR Program runtime flux coefficient failed collectively");
}

ExactPolynomial exact_runtime_axpy_coefficient_(Real factor, const field_type& source) const {
  return prepare_flux_metadata_collectively_(
      [&] {
        if (active_attempt_states_.empty() || active_flux_expression_(source).empty())
          return ExactPolynomial{};
        if (!std::isfinite(current_dt_) || !(current_dt_ > 0.0))
          throw std::logic_error("AMR Program flux axpy lacks its active dt");
        const Real quotient = factor / static_cast<Real>(current_dt_);
        if (quotient * static_cast<Real>(current_dt_) != factor)
          throw std::invalid_argument(
              "AMR Program flux axpy factor has no exact symbolic dt coefficient");
        return ExactPolynomial{{1, exact_binary_rational_(quotient)}};
      },
      "AMR Program runtime flux axpy coefficient failed collectively");
}

FluxExpression active_flux_expression_(const field_type& field) const {
  if (active_attempt_states_.empty() || static_flux_tables_.bound)
    return {};
  const auto found = active_flux_expressions_.find(&field);
  return found == active_flux_expressions_.end() ? FluxExpression{} : found->second;
}

/// Return the last issued occurrence identity across every live, resident and retained
/// expression authority.  `next_active_flux_basis_identity_` is reset for each level group, but
/// retained history survives that boundary.  A rehydrated lag occurrence therefore cannot start
/// from the reset counter: it must be issued strictly after both the current sample and every
/// restored/static occurrence it can coexist with.  Map keys are the finite occurrence authority;
/// pointer addresses and basis payload order are deliberately not identities here.
std::uint64_t flux_basis_identity_floor_(const FluxExpressionRegistry& current,
                                         std::uint64_t identity) const {
  const auto observe_expression = [&](const FluxExpression& expression) {
    for (const auto& [registered_identity, term] : expression) {
      if (!term.basis)
        throw std::logic_error(
            "AMR Program flux expression has a non-canonical occurrence identity");
      // Every inactive static slot retains its finite map key to preserve the bind-sealed map
      // topology, while its payload carries the explicit inactive sentinel.  It has no live
      // occurrence to collide with; any other key/payload disagreement is malformed.
      if (term.basis->identity != registered_identity &&
          term.basis->identity != kStaticHistoryFluxInactiveIdentity)
        throw std::logic_error(
            "AMR Program flux expression has a non-canonical occurrence identity");
      identity = std::max(identity, registered_identity);
    }
  };
  const auto observe_registry = [&](const FluxExpressionRegistry& registry) {
    for (const auto& [field, expression] : registry) {
      (void)field;
      observe_expression(expression);
    }
  };

  observe_registry(current);
  observe_registry(active_flux_expressions_);
  for (const auto& [key, ring] : history_flux_expressions_) {
    (void)key;
    for (const FluxExpression& expression : ring)
      observe_expression(expression);
  }
  if (static_flux_tables_.bound) {
    if (static_flux_basis_active_.size() != static_flux_basis_payloads_.size())
      throw std::logic_error("AMR Program static flux identity carrier is incomplete");
    for (std::size_t slot = 0; slot < static_flux_basis_payloads_.size(); ++slot)
      if (static_flux_basis_active_[slot] != 0)
        identity = std::max(identity, static_flux_basis_payloads_[slot].identity);
  }
  return identity;
}

/// Rebind an immutable retained sample to this exact level-group.  The scientific face payload and
/// exact coefficient polynomial are the authenticated history sample; addresses, basis identity,
/// clock tick, substep and window are attempt-local and must never leak across the boundary.
std::pair<FluxExpression, std::uint64_t> rehydrated_history_flux_expression_(
    const std::string& key, int lag, const FluxExpressionRegistry& current,
    std::uint64_t identity) const {
  const auto ring = history_flux_expressions_.find(key);
  if (ring == history_flux_expressions_.end() || lag < 0 ||
      static_cast<std::size_t>(lag) >= ring->second.size())
    throw std::out_of_range("AMR Program history flux provenance slot is absent");
  const FluxExpression retained = ring->second[static_cast<std::size_t>(lag)];
  if (retained.empty())
    return {{}, identity};
  identity = flux_basis_identity_floor_(current, identity);
  FluxExpression rebound;
  const auto same_face_identity = [](const FluxBasisFace& left, const FluxBasisFace& right) {
    return left.role == right.role && left.axis == right.axis && left.face == right.face &&
           left.coarse_face == right.coarse_face && left.face_measure == right.face_measure;
  };
  for (const auto& [stored_identity, term] : retained) {
    (void)stored_identity;
    if (!term.basis)
      throw std::logic_error("AMR Program history flux provenance has no basis");
    if (identity == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("AMR Program history flux basis identity overflow");
    FluxBasis basis = *term.basis;
    basis.identity = ++identity;
    basis.level = active_level_;
    basis.point.level = active_level_;
    basis.point.tick = active_subcycling_window_.begin.macro_step;
    basis.point.substep = logical_substep_;
    basis.point.dt = current_dt_;
    basis.window = active_subcycling_window_;
    basis.point.physical_time =
        basis.window.begin.physical_time + basis.point.stage_fraction.value() * current_dt_;
    // A retained sample can outlive an accepted regrid.  Its numerical history is prolonged by
    // the native rematerialization transaction, while a new or moved interface receives an exact
    // zero-mismatch stage.  Unchanged faces retain their authenticated density byte-for-byte.
    const std::vector<FluxBasisFace>* current_faces = nullptr;
    for (const auto& [field, expression] : current) {
      (void)field;
      for (const auto& [active_identity, active_term] : expression) {
        (void)active_identity;
        if (!active_term.basis || active_term.basis->runtime_block != basis.runtime_block ||
            active_term.basis->level != basis.level ||
            active_term.basis->rhs_identity != basis.rhs_identity ||
            active_term.basis->provider != basis.provider)
          continue;
        current_faces = &active_term.basis->faces;
      }
    }
    if (current_faces != nullptr) {
      std::vector<FluxBasisFace> rebound_faces;
      rebound_faces.reserve(current_faces->size());
      for (const FluxBasisFace& current_face : *current_faces) {
        const auto historical = std::find_if(
            basis.faces.begin(), basis.faces.end(), [&](const FluxBasisFace& retained_face) {
              return same_face_identity(retained_face, current_face);
            });
        if (historical != basis.faces.end())
          rebound_faces.push_back(*historical);
        else {
          FluxBasisFace zero = current_face;
          std::fill(zero.flux_density.begin(), zero.flux_density.end(), Real(0));
          rebound_faces.push_back(std::move(zero));
        }
      }
      basis.faces = std::move(rebound_faces);
    }
    rebound.emplace(
        basis.identity,
        FluxExpressionTerm{std::make_shared<const FluxBasis>(std::move(basis)), term.coefficient});
  }
  require_flux_expression_budget_(rebound);
  return {std::move(rebound), identity};
}

void rehydrate_history_flux_expression_(const std::string& key, int lag,
                                        const field_type& destination) const {
  if (active_attempt_states_.empty())
    return;
  std::pair<FluxExpressionRegistry, std::uint64_t> prepared = prepare_flux_metadata_collectively_(
      [&] {
        FluxExpressionRegistry candidate = active_flux_expressions_;
        auto [rebound, identity] = rehydrated_history_flux_expression_(
            key, lag, active_flux_expressions_, next_active_flux_basis_identity_);
        if (!rebound.empty())
          candidate[&destination] = std::move(rebound);
        return std::pair<FluxExpressionRegistry, std::uint64_t>{std::move(candidate), identity};
      },
      "AMR Program history flux-expression rehydration failed collectively");
  static_assert(std::is_nothrow_swappable_v<FluxExpressionRegistry>);
  active_flux_expressions_.swap(prepared.first);
  next_active_flux_basis_identity_ = prepared.second;
}

template <class Writer>
static bool write_history_flux_payload_(
    Writer& out,
    const std::map<std::string, std::vector<FluxExpression>>& history_flux_expressions) {
  const auto active_terms = [](const FluxExpression& expression) {
    return std::count_if(expression.begin(), expression.end(), [](const auto& entry) {
      const auto& [identity, term] = entry;
      return term.basis && term.basis->identity == identity;
    });
  };
  const bool any_expression = std::any_of(
      history_flux_expressions.begin(), history_flux_expressions.end(), [](const auto& entry) {
        return std::any_of(
            entry.second.begin(), entry.second.end(), [](const FluxExpression& expression) {
              return std::any_of(expression.begin(), expression.end(), [](const auto& term) {
                return term.second.basis && term.second.basis->identity == term.first;
              });
            });
      });
  if (!any_expression)
    return false;
  out.size(history_flux_expressions.size());
  for (const auto& [key, slots] : history_flux_expressions) {
    out.string(key);
    out.size(slots.size());
    for (const FluxExpression& expression : slots) {
      out.size(static_cast<std::size_t>(active_terms(expression)));
      for (const auto& [identity, term] : expression) {
        if (!term.basis)
          throw std::logic_error("AMR Program history flux payload has an unauthenticated basis");
        // Cold-bound static images retain their complete map topology in every history slot.  An
        // inactive basis deliberately has no sample for this slot and is omitted from the wire;
        // the map node itself stays resident so the next store cannot allocate it.
        if (term.basis->identity != identity) {
          if (term.basis->identity != kStaticHistoryFluxInactiveIdentity)
            throw std::logic_error("AMR Program history flux payload has an invalid basis state");
          continue;
        }
        out.u64(identity);
        out.size(term.coefficient.size());
        for (const auto& [power, coefficient] : term.coefficient) {
          out.i32(power);
          out.i64(coefficient.numerator);
          out.i64(coefficient.denominator);
        }
        const FluxBasis& basis = *term.basis;
        out.u64(basis.runtime_block);
        out.i32(basis.level);
        out.i32(basis.rhs_identity);
        out.u64(static_cast<std::uint64_t>(basis.provider));
        const auto write_point = [&](const auto& point) {
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
        };
        write_point(basis.point);
        checkpoint_detail::write_clock(out, basis.window.begin);
        checkpoint_detail::write_clock(out, basis.window.end);
        out.size(basis.faces.size());
        for (const FluxBasisFace& face : basis.faces) {
          out.u64(static_cast<std::uint64_t>(face.role));
          out.i32(face.axis);
          for (int axis = 0; axis < Dim; ++axis) {
            out.i32(face.face[axis]);
            out.i32(face.coarse_face[axis]);
          }
          out.real(face.face_measure);
          out.size(face.flux_density.size());
          for (const Real value : face.flux_density)
            out.real(static_cast<double>(value));
        }
      }
    }
  }
  return true;
}

static std::vector<std::uint8_t> serialize_history_flux_payload_(
    const std::map<std::string, std::vector<FluxExpression>>& history_flux_expressions) {
  checkpoint_detail::Writer out;
  if (!write_history_flux_payload_(out, history_flux_expressions))
    return {};
  return std::move(out).take();
}

/// Serialize the already bind-authenticated history provenance into its resident checkpoint
/// payload arena.  The public convenience serializer above remains intentionally allocating for
/// restart/cold paths; accepted-step refreshes use this companion after their complete shape has
/// been cold-primed.
static void serialize_history_flux_payload_into_(
    const std::map<std::string, std::vector<FluxExpression>>& history_flux_expressions,
    std::vector<std::uint8_t>& bytes) {
  checkpoint_detail::CountingWriter count;
  if (!write_history_flux_payload_(count, history_flux_expressions)) {
    bytes.clear();
    return;
  }
  if (count.count() > bytes.capacity())
    throw std::length_error("AMR Program history flux payload arena was not primed");
  bytes.resize(count.count());
  checkpoint_detail::PreallocatedWriter out(std::span<std::uint8_t>(bytes.data(), bytes.size()));
  (void)write_history_flux_payload_(out, history_flux_expressions);
  out.require_complete();
}

std::vector<std::uint8_t> serialize_history_flux_payload_() const {
  return serialize_history_flux_payload_(history_flux_expressions_);
}

void serialize_history_flux_payload_into_(std::vector<std::uint8_t>& bytes) const {
  serialize_history_flux_payload_into_(history_flux_expressions_, bytes);
}

std::map<std::string, std::vector<FluxExpression>> prepare_history_flux_payload_restore_(
    std::span<const std::uint8_t> bytes) const {
  if (bytes.empty()) {
    for (const auto& [key, ring] : runtime_state().hist_.histories) {
      (void)ring;
      const auto owner = runtime_state().hist_.owner.find(key);
      if (owner == runtime_state().hist_.owner.end() || owner->second < 0 ||
          static_cast<std::size_t>(owner->second) >= prepared_rhs_basis_bounds_.size())
        throw std::invalid_argument("AMR Program history flux payload has a foreign runtime owner");
      if (runtime_state().hist_.initialized.at(key) &&
          prepared_rhs_basis_bounds_[static_cast<std::size_t>(owner->second)] != 0)
        throw std::invalid_argument(
            "AMR Program checkpoint omits initialized history-flux provenance");
    }
    std::map<std::string, std::vector<FluxExpression>> empty;
    for (const auto& [key, ring] : runtime_state().hist_.histories)
      empty.emplace(key, std::vector<FluxExpression>(ring.size()));
    return empty;
  }
  checkpoint_detail::Reader in(bytes);
  constexpr std::size_t coefficient_record_bytes = 3 * checkpoint_detail::kEncodedScalarBytes;
  constexpr std::size_t face_record_bytes =
      (5 + 2 * static_cast<std::size_t>(Dim)) * checkpoint_detail::kEncodedScalarBytes;
  constexpr std::size_t basis_term_bytes =
      (28 + 2 * static_cast<std::size_t>(Dim)) * checkpoint_detail::kEncodedScalarBytes;
  std::map<std::string, std::vector<FluxExpression>> candidate;
  const std::size_t history_count = in.size(2 * checkpoint_detail::kEncodedScalarBytes);
  for (std::size_t history = 0; history < history_count; ++history) {
    const std::string key = in.string();
    const auto numeric = runtime_state().hist_.histories.find(key);
    if (key.empty() || numeric == runtime_state().hist_.histories.end())
      throw std::invalid_argument("AMR Program history flux payload names a foreign ring");
    const std::size_t slot_count = in.size(checkpoint_detail::kEncodedScalarBytes);
    if (slot_count != numeric->second.size())
      throw std::invalid_argument("AMR Program history flux payload differs from ring depth");
    std::vector<FluxExpression> slots(slot_count);
    for (std::size_t slot = 0; slot < slot_count; ++slot) {
      const std::size_t term_count = in.size(basis_term_bytes);
      FluxExpression expression;
      for (std::size_t index = 0; index < term_count; ++index) {
        const std::uint64_t identity = in.u64();
        ExactPolynomial coefficient;
        const std::size_t coefficient_count = in.size(coefficient_record_bytes);
        for (std::size_t term_index = 0; term_index < coefficient_count; ++term_index) {
          const int power = in.i32();
          const std::int64_t numerator = in.i64();
          const std::int64_t denominator = in.i64();
          const ::pops::amr::Rational rational{numerator, denominator};
          if (power < 0 || rational.denominator <= 0 ||
              ::pops::amr::Rational{rational.numerator, rational.denominator} != rational ||
              !coefficient.emplace(power, rational).second)
            throw std::invalid_argument("AMR Program history flux payload has invalid coefficient");
        }
        if (coefficient.empty())
          throw std::invalid_argument("AMR Program history flux payload has an empty coefficient");
        FluxBasis basis;
        basis.identity = identity;
        basis.runtime_block = in.u64();
        basis.level = in.i32();
        basis.rhs_identity = in.i32();
        const std::uint64_t provider = in.u64();
        if (provider > static_cast<std::uint64_t>(FluxBasisProvider::NamedCell))
          throw std::invalid_argument("AMR Program history flux payload has an invalid provider");
        basis.provider = static_cast<FluxBasisProvider>(provider);
        auto read_point = [&] {
          runtime::multiblock::BoundaryEvaluationPoint point;
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
          if (point.clock.empty() || point.stage < 0 || point.stage_fraction.denominator <= 0 ||
              ::pops::amr::Rational{point.stage_fraction.numerator,
                                    point.stage_fraction.denominator} != point.stage_fraction ||
              !std::isfinite(point.dt) || !(point.dt > 0.0) || !std::isfinite(point.physical_time))
            throw std::invalid_argument("AMR Program history flux payload has an invalid point");
          return point;
        };
        basis.point = read_point();
        basis.window.begin = checkpoint_detail::read_clock(in);
        basis.window.end = checkpoint_detail::read_clock(in);
        if (!(basis.window.begin.phase < basis.window.end.phase) ||
            !(basis.window.end.physical_time > basis.window.begin.physical_time))
          throw std::invalid_argument("AMR Program history flux payload has an invalid window");
        const std::size_t face_count = in.size(face_record_bytes);
        basis.faces.reserve(face_count);
        for (std::size_t face_index = 0; face_index < face_count; ++face_index) {
          const std::uint64_t role = in.u64();
          FluxBasisFace face;
          if (role > static_cast<std::uint64_t>(::pops::amr::reflux::FaceLedgerRole::Fine))
            throw std::invalid_argument(
                "AMR Program history flux payload has an invalid face role");
          face.role = static_cast<::pops::amr::reflux::FaceLedgerRole>(role);
          face.axis = in.i32();
          if (face.axis < 0 || face.axis >= Dim)
            throw std::invalid_argument(
                "AMR Program history flux payload has an invalid face axis");
          for (int axis = 0; axis < Dim; ++axis) {
            face.face[axis] = in.i32();
            face.coarse_face[axis] = in.i32();
          }
          face.face_measure = in.real();
          const std::size_t density_count = in.size(sizeof(double));
          if (!(face.face_measure > 0.0) || !std::isfinite(face.face_measure) || density_count == 0)
            throw std::invalid_argument("AMR Program history flux payload has an invalid face");
          face.flux_density.resize(density_count);
          for (Real& value : face.flux_density) {
            value = static_cast<Real>(in.real());
            if (!std::isfinite(static_cast<double>(value)))
              throw std::invalid_argument(
                  "AMR Program history flux payload has a non-finite density");
          }
          basis.faces.push_back(std::move(face));
        }
        // The checkpoint payload contains only the logical face prefix.  Preserve that boundary
        // before the cold successor normalizer expands it into its bind-sealed route slots;
        // treating vector capacity as logical faces would serialize stale template tails.
        basis.face_count = face_count;
        // A completely covered level legitimately has no coarse--fine faces.  Archive bytes alone
        // cannot establish that an empty vector is meaningful, so admit it only when the rebuilt,
        // authenticated hierarchy proves that neither adjacent transition exposes an interface for
        // this basis level.  An empty vector at any live nonempty/ambiguous transition remains a
        // malformed provenance payload rather than an implicit zero stage.
        const auto empty_faces_authorized_by_live_topology = [&] {
          if (basis.level < 0 || static_cast<std::size_t>(basis.level) >= nlev())
            return false;
          const std::size_t level = static_cast<std::size_t>(basis.level);
          if (level > 0 && !program_interface_faces_(level - 1).empty())
            return false;
          if (level + 1 < nlev() && !program_interface_faces_(level).empty())
            return false;
          return true;
        };
        if (basis.runtime_block >= prepared_rhs_basis_bounds_.size() ||
            (basis.faces.empty() && !empty_faces_authorized_by_live_topology()) ||
            !expression
                 .emplace(identity,
                          FluxExpressionTerm{std::make_shared<const FluxBasis>(std::move(basis)),
                                             std::move(coefficient)})
                 .second)
          throw std::invalid_argument("AMR Program history flux payload has a duplicate basis");
      }
      require_flux_expression_budget_(expression);
      slots[slot] = std::move(expression);
    }
    if (!candidate.emplace(key, std::move(slots)).second)
      throw std::invalid_argument("AMR Program history flux payload repeats a ring");
  }
  in.finish();
  if (candidate.size() != runtime_state().hist_.histories.size())
    throw std::invalid_argument("AMR Program history flux payload omits a live ring");
  if (static_flux_tables_.bound) {
    // Restore is cold, but the following accepted snapshot is not: rebuild each restored logical
    // prefix into the exact successor slots carried by the static table. This retains densities
    // only for complete geometric identities and leaves newly exposed faces at their prepared
    // zero value, so subsequent capture observes the same sealed shape as the live carrier.
    candidate = prepare_static_history_flux_provenance_from_sealed_history_(
        static_flux_tables_, static_flux_basis_payloads_, runtime_state().hist_.owner,
        history_levels_, candidate, primary_clock_);
  }
  return candidate;
}

void require_flux_expression_budget_(const FluxExpression& expression) const {
  std::map<std::size_t, std::size_t> bases_by_block;
  for (const auto& [identity, term] : expression) {
    (void)identity;
    if (!term.basis || term.basis->runtime_block >= prepared_rhs_basis_bounds_.size() ||
        term.basis->runtime_block >= prepared_coefficient_term_bounds_.size())
      throw std::logic_error("AMR Program flux expression has a foreign basis identity");
    const std::size_t block = term.basis->runtime_block;
    if (++bases_by_block[block] > prepared_rhs_basis_bounds_[block] ||
        term.coefficient.size() > prepared_coefficient_term_bounds_[block])
      throw std::length_error(
          "AMR Program flux expression exceeds its authenticated artifact budget");
  }
}

static FluxExpression scaled_flux_expression_(const FluxExpression& expression,
                                              const ExactPolynomial& coefficient) {
  FluxExpression result;
  if (coefficient.empty())
    return result;
  for (const auto& [identity, term] : expression) {
    ExactPolynomial scaled = multiply_exact_polynomials_(term.coefficient, coefficient);
    if (!scaled.empty())
      result.emplace(identity, FluxExpressionTerm{term.basis, std::move(scaled)});
  }
  return result;
}

static void add_flux_expression_(FluxExpression& destination, const FluxExpression& source) {
  for (const auto& [identity, term] : source) {
    const auto found = destination.find(identity);
    if (found == destination.end()) {
      destination.emplace(identity, term);
      continue;
    }
    if (found->second.basis != term.basis)
      throw std::logic_error("AMR Program flux basis identity aliases another evaluation");
    add_exact_polynomial_(found->second.coefficient, term.coefficient);
    if (found->second.coefficient.empty())
      destination.erase(found);
  }
}

FluxExpressionUpdate prepare_active_axpy_flux_expression_(
    field_type& destination, const field_type& source, const ExactPolynomial& coefficient) const {
  if (active_attempt_states_.empty() || static_flux_tables_.bound)
    return std::nullopt;
  return prepare_flux_metadata_collectively_(
      [&]() {
        FluxExpressionRegistry candidate = active_flux_expressions_;
        FluxExpression result = active_flux_expression_(destination);
        add_flux_expression_(result,
                             scaled_flux_expression_(active_flux_expression_(source), coefficient));
        require_flux_expression_budget_(result);
        candidate[&destination] = std::move(result);
        return FluxExpressionUpdate(std::move(candidate));
      },
      "AMR Program flux-expression axpy failed collectively");
}

FluxExpressionUpdate prepare_active_lincomb_flux_expression_(
    field_type& destination, const field_type& left, const ExactPolynomial& left_coefficient,
    const field_type& right, const ExactPolynomial& right_coefficient) const {
  if (active_attempt_states_.empty() || static_flux_tables_.bound)
    return std::nullopt;
  return prepare_flux_metadata_collectively_(
      [&]() {
        FluxExpressionRegistry candidate = active_flux_expressions_;
        FluxExpression result =
            scaled_flux_expression_(active_flux_expression_(left), left_coefficient);
        add_flux_expression_(
            result, scaled_flux_expression_(active_flux_expression_(right), right_coefficient));
        require_flux_expression_budget_(result);
        candidate[&destination] = std::move(result);
        return FluxExpressionUpdate(std::move(candidate));
      },
      "AMR Program flux-expression linear combination failed collectively");
}

void publish_active_flux_expression_update_(FluxExpressionUpdate update) const noexcept {
  if (update)
    active_flux_expressions_.swap(*update);
}

void copy_active_flux_expression_(const field_type& source, field_type& destination) const {
  if (active_attempt_states_.empty() || static_flux_tables_.bound)
    return;
  FluxExpressionRegistry candidate = prepare_flux_metadata_collectively_(
      [&] {
        FluxExpressionRegistry result = active_flux_expressions_;
        result[&destination] = active_flux_expression_(source);
        return result;
      },
      "AMR Program flux-expression copy failed collectively");
  active_flux_expressions_.swap(candidate);
}
