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
  if (active_attempt_states_.empty())
    return {};
  return {{0, exact_binary_rational_(factor)}};
}

template <class Build>
auto prepare_flux_metadata_collectively_(Build&& build, std::string_view failure) const
    -> std::invoke_result_t<Build> {
  using result_type = std::invoke_result_t<Build>;
  if (active_attempt_states_.empty())
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
  if (active_attempt_states_.empty())
    return {};
  const auto found = active_flux_expressions_.find(&field);
  return found == active_flux_expressions_.end() ? FluxExpression{} : found->second;
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
  if (active_attempt_states_.empty())
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
  if (active_attempt_states_.empty())
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
  if (active_attempt_states_.empty())
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
