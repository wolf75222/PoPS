
struct FluxExpressionTerm {
  std::shared_ptr<const FluxBasis> basis;
  ExactPolynomial coefficient;
};

using FluxExpression = std::map<std::uint64_t, FluxExpressionTerm>;
using FluxExpressionRegistry = std::map<const field_type*, FluxExpression>;
using FluxExpressionUpdate = std::optional<FluxExpressionRegistry>;

struct LevelAttemptEnvelope {
  int active_level_ = 0;
  double current_dt_ = 0.0;
  double current_interval_start_time_ = 0.0;
  ::pops::amr::Rational current_interval_begin_phase_{0, 1};
  ::pops::amr::Rational current_interval_end_phase_{1, 1};
  int logical_substep_ = 0;
  ::pops::amr::Rational stage_time_{0, 1};
  std::vector<field_type*> active_attempt_states_;
  std::vector<const field_type*> active_staged_parents_;
  std::vector<multiblock_flux_ledger_type*> active_incoming_flux_;
  std::vector<multiblock_flux_ledger_type*> active_outgoing_flux_;
  std::vector<std::string_view> active_block_identities_;
  FluxExpressionRegistry active_flux_expressions_;
  std::vector<std::size_t> active_flux_basis_counts_;
  std::uint64_t next_active_flux_basis_identity_ = 0;
  ::pops::amr::ClockWindow active_subcycling_window_{};
  std::uint64_t active_subcycling_attempt_ = 0;
};
