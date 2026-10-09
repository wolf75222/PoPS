// Generic simultaneous-candidate producer exercised by the actual Newton/GMRES core.
// This is a native mechanism witness, not a continuum PDE or M26 certificate.
TEST(CompositeGeneralField, OriginalInteractionReevaluatesFullCandidateForEveryResidual) {
  auto lane = ExecutionLane::duplicate_world_collectively("original-interaction-full-F");
  constexpr int cells = 4;
  auto provider = original_field_provider(lane, cells, {0, 1, 2});
  const auto geometry = three_field_request(lane, cells);
  auto exact = original_tower(*provider), forcing = original_tower(*provider);
  for (int level = 0; level < provider->level_count(); ++level) {
    const auto mesh = geometry.levels.at(level).geometry;
    for (std::size_t patch = 0; patch < exact[level].local_size(); ++patch) {
      const auto q = exact[level].fab(patch).view();
      for_each_cell(exact[level].box(patch), [=] POPS_HD(const Index<1>& cell) {
        const Real x = mesh.lower()[0] + (Real(cell[0] - mesh.domain().lo[0]) + Real(.5)) * mesh.spacing(0);
        q(cell, 0) = Real(.2) + Real(.03) * x;
        q(cell, 1) = Real(-.1) + Real(.02) * x;
        q(cell, 2) = Real(.15) - Real(.01) * x;
      });
    }
  }
  provider->synchronize_original_field_candidate(exact);
  provider->prepare_original_field_operator();
  provider->apply_original_field_operator(exact, forcing);
  Solver::hierarchy_type interaction;
  std::size_t producer_calls = 0, issued_calls = 0;
  std::shared_ptr<OriginalResidual> issued_core;
  const AmrFieldResidualAuthority* issued_authority = nullptr;
  const Solver::hierarchy_type* last_evaluation = nullptr;
  std::uint64_t last_ordinal = 0;
  std::uint64_t previous_nonce = 0;
  auto produce = [&](const Solver::hierarchy_type& q, std::uint64_t evaluation) {
    if (issued_core) {
      issued_core->require_evaluation(q, *issued_authority, provider.get(), evaluation, lane);
      EXPECT_GT(evaluation, previous_nonce);
      if (previous_nonce)
        EXPECT_THROW(issued_core->require_evaluation(q, *issued_authority, provider.get(), previous_nonce, lane),
                     std::exception); // Same reused q pointer does not reissue an old F nonce.
      previous_nonce = evaluation;
      auto foreign = original_tower(*provider);
      EXPECT_THROW(issued_core->require_evaluation(foreign, *issued_authority, provider.get(), evaluation, lane),
                   std::exception);
      last_evaluation = &q;
      last_ordinal = evaluation;
    }
    std::vector<InteractionLevelView<1, typename Solver::field_type::memory_space>> views;
    interaction_phase(lane, [&] {
      for (int level = 0; level < provider->level_count(); ++level)
        views.push_back({&q.at(level), &provider->original_field_active_cells(level), nullptr, nullptr,
                         geometry.levels.at(level).geometry});
    });
    Solver::hierarchy_type detached;
    interaction_phase(lane, [&] { detached.reserve(views.size()); });
    for (std::size_t level = 0; level < views.size(); ++level) {
      auto output = direct_spatial_interaction<1, typename Solver::field_type::memory_space>(views,
          level, std::array<int, 3>{2, 0, 1}, 8*1024*1024, "witness.original-W-source-order-201@1", lane,
          [] POPS_HD(const auto& x, const auto& y) { return Real(1) + Real(.2)*x[0] - Real(.3)*y[0]; });
      interaction_phase(lane, [&] { detached.push_back(std::move(output)); });
    }
    interaction.swap(detached);
    ++producer_calls;
    if (evaluation != 0) ++issued_calls;
  };
  auto local = [&](const auto& q, const auto& captures, auto& result, int evaluation) {
    original_local_body({0, 1, 2})(q, captures, result, evaluation);
    for (std::size_t level = 0; level < result.size(); ++level)
      for (std::size_t patch = 0; patch < result[level].local_size(); ++patch) {
        const auto integral = std::as_const(interaction[level]).fab(patch).view();
        const auto output = result[level].fab(patch).view();
        for_each_cell(result[level].box(patch), [=] POPS_HD(const Index<1>& cell) {
          for (int component = 0; component < 3; ++component)
            output(cell, component) += Real(.05) * integral(cell, component);
        });
      }
  };
  std::vector<Solver::hierarchy_type> zeros{original_tower(*provider)};
  for (auto& level : zeros[0]) level.set_val(Real(0));
  produce(exact, 0);
  local(exact, zeros, forcing, 0);  // Original discrete F, with all declared terms.
  PreparedResourceCache cache;
  auto authority = original_authority(provider.get(), cache.begin_attempt(lane));
  const std::array<const Solver::hierarchy_type*, 1> captures{&forcing};
  auto core = OriginalResidual::prepare(provider, authority, captures, original_controls(), Real(1e-5), lane,
      AmrFieldRightPreconditioner::kFullResidualBasisLU, AmrFieldCoefficientEvaluation::kFrozen,
      std::uint64_t{64*1024*1024});
  issued_core = core;
  issued_authority = &authority;
  producer_calls = issued_calls = 0;
  for (int level = 0; level < provider->level_count(); ++level) provider->solution(level).set_val(Real(9));
  auto report = core->solve_interaction(authority, nullptr, local, produce, lane);
  ASSERT_TRUE(report.solved()) << report.reason;
  const auto& q = core->candidate(authority, lane);
  EXPECT_EQ(producer_calls, core->residual_evaluations());
  EXPECT_GT(issued_calls, std::size_t(0));
  EXPECT_GT(core->derivative_evaluations(), 0);
  ASSERT_NE(last_evaluation, nullptr);
  EXPECT_THROW(core->require_evaluation(*last_evaluation, authority, provider.get(), last_ordinal, lane),
               std::exception); // Lease ended before local F consumption.
  for (int level = 0; level < provider->level_count(); ++level) {
    EXPECT_EQ(norm_inf(provider->solution(level), 0), Real(9)); // No publication before Outcome.
    for (std::size_t patch = 0; patch < q[level].local_size(); ++patch) {
      const auto actual = q[level].fab(patch).view();
      const auto expected = std::as_const(exact[level]).fab(patch).view();
      const auto active = provider->original_field_active_cells(level).fab(patch).view();
      const Real error = for_each_cell_reduce_max(q[level].box(patch), [=] POPS_HD(const Index<1>& cell) {
        if (active(cell, 0) < Real(.5)) return Real(0);
        Real result = 0;
        for (int c = 0; c < 3; ++c) result = std::max(result, std::abs(actual(cell,c)-expected(cell,c)));
        return result;
      });
      EXPECT_LT(error, Real(2e-7));
    }
  }
  auto refuse = [&](const auto& candidate, int) {
    std::vector<InteractionLevelView<1, typename Solver::field_type::memory_space>> views;
    interaction_phase(lane, [&] {
      for (int level = 0; level < provider->level_count(); ++level)
        views.push_back({&candidate.at(level), &provider->original_field_active_cells(level), nullptr, nullptr,
                         geometry.levels.at(level).geometry});
    });
    (void)direct_spatial_interaction<1, typename Solver::field_type::memory_space>(views, 0,
        std::array<int, 1>{0}, 1, "witness.explicit-budget-refusal@1", lane,
        [] POPS_HD(const auto&, const auto&) { return Real(1); });
  };
  EXPECT_THROW(core->solve_interaction(authority, nullptr, local, refuse, lane), std::exception);
  EXPECT_THROW(core->candidate(authority, lane), std::exception);
  auto throwing = [&](const auto& candidate, std::uint64_t evaluation) {
    core->require_evaluation(candidate, authority, provider.get(), evaluation, lane);
    last_evaluation = &candidate; last_ordinal = evaluation;
    interaction_phase(lane, [&] {
      if (lane.rank() == 0) throw std::runtime_error("witness producer rank-local refusal");
    });
  };
  EXPECT_THROW(core->solve_interaction(authority, nullptr, local, throwing, lane), std::exception);
  EXPECT_THROW(core->require_evaluation(*last_evaluation, authority, provider.get(), last_ordinal, lane),
               std::exception); // RAII also revoked on callback refusal.
  EXPECT_THROW(core->candidate(authority, lane), std::exception);
  for (int level = 0; level < provider->level_count(); ++level)
    EXPECT_EQ(norm_inf(provider->solution(level), 0), Real(9));
}
