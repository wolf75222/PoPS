// Native discrete manufactured witness, not an M14/M27 physical campaign.
using OriginalResidual = PreparedAmrFieldResidual<1>;

std::shared_ptr<Solver> original_field_provider(const ExecutionLane& lane, int cells,
                                                std::array<int, 3> permutation) {
  auto request = three_field_request(lane, cells);
  request.plan_identity = "witness.original-cubic-field.composite@1";
  request.operator_contract_identity = "witness.signed-spd-diffusion@1";
  auto& options = request.options.values;
  for (auto it = options.begin(); it != options.end();)
    if (it->first.starts_with("modes.") || it->first.starts_with("gauge."))
      it = options.erase(it);
    else
      ++it;
  options["modes.count"] = std::uint64_t{0};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      options["reaction." + std::to_string(i) + "." + std::to_string(j)] = 0.0;
  Provider provider;
  auto result =
      std::make_shared<Solver>(request, provider.expected_prepared_contract(request), lane);
  result->seal_preparation(lane);
  constexpr std::array<Real, 9> diffusion{2, -.35, .2, -.35, 1.6, -.25, .2, -.25, 1.3};
  std::array<Real, 9> matrix{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      matrix[i * 3 + j] = diffusion[permutation[i] * 3 + permutation[j]];
  for (int level = 0; level < 2; ++level) {
    auto& coefficients = result->assembly_target("pops.general-field.coefficients", level);
    for (std::size_t patch = 0; patch < coefficients.local_size(); ++patch) {
      const auto output = coefficients.fab(patch).view();
      for_each_cell(coefficients.box(patch), [=] POPS_HD(const Index<1>& cell) {
        for (int component = 0; component < 9; ++component)
          output(cell, component) = matrix[component];
      });
    }
  }
  return result;
}

Solver::hierarchy_type original_tower(const Solver& provider) {
  Solver::hierarchy_type result;
  for (int level = 0; level < provider.level_count(); ++level) {
    const auto& shape = provider.original_field_layout(level);
    result.emplace_back(shape.layout(), shape.distribution(), shape.local_rank(), 3, Extent<1>{});
  }
  return result;
}

AmrFieldResidualAuthority original_authority(const void* owner, PreparedResourceAttempt attempt) {
  AmrFieldResidualAuthority result;
  result.owner = owner;
  result.attempt = std::move(attempt);
  result.level_attempts = {result.attempt, result.attempt};
  result.topology_epoch = 1;
  result.materialization_generation = 1;
  result.original_equation_identity =
      "witness.diffusion-plus-cubic-plus-coupled-reaction-minus-forcing@1";
  result.capture_identities = {"witness.original-field.forcing:three-ordered-components@1"};
  result.capture_components = {3};
  for (int level = 0; level < 2; ++level)
    result.points.push_back({.clock = "macro",
                             .tick = 1,
                             .level = level,
                             .substep = 0,
                             .stage = 0,
                             .dt = .01,
                             .physical_time = .01,
                             .graph_identity = "witness.original-field-program@1",
                             .rate_identity = "witness.original-field-body@1",
                             .application_identity = "witness.original-field-solve@1"});
  return result;
}

FieldNewtonOptions original_controls() {
  return {.tolerance = Real(2e-9),
          .max_iterations = 12,
          .linear_tolerance = Real(1e-5),
          .linear_max_iterations = 240,
          .restart = 80,
          .armijo = Real(1e-4),
          .minimum_step = Real(1e-6)};
}

// These numbers are witness data, not a provider branch or a recognized model name.
auto original_local_body(std::array<int, 3> permutation) {
  constexpr std::array<Real, 9> reaction{2, .2, -.1, .2, 1.8, .15, -.1, .15, 2.2};
  std::array<Real, 9> matrix{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      matrix[i * 3 + j] = reaction[permutation[i] * 3 + permutation[j]];
  return [matrix](const auto& candidate, const auto& captures, auto& result, int) {
    for (std::size_t level = 0; level < result.size(); ++level)
      for (std::size_t patch = 0; patch < result[level].local_size(); ++patch) {
        const auto q = candidate[level].fab(patch).view(),
                   forcing = captures[0][level].fab(patch).view();
        const auto output = result[level].fab(patch).view();
        for_each_cell(result[level].box(patch), [=] POPS_HD(const Index<1>& cell) {
          for (int i = 0; i < 3; ++i) {
            const Real u = q(cell, i);
            Real local = Real(.2) * u * u * u - forcing(cell, i);
            for (int j = 0; j < 3; ++j)
              local += matrix[i * 3 + j] * q(cell, j);
            output(cell, i) += local;
          }
        });
      }
  };
}

TEST(CompositeGeneralField, OriginalAmrNonlinearResidualTwoResolutionsAndPermutation) {
  auto lane = ExecutionLane::duplicate_world_collectively("original-amr-field-witness");
  for (int cells : {16, 32})
    for (auto permutation : {std::array<int, 3>{0, 1, 2}, std::array<int, 3>{2, 0, 1}}) {
      auto provider = original_field_provider(lane, cells, permutation);
      auto exact = original_tower(*provider), forcing = original_tower(*provider),
           defect = original_tower(*provider);
      const std::array<Real, 3> means{1, -.4, .7}, waves{.15, .1, -.05};
      for (int level = 0; level < 2; ++level) {
        const Real dx = Real(1) / Real(cells * (level == 0 ? 1 : 2));
        for (std::size_t patch = 0; patch < exact[level].local_size(); ++patch) {
          const auto output = exact[level].fab(patch).view();
          for_each_cell(exact[level].box(patch), [=] POPS_HD(const Index<1>& cell) {
            const Real wave =
                std::cos(Real(2) * Real(3.14159265358979323846) * (Real(cell[0]) + Real(.5)) * dx);
            for (int i = 0; i < 3; ++i)
              output(cell, i) = means[permutation[i]] + waves[permutation[i]] * wave;
          });
        }
      }
      provider->prepare_original_field_operator();
      provider->synchronize_original_field_candidate(exact);
      provider->apply_original_field_operator(exact, forcing);
      std::vector<Solver::hierarchy_type> zeros;
      zeros.push_back(original_tower(*provider));
      for (auto& level : zeros[0])
        level.set_val(Real(0));
      auto body = original_local_body(permutation);
      body(exact, zeros, forcing,
           0);  // Real discrete forcing A(exact)+f(exact), no guessed results.
      PreparedResourceCache cache;
      auto authority = original_authority(provider.get(), cache.begin_attempt(lane));
      const std::array<const Solver::hierarchy_type*, 1> inputs{&forcing};
      auto prepared = OriginalResidual::prepare(provider, authority, inputs, original_controls(),
                                                Real(1e-5), lane);
      for (auto& level : forcing)
        level.set_val(Real(99));  // Captures must own their original numeric storage.
      for (int level = 0; level < 2; ++level)
        provider->solution(level).set_val(Real(4));
      auto report = prepared->solve(authority, nullptr, body, lane);
      auto outcome = SolveOutcome::collective_lane(std::move(report), lane);
      const auto accepted = outcome.consume(outcome.report().solved() ? SolveConsumption::kAccept
                                                                      : SolveConsumption::kFailRun);
      ASSERT_TRUE(accepted.solved()) << accepted.reason;
      const auto& solved = prepared->candidate(authority, lane);
      EXPECT_GT(prepared->residual_evaluations(), 1);
      EXPECT_GT(prepared->derivative_evaluations(), 0);
      for (int level = 0; level < 2; ++level)
        lincomb(defect[level], Real(1), solved[level], Real(-1), exact[level]);
      EXPECT_LT(std::sqrt(provider->original_field_dot(defect, defect)), Real(3e-7));
      EXPECT_LE(accepted.residual_norm, original_controls().tolerance *
                                            std::max(Real(1), accepted.reference_residual_norm));
      // The original body and its captured forcing are independently reconstructed for
      // this saved candidate, rather than accepting Newton's linearized defect.
      provider->apply_original_field_operator(solved, defect);
      constexpr std::array<Real, 9> reaction{2, .2, -.1, .2, 1.8, .15, -.1, .15, 2.2};
      for (int level = 0; level < 2; ++level)
        for (std::size_t patch = 0; patch < defect[level].local_size(); ++patch) {
          const auto q = solved[level].fab(patch).view();
          const auto target = exact[level].fab(patch).view();
          const auto output = defect[level].fab(patch).view();
          for_each_cell(defect[level].box(patch), [=] POPS_HD(const Index<1>& cell) {
            for (int i = 0; i < 3; ++i) {
              Real value = Real(.2) * (q(cell, i) * q(cell, i) * q(cell, i) -
                                       target(cell, i) * target(cell, i) * target(cell, i));
              for (int j = 0; j < 3; ++j)
                value +=
                    reaction[permutation[i] * 3 + permutation[j]] * (q(cell, j) - target(cell, j));
              output(cell, i) += value;
            }
          });
        }
      provider->apply_original_field_operator(exact, zeros[0]);
      for (int level = 0; level < 2; ++level)
        saxpy(defect[level], Real(-1), zeros[0][level]);
      EXPECT_LT(std::sqrt(provider->original_field_dot(defect, defect)), Real(3e-7));
      for (int level = 0; level < 2; ++level)
        EXPECT_EQ(norm_inf(provider->solution(level), 0), Real(4));
      // Re-preparing shared coefficients revokes this original-operator snapshot.
      provider->prepare_original_field_operator();
      EXPECT_THROW(prepared->require_authority(authority, lane), std::exception);
    }
}

TEST(CompositeGeneralField, OriginalAmrResidualAuthorityAndLocalFailureDoNotPublish) {
  auto lane = ExecutionLane::duplicate_world_collectively("original-amr-field-refusals");
  auto provider = original_field_provider(lane, 16, {0, 1, 2});
  auto capture = original_tower(*provider);
  fill_constant(capture, {1, 2, 3});
  PreparedResourceCache cache;
  auto authority = original_authority(provider.get(), cache.begin_attempt(lane));
  const std::array<const Solver::hierarchy_type*, 1> inputs{&capture};
  auto prepared =
      OriginalResidual::prepare(provider, authority, inputs, original_controls(), Real(1e-5), lane);
  for (int mutation = 0; mutation < 5; ++mutation) {
    ASSERT_NO_THROW(prepared->require_authority(authority, lane));
    auto forged = authority;
    if (lane.rank() == 0) {
      if (mutation == 0)
        forged.points[1].physical_time += .01;
      else if (mutation == 1)
        forged.capture_identities[0] += "changed";
      else if (mutation == 2)
        ++forged.materialization_generation;
      else if (mutation == 3)
        forged.owner = nullptr;
      else
        forged.capture_components[0] = 2;
    }
    EXPECT_THROW(prepared->require_authority(forged, lane), std::exception);
  }
  auto body = original_local_body({0, 1, 2});
  int callbacks = 0;
  EXPECT_THROW(prepared->solve(
                   authority, nullptr,
                   [&](const auto& q, const auto& captured, auto& result, int evaluation) {
                     ++callbacks;
                     if (lane.rank() == 0)
                       throw std::runtime_error("injected original local body failure");
                     body(q, captured, result, evaluation);
                   },
                   lane),
               std::exception);
  EXPECT_GT(all_reduce_min(static_cast<long>(callbacks), lane), 0);
  EXPECT_THROW(prepared->candidate(authority, lane), std::exception);
  PreparedResourceCache foreign_cache;
  auto foreign = authority;
  foreign.attempt = foreign_cache.begin_attempt(lane);
  ASSERT_NO_THROW(prepared->require_authority(authority, lane));
  EXPECT_EQ(foreign.attempt.ordinal(), authority.attempt.ordinal());
  EXPECT_THROW(prepared->require_authority(foreign, lane), std::exception);
  foreign = authority;
  foreign.level_attempts[1] = foreign_cache.current_attempt();
  EXPECT_THROW(prepared->require_authority(foreign, lane), std::exception);
  authority.attempt.reject();
  EXPECT_THROW(prepared->require_authority(authority, lane), std::exception);
  auto retry = original_authority(provider.get(), cache.begin_attempt(lane));
  EXPECT_THROW(prepared->require_authority(retry, lane), std::exception);
  EXPECT_NO_THROW(
      OriginalResidual::prepare(provider, retry, inputs, original_controls(), Real(1e-5), lane));
  for (int level = 0; level < 2; ++level)
    EXPECT_EQ(norm_inf(provider->solution(level), 0), Real(0));
}

TEST(CompositeGeneralField, OriginalApplyOnlyFiniteMatrixDoesNotClaimLinearSpdSolve) {
  auto lane = ExecutionLane::duplicate_world_collectively("original-amr-signed-apply");
  auto provider = original_field_provider(lane, 16, {0, 1, 2});
  for (int level = 0; level < 2; ++level) {
    auto& coefficients = provider->assembly_target("pops.general-field.coefficients", level);
    coefficients.set_val(Real(0));
    for (std::size_t patch = 0; patch < coefficients.local_size(); ++patch) {
      const auto output = coefficients.fab(patch).view();
      for_each_cell(coefficients.box(patch), [=] POPS_HD(const Index<1>& cell) {
        output(cell, 0) = Real(-1);
        output(cell, 4) = Real(2);
        output(cell, 8) = Real(3);
        output(cell, 1) = Real(.3);  // Non-symmetric full matrix, no scalar solve.
      });
    }
  }
  ASSERT_NO_THROW(provider->prepare_original_field_operator());
  auto q = original_tower(*provider), image = original_tower(*provider);
  fill_constant(q, {Real(1), Real(2), Real(3)});
  EXPECT_NO_THROW(provider->apply_original_field_operator(q, image));
  EXPECT_NEAR(provider->original_field_dot(image, image), Real(0), Real(1e-20));
  auto rejected = provider->execute_collectively({Real(1e-10), Real(1e-12), 100}, lane);
  EXPECT_EQ(rejected.consume(SolveConsumption::kFailRun).status, SolveStatus::kInvalidEvaluation);
}

TEST(CompositeGeneralField, OriginalAmrResidualRefusesNoncanonicalStageFractionsBeforePreparation) {
  auto lane = ExecutionLane::duplicate_world_collectively("original-amr-stage-fractions");
  auto provider = original_field_provider(lane, 16, {0, 1, 2});
  auto capture = original_tower(*provider);
  fill_constant(capture, {1, 2, 3});
  PreparedResourceCache cache;
  auto authority = original_authority(provider.get(), cache.begin_attempt(lane));
  const std::array<const Solver::hierarchy_type*, 1> inputs{&capture};
  auto prepared =
      OriginalResidual::prepare(provider, authority, inputs, original_controls(), Real(1e-5), lane);
  for (auto fraction : {std::array<std::int64_t, 2>{0, 0}, {2, 1}, {2, 4}, {-1, 1}}) {
    ASSERT_NO_THROW(prepared->require_authority(authority, lane));
    auto forged = authority;
    if (lane.rank() == 0)
      for (auto& point : forged.points) {
        point.stage_fraction.numerator = fraction[0];
        point.stage_fraction.denominator = fraction[1];
      }
    EXPECT_THROW(
        OriginalResidual::prepare(provider, forged, inputs, original_controls(), Real(1e-5), lane),
        std::exception);
    EXPECT_NO_THROW(prepared->require_authority(authority, lane));
  }
}
