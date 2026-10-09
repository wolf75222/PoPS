#include <pops/runtime/program/spatial_direct_interaction.hpp>
#include <pops/runtime/program/spatial_interaction_history_source.hpp>
#include <pops/numerics/time/amr/levels/amr_subcycling.hpp>
#include <cassert>
#include <iostream>

using namespace pops;
using namespace pops::runtime::program;

template <int D>
int composite(const ExecutionLane& lane) {
  using Field = MultiFab<D>;
  using Level = InteractionLevelView<D, typename Field::memory_space>;
  Index<D> lo{}, coarse_hi{}, fine_hi{}, patch_lo{};
  RealVector<D> lower{}, upper{};
  Extent<D> rank_ext{};
  for (int axis = 0; axis < D; ++axis) { upper[axis] = axis + 1; rank_ext[axis] = 1; }
  upper[0] = 2;
  coarse_hi[0] = 1; fine_hi[0] = 3; patch_lo[0] = 2;
  const auto coarse_geom = Geometry<D>::from_bounds(Box<D>(lo, coarse_hi), lower, upper);
  const auto fine_geom = Geometry<D>::from_bounds(Box<D>(lo, fine_hi), lower, upper);
  const mesh::BoxArray<D> coarse_boxes({Box<D>(lo, coarse_hi)}), fine_boxes({Box<D>(patch_lo, fine_hi)});
  const mesh::RankSpace<D> ranks(lo, rank_ext);
  const auto cd = mesh::Distribution<D>::replicated(coarse_boxes, ranks), fd = mesh::Distribution<D>::replicated(fine_boxes, ranks);
  Field coarse(coarse_boxes, cd, lo, 1, Extent<D>{}), fine(fine_boxes, fd, lo, 1, Extent<D>{});
  Field coverage(coarse_boxes, cd, lo, 1, Extent<D>{});
  auto q = coarse.fab(0).view(), c = coverage.fab(0).view(), f = fine.fab(0).view();
  for_each_cell(coarse.box(0), [=] POPS_HD(const Index<D>& cell) { q(cell, 0) = cell[0] == 0 ? 2 : 1000; c(cell, 0) = cell[0] == 0 ? 1 : 0; });
  for_each_cell(fine.box(0), [=] POPS_HD(const Index<D>& cell) { f(cell, 0) = 6; });
  const std::array<Level, 2> levels{{{&coarse, nullptr, &coverage, nullptr, coarse_geom}, {&fine, nullptr, nullptr, nullptr, fine_geom}}};
  const std::array<int, 1> components{0};
  auto kernel = [] POPS_HD(const RealVector<D>&, const RealVector<D>&) { return Real(1); };
  auto result = direct_spatial_interaction<D, typename Field::memory_space>(levels, 0, components, 1U << 20, "composite", lane, kernel);
  sync_host();
  Real transverse = 1;
  for (int axis = 1; axis < D; ++axis) transverse *= axis + 1;
  assert(result.fab(0).view()(lo, 0) == 8 * transverse);
  assert(result.fab(0).view()(coarse_hi, 0) == 0);
  auto fine_result = direct_spatial_interaction<D, typename Field::memory_space>(levels, 1, components, 1U << 20, "composite-fine", lane, kernel);
  sync_host();
  assert(fine_result.fab(0).view()(patch_lo, 0) == 8 * transverse);
  assert(fine_result.fab(0).view()(fine_hi, 0) == 8 * transverse);
  return 4;
}

int accepted_carriers_after_coarse_commit(const ExecutionLane& lane) {
  constexpr int D = 1;
  namespace hierarchy = pops::amr::hierarchy;
  namespace runtime = pops::runtime::amr;
  using Field = MultiFab<D>;
  using Hierarchy = runtime::PreparedMultiBlockAmrHierarchy<D>;
  using Engine = pops::numerics::time::amr::PreparedMultiBlockAmrSubcyclingEngine<D, double>;
  auto authority = std::make_shared<const PreparedLoadBalanceAuthority<D>>(
      prepare_load_balance_authority<D>("space_filling_curve", "nonlocal-host.sfc",
          PreparedProviderOptions{"pops.amr.load-balance.space-filling-curve@1", {}}));
  const mesh::RankSpace<D> ranks(Index<D>{0}, Extent<D>{1});
  const mesh::BoxArray<D> coarse_boxes({Box<D>(Index<D>{0}, Index<D>{3})});
  const mesh::BoxArray<D> fine_boxes({Box<D>(Index<D>{0}, Index<D>{3})});
  const auto cd = mesh::Distribution<D>::replicated(coarse_boxes, ranks);
  const auto fd = mesh::Distribution<D>::replicated(fine_boxes, ranks);
  const mesh::BoxArrayValidationBudget boxes_budget{4, 6};
  hierarchy::LevelLayout<D> coarse_layout(0, Box<D>(Index<D>{0}, Index<D>{3}), coarse_boxes, cd,
                                        pops::amr::RefinementRatio<D>{}, boxes_budget);
  hierarchy::LevelLayout<D> fine_layout(1, Box<D>(Index<D>{0}, Index<D>{7}), fine_boxes, fd,
                                      pops::amr::RefinementRatio<D>(std::array<int, D>{2}), boxes_budget);
  std::vector<hierarchy::AmrLevelState<D>> levels;
  Field coarse(coarse_boxes, cd, Index<D>{0}, 1, Extent<D>{});
  Field fine(fine_boxes, fd, Index<D>{0}, 1, Extent<D>{});
  coarse.set_val(1); fine.set_val(1);
  levels.emplace_back(coarse_layout, std::move(coarse));
  levels.emplace_back(fine_layout, std::move(fine));
  runtime::AmrRuntime<D> topology(hierarchy::AmrHierarchy<D>(std::move(levels), {2, 64}), authority, "nonlocal-host.space");
  auto accepted = Hierarchy::prepare_collectively(lane, std::move(topology), "rho", {}, "nonlocal-host.lane");
  const std::vector<pops::amr::ParentChildClockRelation> relations{{0, 1, {1, 1}, pops::amr::RemainderPolicy::IntegralOnly}};
  auto engine = Engine::prepare(accepted, relations, {{1, boxes_budget}, pops::amr::reflux::FaceFluxLedgerBudget{32, 32, 1}});
  Field covered(coarse_boxes, cd, Index<D>{0}, 1, Extent<D>{});
  auto mask = covered.fab(0).view();
  for_each_cell(covered.box(0), [=] POPS_HD(const Index<D>& i) { mask(i, 0) = i[0] < 2 ? 0 : 1; });
  const auto cg = Geometry<D>::from_bounds(coarse_layout.domain(), RealVector<D>{0}, RealVector<D>{4});
  const auto fg = Geometry<D>::from_bounds(fine_layout.domain(), RealVector<D>{0}, RealVector<D>{4});
  const auto epoch = accepted.topology_runtime().topology_epoch();
  const auto generation = accepted.topology_runtime().materialization_generation();
  const auto revision = accepted.accepted_revision();
  const auto* initial_coarse = &accepted.state(0, 0);
  const auto* initial_fine = &accepted.state(0, 1);
  const pops::amr::ClockWindow window{{0, 7, {0, 1}, 1.25}, {0, 7, {1, 1}, 1.5}};
  int assertions = 0;
  auto body = [&](auto) {
    auto coarse_group = engine.synchronized_level_group(0);
    auto fine_group = engine.synchronized_level_group(1);
    assert(coarse_group.front().attempt == fine_group.front().attempt); ++assertions;
    assert(coarse_group.front().window.begin.physical_time == fine_group.front().window.begin.physical_time); ++assertions;
    assert(coarse_group.front().window.end.physical_time == fine_group.front().window.end.physical_time); ++assertions;
    assert(&coarse_group.front().candidate != initial_coarse); ++assertions;
    Field coarse_candidate(coarse_group.front().candidate);
    coarse_candidate.set_val(10);
    // Actual candidate move performed by commit_many's active-attempt branch.
    coarse_group.front().candidate = std::move(coarse_candidate);
    assert(&accepted.state(0, 0) == initial_coarse && &accepted.state(0, 1) == initial_fine); ++assertions;
    assert(accepted.accepted_revision() == revision); ++assertions;
    assert(accepted.topology_runtime().topology_epoch() == epoch && accepted.topology_runtime().materialization_generation() == generation); ++assertions;
    assert(reduce_sum(accepted.state(0, 0)) == 4 && reduce_sum(accepted.state(0, 1)) == 4); ++assertions;
    const std::array<InteractionLevelView<D, Field::memory_space>, 2> source{{
        {&accepted.state(0, 0), nullptr, &covered, nullptr, cg}, {&accepted.state(0, 1), nullptr, nullptr, nullptr, fg}}};
    const std::array<int, 1> component{0};
    auto kernel = [] POPS_HD(const RealVector<D>&, const RealVector<D>&) { return Real(1); };
    auto integral = direct_spatial_interaction<D, Field::memory_space>(source, 1, component, 1U << 20, "after-coarse-before-fine", accepted.lane(), kernel);
    sync_host();
    assert(integral.fab(0).view()(Index<D>{0}, 0) == 4); ++assertions;
    assert(reduce_sum(coarse_group.front().candidate) == 40); ++assertions;
    fine_group.front().candidate.set_val(20);
  };
  engine.advance(window, body, [](auto&) {}, [](auto, auto, const auto&) {}, [](auto, auto) {}, true);
  assert(accepted.accepted_revision() != revision); ++assertions;
  assert(reduce_sum(accepted.state(0, 1)) == 80); ++assertions;
  return assertions;
}

int retained_history_authority() {
  HistoryManager<1> manager;
  const auto sample = HistorySampleIdentity{std::bit_cast<std::uint64_t>(0.0),
      std::bit_cast<std::uint64_t>(.01), 1, HistorySampleKind::Publication};
  const mesh::BoxArray<1> boxes({Box<1>(Index<1>{0}, Index<1>{0})});
  const mesh::RankSpace<1> ranks(Index<1>{0}, Extent<1>{1});
  const auto distribution = mesh::Distribution<1>::replicated(boxes, ranks);
  MultiFab<1> seed(boxes, distribution, Index<1>{0}, 1, Extent<1>{}); seed.set_val(Real(2));
  for (const std::string key : {"coarse", "fine"}) {
    MultiFab<1> retained(boxes, distribution, Index<1>{0}, 1, Extent<1>{}); retained.set_val(Real(2));
    manager.histories[key] = {retained, retained}; manager.depth[key] = 2; manager.owner[key] = 0;
    manager.state_identity[key] = "rho"; manager.space_identity[key] = "rho-space";
    manager.clock_identity[key] = "clock"; manager.interpolation_identity[key] = "none";
    manager.initialized[key] = true; manager.fill_count[key] = 2; manager.store_pending[key] = false;
    manager.slot_dt[key] = {Real(.01), Real(.01)}; manager.slot_sample[key] = {sample, sample};
  }
  int checks = 0;
  auto selected = [&](const std::string& key) {
    ExactContractBuilder exact;
    return interaction_history_selected(manager, key, 1, 0, "rho", "rho-space", "clock", "none", exact);
  };
  assert(!selected("coarse")); ++checks;
  manager.slot_sample["coarse"] = manager.prepare_sample_store("coarse", .01, .02);
  manager.slot_dt["coarse"][0] = Real(.02); manager.store_pending["coarse"] = true;
  assert(!selected("coarse") && !selected("fine")); ++checks;
  bool refused = false;
  try { manager.matching_authenticated_sample("coarse", "fine", 1); } catch (const std::invalid_argument&) { refused = true; }
  assert(refused); ++checks;  // Old consumer guard still refuses the exact counter-before.
  manager.state_identity["coarse"] = "storage-Q";
  refused = false; try { selected("coarse"); } catch (const std::invalid_argument&) { refused = true; }
  assert(refused); ++checks; manager.state_identity["coarse"] = "rho";
  for (const std::string key : {"coarse", "fine"}) {
    manager.fill_count[key] = 0; manager.store_pending[key] = false; manager.initialized[key] = false;
    manager.slot_dt[key] = {Real(0), Real(0)};
    manager.slot_sample[key] = {HistorySampleIdentity::zero_start(), HistorySampleIdentity::zero_start()};
    assert(selected(key)); ++checks;
  }
  manager.slot_sample["coarse"] = manager.prepare_sample_store("coarse", 0., .01);
  manager.slot_dt["coarse"] = {Real(.01), Real(.01)};
  manager.initialized["coarse"] = true; manager.store_pending["coarse"] = true;
  assert(selected("coarse") && selected("fine")); ++checks;
  interaction_history_cold_equal(manager.histories["coarse"][1], seed); ++checks;
  seed.set_val(Real(6));  // Q=T+T^2 must never substitute the physical T=2 seed.
  refused = false; try { interaction_history_cold_equal(manager.histories["coarse"][1], seed); }
  catch (const std::invalid_argument&) { refused = true; }
  assert(refused); ++checks;
  auto negative_zero = seed.fab(0).view();
  for_each_cell(seed.box(0), [=] POPS_HD(const Index<1>& cell) { negative_zero(cell, 0) = -Real(0); });
  manager.histories["coarse"][1].set_val(Real(0));
  refused = false; try { interaction_history_cold_equal(manager.histories["coarse"][1], seed); }
  catch (const std::invalid_argument&) { refused = true; }
  assert(refused); ++checks;
  return checks;
}

int main() {
  auto lane = ExecutionLane::world("direct-interaction-host");
  constexpr int D = 2;
  const auto geometry = Geometry<D>::from_bounds(Box<D>(Index<D>{-1, 2}, Index<D>{2, 3}), RealVector<D>{1, -2}, RealVector<D>{3, 2});
  const mesh::BoxArray<D> boxes({Box<D>(Index<D>{-1, 2}, Index<D>{0, 3}), Box<D>(Index<D>{1, 2}, Index<D>{2, 3})});
  const mesh::RankSpace<D> ranks(Index<D>{0, 0}, Extent<D>{1, 1});
  const auto distribution = mesh::Distribution<D>::replicated(boxes, ranks);
  using Field = MultiFab<D>;
  Field rho(boxes, distribution, Index<D>{0, 0}, 3, Extent<D>{0, 0});
  Field active(boxes, distribution, Index<D>{0, 0}, 1, Extent<D>{0, 0});
  Field coverage(boxes, distribution, Index<D>{0, 0}, 1, Extent<D>{0, 0});
  Field kappa(boxes, distribution, Index<D>{0, 0}, 1, Extent<D>{0, 0});
  for (std::size_t patch = 0; patch < rho.local_size(); ++patch) {
    auto q = rho.fab(patch).view(), a = active.fab(patch).view(), c = coverage.fab(patch).view(), k = kappa.fab(patch).view();
    for_each_cell(rho.box(patch), [=] POPS_HD(const Index<D>& i) {
      q(i, 0) = 2 + i[0]; q(i, 1) = -3 + i[1]; q(i, 2) = .5 - i[0];
      a(i, 0) = i == Index<D>{-1, 2} ? 0 : 1;
      c(i, 0) = i == Index<D>{2, 3} ? 0 : 1;
      k(i, 0) = .25 * (i[0] + 2);
      if (a(i, 0) == 0 || c(i, 0) == 0) q(i, 0) = std::numeric_limits<Real>::quiet_NaN();
    });
  }
  const auto kernel = [] POPS_HD(const RealVector<D>& x, const RealVector<D>& y) { return 1 + x[0] * y[0] - 2 * y[1]; };
  const std::array<int, 2> components{2, 0};
  const std::array<InteractionLevelView<D, Field::memory_space>, 1> levels{{{&rho, &active, &coverage, &kappa, geometry}}};
  auto result = direct_spatial_interaction<D, Field::memory_space>(levels, 0, components, 1U << 20, "nonsymmetric-real-field", lane, kernel);
  int assertions = 0;
  for (std::size_t p = 0; p < result.local_size(); ++p) {
    sync_host();
    const auto values = result.fab(p).view();
    const auto box = result.box(p);
    for (int j = box.lo[1]; j <= box.hi[1]; ++j) for (int i = box.lo[0]; i <= box.hi[0]; ++i) {
      const Index<D> target{i, j};
      const bool included = target != Index<D>{-1, 2} && target != Index<D>{2, 3};
      const auto x = geometry.cell_center(target);
      for (int out = 0; out < 2; ++out) {
        double expected = 0;
        for (int sy = 2; sy <= 3; ++sy) for (int sx = -1; sx <= 2; ++sx) {
          if (Index<D>{sx, sy} == Index<D>{-1, 2} || Index<D>{sx, sy} == Index<D>{2, 3}) continue;
          const auto y = geometry.cell_center(Index<D>{sx, sy});
          const double density = out == 0 ? .5 - sx : 2 + sx;
          expected += (1 + x[0] * y[0] - 2 * y[1]) * (.25 * (sx + 2)) * density;
        }
        if (!included) expected = 0;
        assert(std::abs(values(target, out) - expected) < 1e-13); ++assertions;
      }
    }
  }
  bool refused = false;
  try { (void)direct_spatial_interaction<D, Field::memory_space>(levels, 0, components, 1, "budget", lane, kernel); }
  catch (const std::length_error&) { refused = true; }
  assert(refused); ++assertions;
  refused = false;
  auto bad_kernel = [] POPS_HD(const RealVector<D>&, const RealVector<D>&) { return std::numeric_limits<Real>::infinity(); };
  try { (void)direct_spatial_interaction<D, Field::memory_space>(levels, 0, components, 1U << 20, "nonfinite", lane, bad_kernel); }
  catch (const std::overflow_error&) { refused = true; }
  assert(refused); ++assertions;
  assertions += composite<1>(lane) + composite<3>(lane);
  assertions += accepted_carriers_after_coarse_commit(lane);
  assertions += retained_history_authority();
  bool overflow = false;
  try { (void)interaction_product(std::numeric_limits<std::size_t>::max(), 2); }
  catch (const std::overflow_error&) { overflow = true; }
  assert(overflow); ++assertions;
  overflow = false;
  try { (void)interaction_add(std::numeric_limits<std::size_t>::max(), 1); }
  catch (const std::overflow_error&) { overflow = true; }
  assert(overflow); ++assertions;
  auto view = rho.fab(0).view();
  view(Index<D>{0, 2}, 1) = -Real(0);
  assert(std::bit_cast<InteractionRealWord>(interaction_cell_value<D>(view, Index<D>{0, 2}, 1)) == (InteractionRealWord{1} << (interaction_real_bits - 1))); ++assertions;
  const auto negative_zero_word = InteractionRealWord{1} << (interaction_real_bits - 1);
  assert(std::bit_cast<InteractionRealWord>(interaction_owner_value(-Real(0), true, lane)) == negative_zero_word); ++assertions;
  const Real finite_max = std::numeric_limits<Real>::max();
  assert(std::bit_cast<InteractionRealWord>(interaction_owner_value(finite_max, true, lane)) == std::bit_cast<InteractionRealWord>(finite_max)); ++assertions;
  auto mask = kappa.fab(0).view();
  mask(Index<D>{0, 2}, 0) = Real(1.1);
  refused = false;
  try { (void)direct_spatial_interaction<D, Field::memory_space>(levels, 0, components, 1U << 20, "bad-kappa", lane, kernel); }
  catch (const std::invalid_argument&) { refused = true; }
  assert(refused); ++assertions;
  mask(Index<D>{0, 2}, 0) = Real(.5);
  view(Index<D>{0, 2}, 0) = std::numeric_limits<Real>::quiet_NaN();
  refused = false;
  try { (void)direct_spatial_interaction<D, Field::memory_space>(levels, 0, components, 1U << 20, "active-nan", lane, kernel); }
  catch (const std::overflow_error&) { refused = true; }
  assert(refused); ++assertions;
  for (std::size_t p = 0; p < kappa.local_size(); ++p) {
    const auto k = kappa.fab(p).view();
    for_each_cell(kappa.box(p), [=] POPS_HD(const Index<D>& cell) { k(cell, 0) = 0; });
  }
  auto empty = direct_spatial_interaction<D, Field::memory_space>(levels, 0, components, 1U << 20, "zero-eb-measure", lane, bad_kernel);
  sync_host();
  for (std::size_t p = 0; p < empty.local_size(); ++p) {
    const auto values = empty.fab(p).view();
    const auto box = empty.box(p);
    for (int j = box.lo[1]; j <= box.hi[1]; ++j) for (int i = box.lo[0]; i <= box.hi[0]; ++i)
      for (int c = 0; c < 2; ++c) { assert(values(Index<D>{i, j}, c) == 0); ++assertions; }
  }
  std::cout << "spatial direct host assertions=" << assertions << '\n';
}
