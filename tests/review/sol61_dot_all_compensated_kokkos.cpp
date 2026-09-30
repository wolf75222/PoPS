// Standalone host reception with real Kokkos, MultiFab and dot-all helper.
// No Program execution, MPI, GPU or installed PoPS extension is received here.
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

using Field = pops::MultiFab<1, Kokkos::HostSpace>;
int checks = 0;
void check(bool condition) {
  ++checks;
  assert(condition);
}

Field field(const std::vector<std::vector<double>>& values, int tile) {
  const auto layout = pops::mesh::BoxArray<1>::from_domain(
      pops::Box<1>{pops::Index<1>{-3}, pops::Index<1>{static_cast<int>(values.size()) - 4}},
      pops::Extent<1>{tile});
  const pops::mesh::RankSpace<1> ranks{pops::Index<1>{-2}, pops::Extent<1>{1}};
  const auto distribution = pops::mesh::Distribution<1>::replicated(layout, ranks);
  Field result(layout, distribution, pops::Index<1>{-2}, static_cast<int>(values[0].size()), {});
  for (std::size_t local = 0; local < result.local_size(); ++local) {
    const auto box = result.box(local);
    auto view = result.fab(local).view();
    for (int i = box.lo[0]; i <= box.hi[0]; ++i)
      for (int c = 0; c < result.ncomp(); ++c)
        view(pops::Index<1>{i}, c) = values.at(i + 3).at(c);
  }
  return result;
}

double dot(const Field& a, const Field& b, const Field* active = nullptr,
           const Field* coverage = nullptr) {
  return pops::dot_owned_active_all_finite_local(a, b, active, coverage);
}
template <class F>
void overflow(F operation) {
  try {
    operation();
    check(false);
  } catch (const std::overflow_error&) {
    check(true);
  }
}

void cancellation() {
  std::vector<double> values{1e16, 1, -1e16};
  std::sort(values.begin(), values.end());
  do {
    for (int tile : {1, 2, 3}) {
      auto a = field({{values[0]}, {values[1]}, {values[2]}}, tile);
      auto b = field({{1}, {1}, {1}}, tile);
      check(dot(a, b) == 1);
    }
  } while (std::next_permutation(values.begin(), values.end()));
  std::sort(values.begin(), values.end());
  do {
    auto a = field({values}, 1), b = field({{1, 1, 1}}, 1);
    check(dot(a, b) == 1);  // Compensation survives the component boundary.
  } while (std::next_permutation(values.begin(), values.end()));
  for (int tile : {1, 16, 64}) {
    std::vector<std::vector<double>> a(64, {0.6, 0.8}), d(64);
    for (auto& value : d) {
      const double x = 0.6, y = 0.8, dt = 1;
      const double px = x - dt * y, py = y + dt * x;
      value = {((x - .5 * dt * y) - .5 * dt * py) - x, ((y + .5 * dt * x) + .5 * dt * px) - y};
    }
    auto q = field(a, tile), increment = field(d, tile);
    const double reached = 0.8 + (-2 * dot(q, increment) / dot(increment, increment));
    double lower = 1.6;
    for (int i = 0; i < 4; ++i)
      lower = std::nextafter(lower, -INFINITY);
    check(reached >= lower && reached <= 1.6);
    std::reverse(a.begin(), a.end());
    std::reverse(d.begin(), d.end());
    auto qp = field(a, tile), dp = field(d, tile);
    check(dot(q, increment) == dot(qp, dp));
    for (auto& row : a)
      std::reverse(row.begin(), row.end());
    for (auto& row : d)
      std::reverse(row.begin(), row.end());
    qp = field(a, tile);
    dp = field(d, tile);
    check(dot(q, increment) == dot(qp, dp));
    // Exact binary64 values from the second authentic checkpoint (retry .5).
    const double x = 0x1.c3c3c3c3c3c3cp-1, y = 0x1.e1e1e1e1e1e1ep-2, dt = .5;
    const double px = x - dt * y, py = y + dt * x;
    a.assign(64, {x, y});
    d.assign(64, {((x - .5 * dt * y) - .5 * dt * py) - x, ((y + .5 * dt * x) + .5 * dt * px) - y});
    q = field(a, tile);
    increment = field(d, tile);
    check(y + (-2 * dot(q, increment) / dot(increment, increment)) * dt == 0.9411764705882353);
  }
}

struct ShiftedTwoDimensionalTerms {
  POPS_HD double operator()(const pops::Index<2>& index) const {
    const int position = index[0] + 2;
    return position == 0 ? 1e16 : (position == 1 ? 1 : -1e16);
  }
};

void two_dimensional_kernel() {
  const pops::Box<2> box{pops::Index<2>{-2, 3}, pops::Index<2>{0, 10}};
  const auto sum = pops::for_each_cell_reduce_finite_sum(box, ShiftedTwoDimensionalTerms{});
  check(sum.finite() && sum.value() == 8);
  const auto empty =
      pops::for_each_cell_reduce_finite_sum(pops::Box<2>{}, ShiftedTwoDimensionalTerms{});
  check(empty.finite() && empty.value() == 0);
}

void masks_and_failures() {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  auto a = field({{nan, nan}, {2, 3}}, 1), b = field({{1, 1}, {2, 3}}, 1);
  auto active = field({{0}, {1}}, 1), coverage = field({{1}, {1}}, 1);
  check(dot(a, b, &active, &coverage) == 13);
  active = field({{1}, {1}}, 1);
  coverage = field({{0}, {1}}, 1);
  check(dot(a, b, &active, &coverage) == 13);
  coverage = field({{1}, {1}}, 1);
  overflow([&] { dot(a, b, &active, &coverage); });
  active = field({{nan}, {1}}, 1);
  overflow([&] { dot(a, b, &active, &coverage); });
  active = field({{1}, {1}}, 1);
  coverage = field({{nan}, {1}}, 1);
  overflow([&] { dot(a, b, &active, &coverage); });
  a = field({{1e308}, {1e308}}, 1);
  b = field({{1}, {1}}, 1);
  overflow([&] { dot(a, b); });  // Cross-patch overflow.
  a = field({{1e308}, {1e308}}, 2);
  b = field({{1}, {1}}, 2);
  overflow([&] { dot(a, b); });  // Kernel overflow.
  a = field({{1e308}}, 1);
  b = field({{1e308}}, 1);
  overflow([&] { dot(a, b); });  // Product overflow.
  a = field({{1e308, 1e308}}, 1);
  b = field({{1, 1}}, 1);
  overflow([&] { dot(a, b); });  // Component overflow.
  pops::FiniteCompensatedSum first, second;
  first.add(1e16);
  first.add(1);
  second.add(-1e16);
  first.join(second);
  check(first.finite() && first.value() == 1);
  second.add(INFINITY);
  first.join(second);
  check(!first.finite());
  // MPI finalization limitation: finalized rank scalars still lose a low part.
  first = {};
  second = {};
  first.add(1e16);
  first.add(1);
  second.add(-1e16);
  check(first.value() + second.value() == 0);
}

void real_lane_summaries() {
  pops::comm_init();
  auto lane = pops::ExecutionLane::duplicate_world_collectively("pops.test.compensated-summary");
  pops::FiniteCompensatedSum local;
  if (lane.rank() == 0) {
    local.add(1e16);
    local.add(1);
  }
  if (lane.rank() == 1 || lane.size() == 1)
    local.add(-1e16);
  check(pops::collective_finite_compensated_sum(local, lane.communicator()) == 1);
  pops::FiniteCompensatedSum bad;
  bad.invalid = lane.rank() == lane.size() - 1;
  overflow([&] { pops::collective_finite_compensated_sum(bad, lane.communicator()); });
  if (lane.size() > 1) {
    local = {};
    local.add(std::numeric_limits<double>::max() * .75);
    overflow([&] { pops::collective_finite_compensated_sum(local, lane.communicator()); });
  }
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    cancellation();
    two_dimensional_kernel();
    masks_and_failures();
    real_lane_summaries();
  }
  Kokkos::finalize();
  std::cout << "real-kokkos-host checks=" << checks << '\n';
}
