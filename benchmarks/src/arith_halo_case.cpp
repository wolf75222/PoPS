#include <pops_bench/cases.hpp>
#include <pops_bench/ranked_setup.hpp>

#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/load_balance.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace pops::bench {
namespace {

constexpr Real kAlpha = Real(0.125);
constexpr int kGhost = 1;
constexpr int kDim = kNativeDimension;

using BenchmarkBox = Box<kDim>;
using BenchmarkLayout = mesh::BoxArray<kDim>;
using BenchmarkField = MultiFab<kDim>;

HaloScheduleBudget halo_budget(const BenchmarkLayout& layout, const BenchmarkBox& domain,
                               int components) {
  const std::size_t boxes = layout.size();
  std::size_t images = 1;
  for (int axis = 0; axis < kDim; ++axis)
    images = checked_product(images, 3, "benchmark halo image budget overflow");
  const std::size_t pairs = checked_product(boxes, boxes, "benchmark halo pair budget overflow");
  const std::size_t work = checked_product(pairs, images, "benchmark halo work budget overflow");
  const std::size_t jobs = checked_product(work, static_cast<std::size_t>(2 * kDim),
                                           "benchmark halo job budget overflow");
  const std::size_t elements = checked_product(
      checked_product(jobs, static_cast<std::size_t>(domain.numPts()),
                      "benchmark halo cell budget overflow"),
      static_cast<std::size_t>(components), "benchmark halo component budget overflow");
  return {
      layout_validation_budget(layout), work, jobs, images, boxes, elements, elements, elements};
}

POPS_HD Real x_pattern(const Index<kDim>& index, int component) {
  Real result = Real(1) + Real(1e-2) * Real(component);
  for (int axis = 0; axis < kDim; ++axis)
    result += Real(axis + 1) * Real(1e-4) * Real(index[axis]);
  return result;
}

POPS_HD Real seed_pattern(const Index<kDim>& index, int component) {
  Real result = Real(-0.25) + Real(2e-2) * Real(component);
  for (int axis = 0; axis < kDim; ++axis) {
    const Real sign = axis % 2 == 0 ? Real(1) : Real(-1);
    result += sign * Real(axis + 3) * Real(1e-5) * Real(index[axis]);
  }
  return result;
}

struct InitializePatterns {
  FieldView<Real, kDim> x;
  FieldView<Real, kDim> seed;
  int components;

  POPS_HD void operator()(const Index<kDim>& index) const {
    for (int component = 0; component < components; ++component) {
      x(index, component) = x_pattern(index, component);
      seed(index, component) = seed_pattern(index, component);
    }
  }
};

Index<kDim> periodic_index(Index<kDim> index, const BenchmarkBox& domain) {
  for (int axis = 0; axis < kDim; ++axis) {
    const int extent = static_cast<int>(domain.length(axis));
    const int offset = index[axis] - domain.lo[axis];
    const int remainder = offset % extent;
    index[axis] = domain.lo[axis] + (remainder < 0 ? remainder + extent : remainder);
  }
  return index;
}

std::string validation_json(bool passed, bool nonfinite_detected, double error_a, double error_b,
                            double difference, double tolerance) {
  std::ostringstream out;
  out << "{\"passed\":" << (passed ? "true" : "false")
      << ",\"nonfinite_detected\":" << (nonfinite_detected ? "true" : "false")
      << ",\"metric\":\"max_abs_error_valid_and_ghost\",\"saxpy_error\":" << json_number(error_a)
      << ",\"lincomb_error\":" << json_number(error_b)
      << ",\"variant_difference\":" << json_number(difference)
      << ",\"tolerance\":" << json_number(tolerance) << ",\"timed\":false}";
  return out.str();
}

std::string parameters_json(const BenchmarkConfig& config, const BenchmarkLayout& boxes,
                            const BenchmarkBox& domain) {
  std::ostringstream out;
  out << std::setprecision(17) << "{\"dimension\":" << kDim
      << ",\"uniform_extent\":" << config.arith_n << ",\"tile\":" << config.arith_tile
      << ",\"boxes\":" << boxes.size() << ",\"components\":" << config.arith_components
      << ",\"ghost_width\":" << kGhost << ",\"periodic_all_axes\":true"
      << ",\"alpha\":" << kAlpha << ",\"global_valid_cells\":" << domain.numPts() << '}';
  return out.str();
}

}  // namespace

void run_arith_halo_case(const BenchmarkConfig& config, const RuntimeMetadata& metadata,
                         JsonlWriter& writer) {
  const BenchmarkBox domain =
      BenchmarkBox::from_extents(filled_ranked<Extent<kDim>>(config.arith_n));
  const BenchmarkLayout boxes =
      BenchmarkLayout::from_domain(domain, filled_ranked<Extent<kDim>>(config.arith_tile));
  const mesh::RankSpace<kDim> ranks = benchmark_rank_space<kDim>();
  const auto ownership = parallel::LoadBalanceProvider<kDim>::space_filling_curve().prepare(
      boxes, ranks, load_balance_budget<kDim>(boxes, domain));
  const auto& distribution = ownership.distribution();
  const Index<kDim> local_rank = benchmark_local_rank(ranks);
  const Extent<kDim> ghosts = filled_ranked<Extent<kDim>>(kGhost);
  BenchmarkField x(boxes, distribution, local_rank, config.arith_components, ghosts);
  BenchmarkField seed(boxes, distribution, local_rank, config.arith_components, ghosts);
  BenchmarkField saxpy_field(boxes, distribution, local_rank, config.arith_components, ghosts);
  BenchmarkField lincomb_field(boxes, distribution, local_rank, config.arith_components, ghosts);

  x.set_val(Real(0));
  seed.set_val(Real(0));
  saxpy_field.set_val(Real(0));
  lincomb_field.set_val(Real(0));
  for (std::size_t local = 0; local < x.local_size(); ++local) {
    for_each_cell(x.box(local), InitializePatterns{x.fab(local).view(), seed.fab(local).view(),
                                                   config.arith_components});
  }
  device_fence();

  std::array<bool, kDim> periodic{};
  periodic.fill(true);
  const HaloSchedule<kDim> schedule =
      prepare_halo_schedule(saxpy_field, domain, BoundaryTopology<kDim>::axis_periodic(periodic),
                            halo_budget(boxes, domain, config.arith_components));
#ifdef POPS_HAS_MPI
  const ExecutionLane lane =
      ExecutionLane::duplicate_world_collectively("pops.benchmark.arith-halo");
  HaloExchangeContext saxpy_context{};
  saxpy_context.context_generation = 1;
  saxpy_context.schedule_generation = 1;
  HaloExchange<kDim> saxpy_exchange(schedule, lane, saxpy_context);
  HaloExchangeContext lincomb_context{};
  lincomb_context.context_generation = 2;
  lincomb_context.schedule_generation = 1;
  HaloExchange<kDim> lincomb_exchange(schedule, lane, lincomb_context);
#endif

  auto reset = [&](BenchmarkField& field) { lincomb(field, Real(1), seed, Real(0), seed); };
  auto run_saxpy = [&] {
    saxpy(saxpy_field, kAlpha, x);
#ifdef POPS_HAS_MPI
    fill_boundary(saxpy_field, saxpy_exchange, lane);
#else
    fill_boundary(saxpy_field, schedule);
#endif
  };
  auto run_lincomb = [&] {
    lincomb(lincomb_field, Real(1), lincomb_field, kAlpha, x);
#ifdef POPS_HAS_MPI
    fill_boundary(lincomb_field, lincomb_exchange, lane);
#else
    fill_boundary(lincomb_field, schedule);
#endif
  };
  auto observe = [](bool) {};

  const PairedSamples samples = run_paired_abba(
      config.warmups, config.repetitions, [&] { reset(saxpy_field); }, run_saxpy, observe,
      [&] { reset(lincomb_field); }, run_lincomb, observe);

  // Numerical validation is intentionally outside every timed interval. Re-run both variants from
  // the same seed and inspect valid cells plus the one-cell periodic halo.
  reset(saxpy_field);
  run_saxpy();
  reset(lincomb_field);
  run_lincomb();
  device_fence();
  barrier();
  double local_error_a = 0.0;
  double local_error_b = 0.0;
  double local_difference = 0.0;
  double local_scale = 1.0;
  long local_nonfinite = 0;
  for (std::size_t local = 0; local < saxpy_field.local_size(); ++local) {
    const auto& a_fab = saxpy_field.fab(local);
    const auto& b_fab = lincomb_field.fab(local);
    auto a = a_fab.create_host_mirror();
    auto b = b_fab.create_host_mirror();
    a_fab.copy_to_host(a);
    b_fab.copy_to_host(b);
    const BenchmarkBox grown = a_fab.grown_box();
    for (std::int64_t ordinal = 0; ordinal < grown.numPts(); ++ordinal) {
      const Index<kDim> index = index_from_ordinal(grown, ordinal);
      const Index<kDim> wrapped = periodic_index(index, domain);
      for (int component = 0; component < config.arith_components; ++component) {
        const double expected = static_cast<double>(seed_pattern(wrapped, component) +
                                                    kAlpha * x_pattern(wrapped, component));
        const double av = static_cast<double>(a(host_offset(grown, index, component)));
        const double bv = static_cast<double>(b(host_offset(grown, index, component)));
        if (!std::isfinite(expected) || !std::isfinite(av) || !std::isfinite(bv)) {
          local_nonfinite = 1;
          continue;
        }
        local_error_a = std::max(local_error_a, std::fabs(av - expected));
        local_error_b = std::max(local_error_b, std::fabs(bv - expected));
        local_difference = std::max(local_difference, std::fabs(av - bv));
        local_scale = std::max(local_scale, std::fabs(expected));
      }
    }
  }
  const double error_a = all_reduce_max(local_error_a);
  const double error_b = all_reduce_max(local_error_b);
  const double difference = all_reduce_max(local_difference);
  const double scale_value = all_reduce_max(local_scale);
  const bool nonfinite_detected = all_reduce_max(local_nonfinite) != 0;
  const double tolerance =
      64.0 * static_cast<double>(std::numeric_limits<Real>::epsilon()) * scale_value;
  const bool passed = !nonfinite_detected && error_a <= tolerance && error_b <= tolerance &&
                      difference <= tolerance;

  const std::string parameters = parameters_json(config, boxes, domain);
  const std::string validation =
      validation_json(passed, nonfinite_detected, error_a, error_b, difference, tolerance);
  const std::string timing_common =
      "\"unit\":\"seconds\",\"clock\":\"steady_clock\",\"rank_aggregation\":\"max\","
      "\"device_fence\":\"before_and_after\",\"mpi_barrier\":\"before_and_after\","
      "\"warmup_abba_blocks\":" +
      std::to_string(config.warmups) +
      ",\"measured_abba_blocks\":" + std::to_string(config.repetitions) +
      ",\"samples_per_variant_per_block\":2,"
      "\"performance_threshold\":null";

  writer.write(record_prefix(metadata, "arith_halo", "saxpy_then_fill_boundary", "paired_abba") +
               ",\"parameters\":" + parameters + ",\"timing\":{" + timing_common +
               ",\"statistics\":" + stats_json(samples.a_seconds) +
               "},\"validation\":" + validation + '}');
  writer.write(record_prefix(metadata, "arith_halo", "lincomb_then_fill_boundary", "paired_abba") +
               ",\"parameters\":" + parameters + ",\"timing\":{" + timing_common +
               ",\"statistics\":" + stats_json(samples.b_seconds) +
               "},\"validation\":" + validation + '}');
  writer.write(record_prefix(metadata, "arith_halo", "saxpy_over_lincomb", "paired_abba",
                             "paired_comparison") +
               ",\"parameters\":" + parameters +
               ",\"comparison\":{\"metric\":\"time_ratio\","
               "\"numerator\":\"saxpy_then_fill_boundary\","
               "\"denominator\":\"lincomb_then_fill_boundary\","
               "\"ordering\":\"ABBA\",\"performance_threshold\":null,"
               "\"statistics\":" +
               stats_json(samples.a_over_b) + "},\"validation\":" + validation + '}');

  if (!passed)
    throw std::runtime_error("arith_halo numerical validation failed");
}

}  // namespace pops::bench
