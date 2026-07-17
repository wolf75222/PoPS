#include <pops_bench/cases.hpp>

#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/load_balance.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace pops::bench {
namespace {

constexpr Real kAlpha = Real(0.125);
constexpr int kGhost = 1;

POPS_HD Real x_pattern(int i, int j, int component) {
  return Real(1) + Real(1e-4) * Real(i) + Real(2e-4) * Real(j) +
         Real(1e-2) * Real(component);
}

POPS_HD Real seed_pattern(int i, int j, int component) {
  return Real(-0.25) + Real(3e-5) * Real(i) - Real(4e-5) * Real(j) +
         Real(2e-2) * Real(component);
}

struct InitializePatterns {
  Array4 x;
  Array4 seed;
  int components;

  POPS_HD void operator()(int i, int j) const {
    for (int component = 0; component < components; ++component) {
      x(i, j, component) = x_pattern(i, j, component);
      seed(i, j, component) = seed_pattern(i, j, component);
    }
  }
};

int periodic_index(int index, int extent) {
  const int remainder = index % extent;
  return remainder < 0 ? remainder + extent : remainder;
}

std::string validation_json(bool passed, bool nonfinite_detected, double error_a, double error_b,
                            double difference, double tolerance) {
  std::ostringstream out;
  out << "{\"passed\":" << (passed ? "true" : "false")
      << ",\"nonfinite_detected\":" << (nonfinite_detected ? "true" : "false")
      << ",\"metric\":\"max_abs_error_valid_and_ghost\",\"saxpy_error\":"
      << json_number(error_a) << ",\"lincomb_error\":" << json_number(error_b)
      << ",\"variant_difference\":" << json_number(difference)
      << ",\"tolerance\":" << json_number(tolerance)
      << ",\"timed\":false}";
  return out.str();
}

std::string parameters_json(const BenchmarkConfig& config, const BoxArray& boxes) {
  std::ostringstream out;
  out << std::setprecision(17) << "{\"nx\":" << config.arith_n
      << ",\"ny\":" << config.arith_n << ",\"tile\":" << config.arith_tile
      << ",\"boxes\":" << boxes.size() << ",\"components\":" << config.arith_components
      << ",\"ghost_width\":" << kGhost << ",\"periodic_x\":true,\"periodic_y\":true"
      << ",\"alpha\":" << kAlpha
      << ",\"global_valid_cells\":"
      << static_cast<long long>(config.arith_n) * static_cast<long long>(config.arith_n) << '}';
  return out.str();
}

}  // namespace

void run_arith_halo_case(const BenchmarkConfig& config, const RuntimeMetadata& metadata,
                         JsonlWriter& writer) {
  const Box2D domain = Box2D::from_extents(config.arith_n, config.arith_n);
  const BoxArray boxes = BoxArray::from_domain(domain, config.arith_tile);
  const DistributionMapping mapping = make_sfc_distribution(boxes, n_ranks());
  MultiFab x(boxes, mapping, config.arith_components, kGhost);
  MultiFab seed(boxes, mapping, config.arith_components, kGhost);
  MultiFab saxpy_field(boxes, mapping, config.arith_components, kGhost);
  MultiFab lincomb_field(boxes, mapping, config.arith_components, kGhost);

  x.set_val(Real(0));
  seed.set_val(Real(0));
  saxpy_field.set_val(Real(0));
  lincomb_field.set_val(Real(0));
  for (int local = 0; local < x.local_size(); ++local) {
    for_each_cell(x.box(local), InitializePatterns{x.fab(local).array(), seed.fab(local).array(),
                                                   config.arith_components});
  }
  device_fence();

  auto reset = [&](MultiFab& field) {
    lincomb(field, Real(1), seed, Real(0), seed);
  };
  auto run_saxpy = [&] {
    saxpy(saxpy_field, kAlpha, x);
    fill_boundary(saxpy_field, domain, Periodicity{true, true});
  };
  auto run_lincomb = [&] {
    lincomb(lincomb_field, Real(1), lincomb_field, kAlpha, x);
    fill_boundary(lincomb_field, domain, Periodicity{true, true});
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
  saxpy_field.sync_host();
  lincomb_field.sync_host();

  double local_error_a = 0.0;
  double local_error_b = 0.0;
  double local_difference = 0.0;
  double local_scale = 1.0;
  long local_nonfinite = 0;
  for (int local = 0; local < saxpy_field.local_size(); ++local) {
    const ConstArray4 a = saxpy_field.fab(local).const_array();
    const ConstArray4 b = lincomb_field.fab(local).const_array();
    const Box2D grown = saxpy_field.box(local).grow(kGhost);
    for (int j = grown.lo[1]; j <= grown.hi[1]; ++j)
      for (int i = grown.lo[0]; i <= grown.hi[0]; ++i) {
        const int wrapped_i = periodic_index(i, config.arith_n);
        const int wrapped_j = periodic_index(j, config.arith_n);
        for (int component = 0; component < config.arith_components; ++component) {
          const double expected =
              static_cast<double>(seed_pattern(wrapped_i, wrapped_j, component) +
                                  kAlpha * x_pattern(wrapped_i, wrapped_j, component));
          const double av = static_cast<double>(a(i, j, component));
          const double bv = static_cast<double>(b(i, j, component));
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

  const std::string parameters = parameters_json(config, boxes);
  const std::string validation = validation_json(passed, nonfinite_detected, error_a, error_b,
                                                 difference, tolerance);
  const std::string timing_common =
      "\"unit\":\"seconds\",\"clock\":\"steady_clock\",\"rank_aggregation\":\"max\","
      "\"device_fence\":\"before_and_after\",\"mpi_barrier\":\"before_and_after\","
      "\"warmup_abba_blocks\":" +
      std::to_string(config.warmups) + ",\"measured_abba_blocks\":" +
      std::to_string(config.repetitions) + ",\"samples_per_variant_per_block\":2,"
      "\"performance_threshold\":null";

  writer.write(record_prefix(metadata, "arith_halo", "saxpy_then_fill_boundary", "paired_abba") +
               ",\"parameters\":" + parameters + ",\"timing\":{" + timing_common +
               ",\"statistics\":" + stats_json(samples.a_seconds) + "},\"validation\":" +
               validation + '}');
  writer.write(record_prefix(metadata, "arith_halo", "lincomb_then_fill_boundary", "paired_abba") +
               ",\"parameters\":" + parameters + ",\"timing\":{" + timing_common +
               ",\"statistics\":" + stats_json(samples.b_seconds) + "},\"validation\":" +
               validation + '}');
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
