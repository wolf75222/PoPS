// This witness assembles the actual PoPS FV tensor operator and independently solves its
// separated radial systems. It tests the disk including r=0, rather than an annulus.
#include <pops/numerics/elliptic/nd/cartesian_tensor_operator.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifndef POPS_MAPPED_DISK_STANDALONE
#include <gtest/gtest.h>
#endif

namespace {

struct DiskMeasurement {
  double l2 = 0;
  double linf = 0;
  double residual = 0;
  double wall = 0;
};

DiskMeasurement mapped_disk_error(int nr, int mode) {
  using namespace pops;
  constexpr double pi = 3.1415926535897932384626433832795;
  const int nt = 2 * nr;
  const Box<2> domain{Index<2>{0, 0}, Index<2>{nr - 1, nt - 1}};
  const auto geometry = Geometry<2>::from_bounds(domain, RealVector<2>{0, 0}, RealVector<2>{1, 2 * pi});
  const std::int64_t stride = static_cast<std::int64_t>(nr + 2) * (nt + 2);
  std::vector<Real> potential(static_cast<std::size_t>(stride), Real(0));
  std::vector<Real> coefficients(static_cast<std::size_t>(4 * stride), Real(0));
  FieldView<Real, 2> phi{potential.data(), Index<2>{-1, -1}, Extent<2>{nr + 2, nt + 2},
                          {1, nr + 2}, 1, stride};
  FieldView<Real, 2> tensor{coefficients.data(), Index<2>{-1, -1}, Extent<2>{nr + 2, nt + 2},
                             {1, nr + 2}, 4, stride};
  for (int j = -1; j <= nt; ++j)
    for (int i = -1; i <= nr; ++i) {
      // Coefficient ghosts have the same constant-extrapolation contract as the runtime.
      const Real r = geometry.cell_coordinate(0, std::clamp(i, 0, nr - 1));
      tensor(Index<2>{i, j}, 0) = r;
      tensor(Index<2>{i, j}, 3) = Real(1) / r;
    }
  FieldView<const Real, 2> phi_read{potential.data(), Index<2>{-1, -1}, Extent<2>{nr + 2, nt + 2},
                                     {1, nr + 2}, 1, stride};
  FieldView<const Real, 2> tensor_read{coefficients.data(), Index<2>{-1, -1}, Extent<2>{nr + 2, nt + 2},
                                        {1, nr + 2}, 4, stride};
  const auto op = elliptic::nd::make_cartesian_tensor_operator<
      elliptic::nd::CartesianTensorDivergenceSign::negative_divergence>(
      phi_read, elliptic::nd::packed_cartesian_tensor_coefficients<2>(tensor_read), geometry,
      {.zero_flux_faces = 1u, .dirichlet_faces = 2u, .arithmetic_diagonal = true});
  const double angular_sample = std::cos(mode * geometry.cell_coordinate(1, 0));
  std::vector<double> matrix(static_cast<std::size_t>(nr * nr), 0);
  std::vector<double> rhs(static_cast<std::size_t>(nr), 0);
  const auto install = [&](const std::vector<double>& values) {
    for (int j = -1; j <= nt; ++j) {
      const int wrapped = (j + nt) % nt;
      const Real phase = std::cos(mode * geometry.cell_coordinate(1, wrapped));
      for (int i = 0; i < nr; ++i)
        phi(Index<2>{i, j}) = values[static_cast<std::size_t>(i)] * phase;
      phi(Index<2>{-1, j}) = phi(Index<2>{0, j});
      phi(Index<2>{nr, j}) = -phi(Index<2>{nr - 1, j});
    }
  };
  for (int column = 0; column < nr; ++column) {
    std::vector<double> basis(static_cast<std::size_t>(nr), 0);
    basis[static_cast<std::size_t>(column)] = 1;
    install(basis);
    for (int row = 0; row < nr; ++row)
      matrix[static_cast<std::size_t>(row * nr + column)] = op.image(Index<2>{row, 0}) / angular_sample;
  }
  for (int row = 0; row < nr; ++row) {
    const double r = geometry.cell_coordinate(0, row);
    // psi=r^m(1-r^2)cos(m theta), -div(K grad psi)=4(m+1)r^(m+1)cos(m theta).
    rhs[static_cast<std::size_t>(row)] = 4 * (mode + 1) * std::pow(r, mode + 1);
  }
  const auto original_rhs = rhs;
  for (int column = 0; column < nr; ++column) {
    int pivot = column;
    for (int row = column + 1; row < nr; ++row)
      if (std::abs(matrix[static_cast<std::size_t>(row * nr + column)]) >
          std::abs(matrix[static_cast<std::size_t>(pivot * nr + column)]))
        pivot = row;
    if (!(std::abs(matrix[static_cast<std::size_t>(pivot * nr + column)]) > 1e-14))
      throw std::runtime_error("mapped disk operator has a singular radial mode");
    for (int entry = column; entry < nr; ++entry)
      std::swap(matrix[static_cast<std::size_t>(column * nr + entry)],
                matrix[static_cast<std::size_t>(pivot * nr + entry)]);
    std::swap(rhs[static_cast<std::size_t>(column)], rhs[static_cast<std::size_t>(pivot)]);
    for (int row = column + 1; row < nr; ++row) {
      const double multiplier = matrix[static_cast<std::size_t>(row * nr + column)] /
                                matrix[static_cast<std::size_t>(column * nr + column)];
      for (int entry = column; entry < nr; ++entry)
        matrix[static_cast<std::size_t>(row * nr + entry)] -=
            multiplier * matrix[static_cast<std::size_t>(column * nr + entry)];
      rhs[static_cast<std::size_t>(row)] -= multiplier * rhs[static_cast<std::size_t>(column)];
    }
  }
  std::vector<double> solution(static_cast<std::size_t>(nr), 0);
  for (int row = nr - 1; row >= 0; --row) {
    double value = rhs[static_cast<std::size_t>(row)];
    for (int column = row + 1; column < nr; ++column)
      value -= matrix[static_cast<std::size_t>(row * nr + column)] * solution[static_cast<std::size_t>(column)];
    solution[static_cast<std::size_t>(row)] = value / matrix[static_cast<std::size_t>(row * nr + row)];
  }
  install(solution);
  DiskMeasurement result;
  double weight = 0;
  for (int row = 0; row < nr; ++row) {
    const double r = geometry.cell_coordinate(0, row);
    const double error = solution[static_cast<std::size_t>(row)] - std::pow(r, mode) * (1 - r * r);
    result.linf = std::max(result.linf, std::abs(error));
    result.l2 += r * error * error;
    weight += r;
    result.residual = std::max(result.residual, std::abs(
        op.image(Index<2>{row, 0}) / angular_sample - original_rhs[static_cast<std::size_t>(row)]));
  }
  result.l2 = std::sqrt(result.l2 / weight);
  result.wall = std::abs(Real(0.5) * (phi(Index<2>{nr - 1, 0}) + phi(Index<2>{nr, 0})));
  return result;
}

std::pair<double, double> conormal_balance() {
  using namespace pops;
  constexpr int nr = 16;
  constexpr int nt = 32;
  constexpr double pi = 3.1415926535897932384626433832795;
  const Box<2> domain{Index<2>{0, 0}, Index<2>{nr - 1, nt - 1}};
  const auto geometry = Geometry<2>::from_bounds(domain, RealVector<2>{0, 0}, RealVector<2>{1, 2 * pi});
  constexpr std::int64_t stride = (nr + 2) * (nt + 2);
  std::vector<Real> potential(stride);
  std::vector<Real> coefficients(4 * stride, 0);
  FieldView<Real, 2> phi{potential.data(), Index<2>{-1, -1}, Extent<2>{nr + 2, nt + 2},
                          {1, nr + 2}, 1, stride};
  FieldView<Real, 2> tensor{coefficients.data(), Index<2>{-1, -1}, Extent<2>{nr + 2, nt + 2},
                             {1, nr + 2}, 4, stride};
  for (int j = -1; j <= nt; ++j)
    for (int i = -1; i <= nr; ++i) {
      const double r = geometry.cell_coordinate(0, std::clamp(i, 0, nr - 1));
      const double angle = geometry.cell_coordinate(1, (j + nt) % nt);
      phi(Index<2>{i, j}) = std::exp(r) * std::cos(angle);
      tensor(Index<2>{i, j}, 0) = 1;
      tensor(Index<2>{i, j}, 1) = 0.25 * std::sin(angle);
      tensor(Index<2>{i, j}, 2) = -0.25 * std::sin(angle);
      tensor(Index<2>{i, j}, 3) = 1;
    }
  FieldView<const Real, 2> phi_read{potential.data(), Index<2>{-1, -1}, Extent<2>{nr + 2, nt + 2},
                                     {1, nr + 2}, 1, stride};
  FieldView<const Real, 2> tensor_read{coefficients.data(), Index<2>{-1, -1}, Extent<2>{nr + 2, nt + 2},
                                        {1, nr + 2}, 4, stride};
  const auto derivative_boundary = elliptic::nd::make_cartesian_tensor_operator<
      elliptic::nd::CartesianTensorDivergenceSign::negative_divergence>(
      phi_read, elliptic::nd::packed_cartesian_tensor_coefficients<2>(tensor_read), geometry);
  const auto flux_boundary = elliptic::nd::make_cartesian_tensor_operator<
      elliptic::nd::CartesianTensorDivergenceSign::negative_divergence>(
      phi_read, elliptic::nd::packed_cartesian_tensor_coefficients<2>(tensor_read), geometry,
      {.zero_flux_faces = 3u});
  double derivative_sum = 0;
  double flux_sum = 0;
  for (int j = 0; j < nt; ++j)
    for (int i = 0; i < nr; ++i) {
      derivative_sum += derivative_boundary.image(Index<2>{i, j});
      flux_sum += flux_boundary.image(Index<2>{i, j});
    }
  const double measure = geometry.spacing(0) * geometry.spacing(1);
  return {derivative_sum * measure, flux_sum * measure};
}

}  // namespace

#ifdef POPS_MAPPED_DISK_STANDALONE
int main() {
  std::cout << std::setprecision(14);
  const auto balance = conormal_balance();
  std::cout << "normal_derivative_balance=" << balance.first
            << " conormal_flux_balance=" << balance.second << '\n';
  if (std::abs(balance.first) < 0.1 || std::abs(balance.second) > 1e-11)
    return 1;
  for (const int mode : {0, 1, 3, 5}) {
    double prior = 0;
    for (const int cells : {16, 32, 64, 128}) {
      const auto measurement = mapped_disk_error(cells, mode);
      std::cout << "mode=" << mode << " nr=" << cells << " l2=" << measurement.l2
                << " linf=" << measurement.linf << " residual=" << measurement.residual
                << " wall=" << measurement.wall << " ratio=" << (prior / measurement.l2) << '\n';
      if (measurement.residual > 1e-8 || measurement.wall != 0 ||
          (prior > 0 && prior / measurement.l2 < 3.3))
        return 1;
      prior = measurement.l2;
    }
  }
}
#else
TEST(test_mapped_disk_tensor, homogeneous_neumann_suppresses_the_complete_tensor_flux) {
  const auto balance = conormal_balance();
  EXPECT_GT(std::abs(balance.first), 0.1);
  EXPECT_LT(std::abs(balance.second), 1e-11);
}

TEST(test_mapped_disk_tensor, conducting_disk_modes_converge_without_a_central_hole) {
  for (const int mode : {0, 1, 3, 5}) {
    const auto coarse = mapped_disk_error(32, mode);
    const auto fine = mapped_disk_error(64, mode);
    EXPECT_LT(coarse.residual, 1e-8);
    EXPECT_LT(fine.residual, 1e-8);
    EXPECT_EQ(coarse.wall, 0);
    EXPECT_EQ(fine.wall, 0);
    EXPECT_GT(coarse.l2, 0);
    EXPECT_LT(fine.l2, coarse.l2 / 3.3) << "mode " << mode;
  }
}
#endif
