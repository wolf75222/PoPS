// This witness assembles the actual PoPS FV tensor operator and independently solves its
// separated radial systems. It tests the disk including r=0, rather than an annulus.
#include <pops/numerics/elliptic/nd/cartesian_tensor_operator.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>
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
  const auto geometry =
      Geometry<2>::from_bounds(domain, RealVector<2>{0, 0}, RealVector<2>{1, 2 * pi});
  const std::int64_t stride = static_cast<std::int64_t>(nr + 2) * (nt + 2);
  std::vector<Real> potential(static_cast<std::size_t>(stride), Real(0));
  std::vector<Real> coefficients(static_cast<std::size_t>(4 * stride), Real(0));
  FieldView<Real, 2> phi{
      potential.data(), Index<2>{-1, -1}, Extent<2>{nr + 2, nt + 2}, {1, nr + 2}, 1, stride};
  FieldView<Real, 2> tensor{
      coefficients.data(), Index<2>{-1, -1}, Extent<2>{nr + 2, nt + 2}, {1, nr + 2}, 4, stride};
  for (int j = -1; j <= nt; ++j)
    for (int i = -1; i <= nr; ++i) {
      // Coefficient ghosts have the same constant-extrapolation contract as the runtime.
      const Real r = geometry.cell_coordinate(0, std::clamp(i, 0, nr - 1));
      tensor(Index<2>{i, j}, 0) = r;
      tensor(Index<2>{i, j}, 3) = Real(1) / r;
    }
  FieldView<const Real, 2> phi_read{
      potential.data(), Index<2>{-1, -1}, Extent<2>{nr + 2, nt + 2}, {1, nr + 2}, 1, stride};
  FieldView<const Real, 2> tensor_read{
      coefficients.data(), Index<2>{-1, -1}, Extent<2>{nr + 2, nt + 2}, {1, nr + 2}, 4, stride};
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
      matrix[static_cast<std::size_t>(row * nr + column)] =
          op.image(Index<2>{row, 0}) / angular_sample;
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
      value -= matrix[static_cast<std::size_t>(row * nr + column)] *
               solution[static_cast<std::size_t>(column)];
    solution[static_cast<std::size_t>(row)] =
        value / matrix[static_cast<std::size_t>(row * nr + row)];
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
    result.residual =
        std::max(result.residual, std::abs(op.image(Index<2>{row, 0}) / angular_sample -
                                           original_rhs[static_cast<std::size_t>(row)]));
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
  const auto geometry =
      Geometry<2>::from_bounds(domain, RealVector<2>{0, 0}, RealVector<2>{1, 2 * pi});
  constexpr std::int64_t stride = (nr + 2) * (nt + 2);
  std::vector<Real> potential(stride);
  std::vector<Real> coefficients(4 * stride, 0);
  FieldView<Real, 2> phi{
      potential.data(), Index<2>{-1, -1}, Extent<2>{nr + 2, nt + 2}, {1, nr + 2}, 1, stride};
  FieldView<Real, 2> tensor{
      coefficients.data(), Index<2>{-1, -1}, Extent<2>{nr + 2, nt + 2}, {1, nr + 2}, 4, stride};
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
  FieldView<const Real, 2> phi_read{
      potential.data(), Index<2>{-1, -1}, Extent<2>{nr + 2, nt + 2}, {1, nr + 2}, 1, stride};
  FieldView<const Real, 2> tensor_read{
      coefficients.data(), Index<2>{-1, -1}, Extent<2>{nr + 2, nt + 2}, {1, nr + 2}, 4, stride};
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
namespace {

template <class Potential, class Coefficient>
std::vector<double> periodic_tensor_image(int cells, Potential potential, Coefficient coefficient,
                                          bool arithmetic = false, bool check_diagonal = false,
                                          double width = -1, unsigned zero_flux_faces = 0,
                                          unsigned dirichlet_faces = 0) {
  using namespace pops;
  const Box<2> domain{Index<2>{0, 0}, Index<2>{cells - 1, cells - 1}};
  const double length = width > 0 ? width : static_cast<double>(cells);
  const auto geometry =
      Geometry<2>::from_bounds(domain, RealVector<2>{0, 0}, RealVector<2>{length, length});
  const std::int64_t plane = static_cast<std::int64_t>(cells + 2) * (cells + 2);
  std::vector<Real> values(static_cast<std::size_t>(plane));
  std::vector<Real> tensors(static_cast<std::size_t>(4 * plane));
  FieldView<Real, 2> phi{
      values.data(), Index<2>{-1, -1}, Extent<2>{cells + 2, cells + 2}, {1, cells + 2}, 1, plane};
  FieldView<Real, 2> tensor{
      tensors.data(), Index<2>{-1, -1}, Extent<2>{cells + 2, cells + 2}, {1, cells + 2}, 4, plane};
  const auto boundary_sample = [=](int coordinate, int axis) {
    if (coordinate >= 0 && coordinate < cells)
      return std::pair<int, Real>{coordinate, Real(1)};
    const int side = coordinate < 0 ? 0 : 1;
    const unsigned face = 1u << (2 * axis + side);
    if (((zero_flux_faces | dirichlet_faces) & face) != 0)
      return std::pair<int, Real>{side == 0 ? 0 : cells - 1,
                                  (dirichlet_faces & face) != 0 ? Real(-1) : Real(1)};
    return std::pair<int, Real>{(coordinate + cells) % cells, Real(1)};
  };
  for (int j = -1; j <= cells; ++j)
    for (int i = -1; i <= cells; ++i) {
      const auto [x, x_sign] = boundary_sample(i, 0);
      const auto [y, y_sign] = boundary_sample(j, 1);
      phi(Index<2>{i, j}, 0) = x_sign * y_sign * potential(x, y);
      for (int slot = 0; slot < 4; ++slot)
        tensor(Index<2>{i, j}, slot) = coefficient(x, y, slot);
    }
  const FieldView<const Real, 2> phi_read{
      values.data(), Index<2>{-1, -1}, Extent<2>{cells + 2, cells + 2}, {1, cells + 2}, 1, plane};
  const FieldView<const Real, 2> tensor_read{
      tensors.data(), Index<2>{-1, -1}, Extent<2>{cells + 2, cells + 2}, {1, cells + 2}, 4, plane};
  const auto op = elliptic::nd::make_cartesian_tensor_operator<
      elliptic::nd::CartesianTensorDivergenceSign::negative_divergence>(
      phi_read, elliptic::nd::packed_cartesian_tensor_coefficients<2>(tensor_read), geometry,
      {.zero_flux_faces = zero_flux_faces,
       .dirichlet_faces = dirichlet_faces,
       .arithmetic_diagonal = arithmetic});
  std::vector<double> result(static_cast<std::size_t>(cells * cells));
  for (int j = 0; j < cells; ++j)
    for (int i = 0; i < cells; ++i)
      result[static_cast<std::size_t>(j * cells + i)] = op.image(Index<2>{i, j});
  if (check_diagonal)
    for (int y = 0; y < cells; ++y)
      for (int x = 0; x < cells; ++x) {
        for (int j = -1; j <= cells; ++j)
          for (int i = -1; i <= cells; ++i) {
            const auto [mapped_x, x_sign] = boundary_sample(i, 0);
            const auto [mapped_y, y_sign] = boundary_sample(j, 1);
            phi(Index<2>{i, j}, 0) = mapped_x == x && mapped_y == y ? x_sign * y_sign : Real(0);
          }
        EXPECT_NEAR(op.diagonal(Index<2>{x, y}), op.image(Index<2>{x, y}), 1e-12);
      }
  return result;
}

}  // namespace

TEST(test_mapped_disk_tensor, discontinuous_spd_tensor_retains_positive_fourier_energy) {
  constexpr double pi = 3.1415926535897932384626433832795;
  for (double contrast : {1., 10., 100.}) {
    const auto mode = [=](int i, int j) { return std::cos(2 * pi * (i - j) / 6); };
    const auto image = periodic_tensor_image(6, mode, [=](int i, int j, int slot) {
      const double a = (i + j) % 2 ? 1. : contrast;
      return a * ((slot == 0 || slot == 3) ? 1. : .9);
    });
    const double eigenvalue = .65 * (2 * contrast / (1 + contrast));
    double energy = 0, mass = 0;
    for (int j = 0; j < 6; ++j)
      for (int i = 0; i < 6; ++i) {
        const double value = image[static_cast<std::size_t>(6 * j + i)];
        EXPECT_NEAR(value, eigenvalue * mode(i, j), 2e-13);
        energy += value * mode(i, j);
        mass += value;
      }
    EXPECT_GT(energy, 0);
    EXPECT_NEAR(mass, 0, 2e-13);
  }
}

TEST(test_mapped_disk_tensor, congruence_retains_representable_extreme_coefficient_contrast) {
  constexpr double pi = 3.1415926535897932384626433832795;
  const auto mode = [=](int i, int j) { return std::cos(2 * pi * (i - j) / 6); };
  const auto image = periodic_tensor_image(6, mode, [](int i, int j, int slot) {
    const double a = (i + j) % 2 ? 1e-200 : 1e200;
    return a * (slot == 0 || slot == 3 ? 1. : .9);
  });
  // H=2e-200 is representable, whereas H/A on the large-coefficient cells underflows.
  // Normalize the observed image, rather than accepting zero with an absolute tolerance.
  for (int j = 0; j < 6; ++j)
    for (int i = 0; i < 6; ++i)
      EXPECT_NEAR(image[static_cast<std::size_t>(6 * j + i)] / 2e-200, .65 * mode(i, j), 2e-13);
}

TEST(test_mapped_disk_tensor, localized_tensor_jump_averages_weighted_gradients) {
  const auto potential = [](int i, int j) {
    if (i == 0 && j == 0)
      return -.2;
    if ((i == 1 && j == 1) || (i == 3 && j == 3))
      return -1.;
    if ((i == 1 && j == 3) || (i == 3 && j == 1))
      return 1.;
    return 0.;
  };
  const auto coefficient = [](int i, int j, int slot) {
    if (i == 0 && j == 0)
      return (slot == 0 || slot == 3) ? 1000. : 900.;
    return (slot == 0 || slot == 3) ? 1. : 0.;
  };
  for (bool arithmetic : {false, true}) {
    const auto image = periodic_tensor_image(4, potential, coefficient, arithmetic);
    double energy = 0, mass = 0;
    for (int j = 0; j < 4; ++j)
      for (int i = 0; i < 4; ++i) {
        const double value = image[static_cast<std::size_t>(4 * j + i)];
        energy += value * potential(i, j);
        mass += value;
      }
    // All weighted central gradients at the sole non-diagonal cell vanish. The remaining
    // energy is exactly scalar edge energy: this oracle does not reuse the native cross flux.
    EXPECT_NEAR(energy, 16 + .16 * (arithmetic ? 500.5 : 2000. / 1001.), 1e-10);
    EXPECT_NEAR(mass, 0, 1e-10);
  }
}

TEST(test_mapped_disk_tensor, variable_tensor_is_symmetric_and_has_the_exact_center_diagonal) {
  const auto coefficient = [](int i, int j, int slot) {
    if (slot == 0)
      return 1. + .4 * i + j;
    if (slot == 3)
      return 2. + .7 * i + .5 * j;
    return .3;
  };
  const auto u = [](int i, int j) { return std::sin(1.3 * i - .7 * j); };
  const auto v = [](int i, int j) { return std::cos(.2 * i + 1.1 * j); };
  for (bool arithmetic : {false, true}) {
    const auto au = periodic_tensor_image(4, u, coefficient, arithmetic, true);
    const auto av = periodic_tensor_image(4, v, coefficient, arithmetic);
    double uav = 0, vau = 0, energy = 0;
    for (int j = 0; j < 4; ++j)
      for (int i = 0; i < 4; ++i) {
        const auto index = static_cast<std::size_t>(4 * j + i);
        uav += u(i, j) * av[index];
        vau += v(i, j) * au[index];
        energy += u(i, j) * au[index];
      }
    EXPECT_NEAR(uav, vau, 1e-12);
    EXPECT_GT(energy, 0);
  }
}

TEST(test_mapped_disk_tensor, tensor_center_diagonal_includes_physical_corner_ghost_influences) {
  const auto coefficient = [](int i, int j, int slot) {
    return slot == 0 ? 2. + .4 * i + .1 * j : slot == 3 ? 3. + .2 * i + .6 * j : .7;
  };
  for (bool arithmetic : {false, true})
    periodic_tensor_image(
        4, [](int i, int j) { return .2 * i - .3 * j; }, coefficient, arithmetic, true, -1, 9u, 6u);
}

TEST(test_mapped_disk_tensor, smooth_variable_tensor_retains_second_order_consistency) {
  constexpr double k = 6.283185307179586476925286766559;
  double previous = 0;
  for (int n : {16, 32}) {
    const auto p = [=](int i, int j) {
      return std::sin(k * (i + .5) / n) * std::sin(k * (j + .5) / n);
    };
    const auto image = periodic_tensor_image(
        n, p,
        [=](int i, int, int slot) {
          const double a = 1 + .2 * std::sin(k * (i + .5) / n);
          return a * (slot == 0 ? 1. : slot == 3 ? 2. : .3);
        },
        false, false, 1.);
    double error = 0;
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        const double x = k * (i + .5) / n, y = k * (j + .5) / n;
        const double a = 1 + .2 * std::sin(x);
        const double exact =
            a * k * k * (3 * std::sin(x) * std::sin(y) - .6 * std::cos(x) * std::cos(y)) -
            .2 * k * k * std::cos(x) * (std::cos(x) * std::sin(y) + .3 * std::sin(x) * std::cos(y));
        const double delta = image[static_cast<std::size_t>(n * j + i)] - exact;
        error += delta * delta;
      }
    error = std::sqrt(error / (n * n));
    if (previous != 0)
      EXPECT_LT(error, .35 * previous);
    previous = error;
  }
}

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
