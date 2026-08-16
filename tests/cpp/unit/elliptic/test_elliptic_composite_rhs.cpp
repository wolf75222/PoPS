#include <gtest/gtest.h>

#include <pops/coupling/base/elliptic_rhs.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/physics/bricks/elliptic.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

template <int Dim, class Value>
std::array<Value, Dim> filled(Value value) {
  std::array<Value, Dim> result{};
  result.fill(value);
  return result;
}

template <class Ranked, int Dim, class Value>
Ranked ranked(Value value) {
  Ranked result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::Index<Dim> index_from_ordinal(const pops::Box<Dim>& box, std::size_t ordinal) {
  pops::Index<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis) {
    const auto length = static_cast<std::size_t>(box.length(axis));
    result[axis] = box.lo[axis] + static_cast<int>(ordinal % length);
    ordinal /= length;
  }
  return result;
}

template <int Dim>
std::size_t storage_ordinal(const pops::Box<Dim>& box, const pops::Index<Dim>& index) {
  std::size_t ordinal = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    ordinal += static_cast<std::size_t>(index[axis] - box.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(box.length(axis));
  }
  return ordinal;
}

template <int Dim>
struct ScalarEllipticModel {
  using State = pops::StateVec<1>;
  static constexpr int n_vars = 1;

  pops::BackgroundDensity elliptic{};
  POPS_HD pops::Real elliptic_rhs(const State& state) const { return elliptic.rhs(state); }
};

struct ChargeModel {
  using State = pops::StateVec<1>;
  static constexpr int n_vars = 1;
  pops::ChargeDensity elliptic;
  POPS_HD pops::Real elliptic_rhs(const State& state) const { return elliptic.rhs(state); }
};

struct GravityModel {
  using State = pops::StateVec<1>;
  static constexpr int n_vars = 1;
  pops::GravityCoupling elliptic;
  POPS_HD pops::Real elliptic_rhs(const State& state) const { return elliptic.rhs(state); }
};

template <int Dim>
void expect_composite_rhs_assembly() {
  using Field = pops::MultiFab<Dim>;
  using Layout = pops::mesh::BoxArray<Dim>;

  const pops::Box<Dim> domain{pops::Index<Dim>{}, ranked<pops::Index<Dim>, Dim>(7)};
  const Layout layout(std::vector<pops::Box<Dim>>{domain});
  const pops::mesh::RankSpace<Dim> ranks{pops::Index<Dim>{},
                                         ranked<pops::Extent<Dim>, Dim>(std::int64_t{1})};
  const auto distribution = pops::mesh::Distribution<Dim>::replicated(layout, ranks);
  Field first(layout, distribution, pops::Index<Dim>{}, 1, pops::Extent<Dim>{});
  Field second(layout, distribution, pops::Index<Dim>{}, 1, pops::Extent<Dim>{});
  Field rhs(layout, distribution, pops::Index<Dim>{}, 1, pops::Extent<Dim>{});

  auto first_host = first.fab(0).create_host_mirror();
  auto second_host = second.fab(0).create_host_mirror();
  for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(domain.numPts()); ++ordinal) {
    const auto index = index_from_ordinal(domain, ordinal);
    pops::Real phase = pops::Real(0);
    for (int axis = 0; axis < Dim; ++axis)
      phase += pops::Real(axis + 1) * pops::Real(index[axis] + 1);
    const std::size_t storage = storage_ordinal(domain, index);
    first_host(storage) = pops::Real(1) + pops::Real(0.03) * phase;
    second_host(storage) = pops::Real(0.7) + pops::Real(0.02) * phase * phase;
  }
  first.fab(0).copy_from_host(first_host);
  second.fab(0).copy_from_host(second_host);

  const ScalarEllipticModel<Dim> background{{pops::Real(1.3), pops::Real(0.4), 0}};
  pops::SingleModelEllipticRhs<Dim, ScalarEllipticModel<Dim>> single{background};
  single(first, rhs);

  auto rhs_host = rhs.fab(0).create_host_mirror();
  rhs.fab(0).copy_to_host(rhs_host);
  first.fab(0).copy_to_host(first_host);
  for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(domain.numPts()); ++ordinal) {
    const auto index = index_from_ordinal(domain, ordinal);
    const std::size_t storage = storage_ordinal(domain, index);
    EXPECT_EQ(rhs_host(storage), background.elliptic_rhs({first_host(storage)}));
  }

  const ChargeModel charge{{pops::Real(-0.8), 0}};
  const GravityModel gravity{{pops::Real(-1), pops::Real(2.5), pops::Real(0.6), 0}};
  rhs.set_val(pops::Real(0));
  pops::add_model_elliptic_rhs(charge, first, rhs);
  pops::add_model_elliptic_rhs(gravity, second, rhs);
  rhs.fab(0).copy_to_host(rhs_host);
  second.fab(0).copy_to_host(second_host);

  bool differs_from_charge_only = false;
  for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(domain.numPts()); ++ordinal) {
    const auto index = index_from_ordinal(domain, ordinal);
    const std::size_t storage = storage_ordinal(domain, index);
    pops::Real expected = pops::Real(0);
    expected += charge.elliptic_rhs({first_host(storage)});
    expected += gravity.elliptic_rhs({second_host(storage)});
    EXPECT_EQ(rhs_host(storage), expected);
    differs_from_charge_only =
        differs_from_charge_only ||
        rhs_host(storage) != pops::Real(-0.8) * (first_host(storage) + second_host(storage));
  }
  EXPECT_TRUE(differs_from_charge_only);
}

}  // namespace

TEST(EllipticCompositeRhsTest, exact_ranked_bricks_accumulate_without_formula_specific_adapter) {
  expect_composite_rhs_assembly<1>();
  expect_composite_rhs_assembly<2>();
  expect_composite_rhs_assembly<3>();
}
