#include <gtest/gtest.h>

#include <pops/core/foundation/allocator.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/runtime/program/source_mask.hpp>

#include <Kokkos_Core.hpp>

#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using pops::Box;
using pops::Extent;
using pops::Index;
using pops::MultiFab;
using pops::MultiFabCapabilities;
using pops::Real;
using pops::mesh::BoxArray;
using pops::mesh::Distribution;
using pops::mesh::RankSpace;
using pops::runtime::program::apply_component_keep_mask;

namespace {

template <int Dim, class MemorySpace>
void expect_all_values(const MultiFab<Dim, MemorySpace>& fields, Real expected) {
  for (std::size_t local = 0; local < fields.local_size(); ++local) {
    const auto& fab = fields.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (std::size_t element = 0; element < host.size(); ++element)
      EXPECT_DOUBLE_EQ(host(element), expected);
  }
}

}  // namespace

TEST(test_multifab, exact_metadata_and_local_global_index_spaces_are_explicit) {
  const BoxArray<2> layout =
      BoxArray<2>::from_domain(Box<2>{Index<2>{-2, 3}, Index<2>{1, 6}}, Extent<2>{2, 2});
  const RankSpace<2> ranks{Index<2>{10, -2}, Extent<2>{2, 2}};
  const Index<2> local_rank{10, -2};
  const auto distribution = Distribution<2>::partitioned(
      layout, ranks, {local_rank, Index<2>{11, -2}, local_rank, Index<2>{11, -1}});

  MultiFab<2, Kokkos::HostSpace> fields(layout, distribution, local_rank, 3, Extent<2>{1, 2});

  EXPECT_EQ(fields.layout(), layout);
  EXPECT_EQ(fields.box_array(), layout);
  EXPECT_EQ(fields.distribution(), distribution);
  EXPECT_EQ(fields.rank_space(), ranks);
  EXPECT_EQ(fields.local_rank(), local_rank);
  EXPECT_EQ(fields.ncomp(), 3);
  EXPECT_EQ(fields.ghosts(), (Extent<2>{1, 2}));
  EXPECT_EQ(fields.local_global_indices(), (std::vector<std::size_t>{0, 2}));
  ASSERT_EQ(fields.local_size(), 2U);
  EXPECT_EQ(fields.global_index(0), 0U);
  EXPECT_EQ(fields.global_index(1), 2U);
  EXPECT_EQ(fields.local_index_of(0), 0U);
  EXPECT_EQ(fields.local_index_of(1), (MultiFab<2, Kokkos::HostSpace>::not_local));
  EXPECT_EQ(fields.local_index_of(2), 1U);
  EXPECT_EQ(fields.fab(1).box(), layout[2]);
  EXPECT_EQ(fields.fab_global(2).box(), layout[2]);
  EXPECT_THROW((void)fields.fab(2), std::out_of_range);
  EXPECT_THROW((void)fields.fab_global(1), std::out_of_range);
  EXPECT_THROW((void)fields.global_index(2), std::out_of_range);
  EXPECT_THROW((void)fields.local_index_of(layout.size()), std::out_of_range);
  // The Program capacity receipt includes copied metadata, not only Kokkos RankSpace payload.
  EXPECT_GT(fields.resident_storage_bytes(), fields.resident_payload_bytes());

  using HostMultiFab = MultiFab<2, Kokkos::HostSpace>;
  EXPECT_THROW((void)HostMultiFab(layout, distribution, local_rank, 0, Extent<2>{}),
               std::invalid_argument);
  EXPECT_THROW((void)HostMultiFab(layout, distribution, local_rank, 1, Extent<2>{1, -1}),
               std::invalid_argument);
}

TEST(test_multifab, set_val_copy_and_move_preserve_deep_1d_and_3d_ownership) {
  const BoxArray<1> line =
      BoxArray<1>::from_domain(Box<1>{Index<1>{-4}, Index<1>{3}}, Extent<1>{3});
  const RankSpace<1> line_ranks{Index<1>{7}, Extent<1>{1}};
  const auto line_distribution = Distribution<1>::replicated(line, line_ranks);
  MultiFab<1> one_dimensional(line, line_distribution, Index<1>{7}, 2, Extent<1>{2});
  one_dimensional.set_val(Real{6.5});
  expect_all_values(one_dimensional, Real{6.5});

  MultiFab<1> copied = one_dimensional;
  ASSERT_EQ(copied.local_size(), one_dimensional.local_size());
  EXPECT_FALSE(copied.shares_storage_with(one_dimensional));
  EXPECT_TRUE(copied.shares_storage_with(copied));
  for (std::size_t local = 0; local < copied.local_size(); ++local)
    EXPECT_NE(copied.fab(local).storage().data(), one_dimensional.fab(local).storage().data());
  copied.set_val(Real{-2});
  expect_all_values(one_dimensional, Real{6.5});
  expect_all_values(copied, Real{-2});

  const BoxArray<3> volume =
      BoxArray<3>::from_domain(Box<3>{Index<3>{-1, 2, 4}, Index<3>{2, 3, 5}}, Extent<3>{2, 1, 2});
  const RankSpace<3> volume_ranks{Index<3>{1, -1, 7}, Extent<3>{2, 1, 1}};
  std::vector<Index<3>> owners(volume.size(), Index<3>{2, -1, 7});
  owners.front() = Index<3>{1, -1, 7};
  const auto volume_distribution =
      Distribution<3>::partitioned(volume, volume_ranks, std::move(owners));
  MultiFab<3, Kokkos::HostSpace> three_dimensional(volume, volume_distribution, Index<3>{1, -1, 7},
                                                   1, Extent<3>{1, 2, 1});

  MultiFab<3, Kokkos::HostSpace> moved(std::move(three_dimensional));
  EXPECT_EQ(three_dimensional.local_size(), 0U);
  EXPECT_TRUE(three_dimensional.layout().empty());
  ASSERT_EQ(moved.local_size(), 1U);
  EXPECT_EQ(moved.global_index(0), 0U);
  EXPECT_EQ(moved.fab(0).ghosts(), (Extent<3>{1, 2, 1}));
  EXPECT_EQ(moved.fab(0).size(), 80U);
}

TEST(test_multifab, communication_is_explicitly_unavailable_and_fails_closed) {
  constexpr MultiFabCapabilities expected{/*local_storage=*/true, /*halo_exchange=*/false,
                                          /*parallel_copy=*/false, /*mpi_exchange=*/false};
  static_assert(MultiFab<1>::capabilities() == expected);
  static_assert(MultiFab<2>::capabilities() == expected);
  static_assert(MultiFab<3>::capabilities() == expected);
  static_assert(std::is_nothrow_move_assignable_v<MultiFab<3>>);

  try {
    MultiFab<2>::require_communication("halo exchange");
    FAIL() << "missing ND communication must fail closed";
  } catch (const std::logic_error& error) {
    EXPECT_NE(std::string(error.what()).find("halo exchange"), std::string::npos);
    EXPECT_NE(std::string(error.what()).find("no halo, parallel-copy, or MPI schedule"),
              std::string::npos);
  }
}

TEST(test_multifab, program_source_mask_reuses_storage_and_refuses_drift_without_clobbering) {
  const BoxArray<1> layout =
      BoxArray<1>::from_domain(Box<1>{Index<1>{0}, Index<1>{3}}, Extent<1>{1});
  const RankSpace<1> ranks{Index<1>{0}, Extent<1>{1}};
  const auto distribution = Distribution<1>::replicated(layout, ranks);
  MultiFab<1, Kokkos::HostSpace> residual(layout, distribution, Index<1>{0}, 3, Extent<1>{});
  residual.set_val(Real(7));

  const auto before = pops::allocation_event_stats();
  apply_component_keep_mask(residual, {0});
  const auto after = pops::allocation_event_stats();
  EXPECT_EQ(after.fab_calls, before.fab_calls);
  EXPECT_EQ(after.fab_bytes, before.fab_bytes);

  ASSERT_GT(residual.local_size(), 0U);
  for (std::size_t local = 0; local < residual.local_size(); ++local) {
    const auto& fab = residual.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const std::size_t component_stride = fab.size() / 3U;
    for (std::size_t cell = 0; cell < component_stride; ++cell) {
      EXPECT_DOUBLE_EQ(host(cell), Real(7));
      EXPECT_DOUBLE_EQ(host(component_stride + cell), Real(0));
      EXPECT_DOUBLE_EQ(host(2U * component_stride + cell), Real(0));
    }
  }

  // Validate every authored component before the first write: a stale mask must leave the
  // already prepared residual entirely intact.
  residual.set_val(Real(11));
  EXPECT_THROW(apply_component_keep_mask(residual, {0, 3}), std::invalid_argument);
  for (std::size_t local = 0; local < residual.local_size(); ++local) {
    const auto& fab = residual.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (std::size_t element = 0; element < host.size(); ++element)
      EXPECT_DOUBLE_EQ(host(element), Real(11));
  }
}
