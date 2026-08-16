// API memoire explicite : sync_host() / sync_device() (cf. for_each.hpp,
// multifab.hpp). Encodent l'intention de residence des donnees. Sous memoire
// unifiee (Kokkos::SharedSpace) sync_host() est un device_fence() cible et
// sync_device() un no-op ; le comportement doit rester BIT-IDENTIQUE a un acces
// hote nu. Ce test verifie :
//   1) les seams libres pops::sync_host()/sync_device() s'appellent (idempotents,
//      sans effet observable sur les donnees) ;
//   2) les copies explicites Fab::copy_from_host() -> kernel natif -> copy_to_host()
//      preservent la transition de residence sans exposer de vue hote rebindable ;
//   3) une lecture/ecriture hote encadree par sync_host() donne exactement le
//      meme resultat qu'aujourd'hui (set_val + sum inchanges) ;
//   4) sync_device() avant un kernel for_each_cell ne perturbe pas le calcul.

#include <gtest/gtest.h>

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <Kokkos_Core.hpp>

using namespace pops;
using namespace pops::mesh;

// Pipeline stateful : le meme MultiFab est synchronise et relu en plusieurs etapes.
TEST(test_sync_residence, sync_host_device_idempotent_and_bit_exact) {
  // 1) Les seams libres existent et sont des appels surs et repetables. Sous
  // SharedSpace : sync_host == fence cible, sync_device == no-op. Aucun effet
  // observable, on verifie juste qu'ils s'enchainent sans planter.
  sync_host();
  sync_host();
  sync_device();
  sync_device();
  SUCCEED() << "free_seams_callable";

  const Box<2> dom = Box<2>::from_extents(Extent<2>{8, 8});
  const BoxArray<2> ba = BoxArray<2>::from_domain(dom, Extent<2>{4, 4});
  const RankSpace<2> ranks{Index<2>{}, Extent<2>{1, 1}};
  const auto distribution = Distribution<2>::replicated(ba, ranks);
  MultiFab<2> mf(ba, distribution, Index<2>{}, /*ncomp=*/1, /*ghosts=*/Extent<2>{1, 1});

  // 2) Stage host values into the default/native allocation before the device kernel consumes them.
  for (std::size_t local = 0; local < mf.local_size(); ++local) {
    auto& fab = mf.fab(local);
    auto host = fab.create_host_mirror();
    for (int j = fab.box().lo[1]; j <= fab.box().hi[1]; ++j)
      for (int i = fab.box().lo[0]; i <= fab.box().hi[0]; ++i) {
        const std::size_t ordinal = static_cast<std::size_t>(i - fab.grown_box().lo[0]) +
                                    static_cast<std::size_t>(j - fab.grown_box().lo[1]) *
                                        static_cast<std::size_t>(fab.grown_box().length(0));
        host(ordinal) = Real(3);
      }
    fab.copy_from_host(host);
  }
  auto sum_valid = [&] {
    Real result = 0;
    for (std::size_t local = 0; local < mf.local_size(); ++local) {
      const auto& fab = mf.fab(local);
      auto host = fab.create_host_mirror();
      fab.copy_to_host(host);
      for (int j = fab.box().lo[1]; j <= fab.box().hi[1]; ++j)
        for (int i = fab.box().lo[0]; i <= fab.box().hi[0]; ++i) {
          const std::size_t ordinal = static_cast<std::size_t>(i - fab.grown_box().lo[0]) +
                                      static_cast<std::size_t>(j - fab.grown_box().lo[1]) *
                                          static_cast<std::size_t>(fab.grown_box().length(0));
          result += host(ordinal);
        }
    }
    return result;
  };
  const Real s0 = sum_valid();
  EXPECT_EQ(s0, 3.0 * 64) << "set_val_sum_exact";

  // 3) IDEMPOTENCE : sync_host()/sync_device() repetes ne touchent aucune
  // donnee. La somme apres N sync est BIT-IDENTIQUE (==, pas une tolerance).
  sync_host();
  sync_device();
  sync_host();
  const Real s1 = sum_valid();
  EXPECT_EQ(s1, s0) << "sync_idempotent_sum";

  // Host staging is complete; the selected native/default memory space now executes the kernel.
  sync_device();  // intention : un kernel va ecrire (no-op sous unifiee)
  for (int li = 0; li < mf.local_size(); ++li) {
    auto field = mf.fab(static_cast<std::size_t>(li)).view();
    for_each_cell(
        Kokkos::DefaultExecutionSpace{}, mf.box(static_cast<std::size_t>(li)),
        [field] POPS_HD(const Index<2>& index) { field(index, 0) = index[0] + 100.0 * index[1]; });
  }
  sync_host();  // intention : copy_to_host va relire les resultats du kernel

  Real expected = 0;
  for (int j = 0; j < 8; ++j)
    for (int i = 0; i < 8; ++i)
      expected += i + 100.0 * j;
  const Real sf = sum_valid();
  EXPECT_EQ(sf, expected) << "field_after_sync_exact";

  // 4) re-sync apres lecture : toujours bit-identique (aucune migration).
  sync_host();
  EXPECT_EQ(sum_valid(), sf) << "resync_no_drift";

  // une cellule precise : la valeur lue cote hote apres sync_host() est exacte.
  bool found = false;
  for (int li = 0; li < mf.local_size(); ++li) {
    if (mf.box(static_cast<std::size_t>(li)).contains(Index<2>{5, 6})) {
      found = true;
      sync_host();
      const auto& fab = mf.fab(static_cast<std::size_t>(li));
      auto host = fab.create_host_mirror();
      fab.copy_to_host(host);
      const std::size_t ordinal = static_cast<std::size_t>(5 - fab.grown_box().lo[0]) +
                                  static_cast<std::size_t>(6 - fab.grown_box().lo[1]) *
                                      static_cast<std::size_t>(fab.grown_box().length(0));
      EXPECT_EQ(host(ordinal), 5 + 600.0) << "cell_value_after_sync";
    }
  }
  EXPECT_TRUE(found) << "cell_located";
}
