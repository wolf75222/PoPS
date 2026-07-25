// Echange de halos distribue (lance via mpirun -np N). On remplit les cellules
// valides avec une valeur ne dependant QUE des coordonnees globales repliees
// (periodiques) : val(i,j,c) = wrap(i) + 0.001*wrap(j) + 100*c. Apres
// fill_boundary periodique, chaque cellule fantome doit valoir la meme chose
// (sa source periodique a la meme valeur repliee). Avec np>1, ces ghosts
// proviennent de fabs distants : le test valide donc le transfert cross-rang.
//
// Invariant au nombre de rangs : reussit en serie (np=1, chemin local) comme en
// distribue (np>1, chemin MPI). Couvre aussi les coins (shifts diagonaux).

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/load_balance.hpp>

#include <cmath>
#include <cstdio>
#include <new>
#include <stdexcept>

using namespace pops;

static int pops_run_test_mpi_fillboundary(int argc, char** argv) {
  comm_init(&argc, &argv);
  const int me = my_rank(), np = n_ranks();
  long fails = 0;

  const int L = 64, ng = 1, ncomp = 2;
  Box2D dom = Box2D::from_extents(L, L);
  auto wrap = [&](int x) { return ((x % L) + L) % L; };
  auto val = [&](int i, int j, int c) {
    return double(wrap(i)) + 0.001 * double(wrap(j)) + 100.0 * c;
  };

  BoxArray ba = BoxArray::from_domain(dom, 16);  // 4x4 = 16 boxes
  DistributionMapping dm = make_sfc_distribution(ba, np);
  MultiFab mf(ba, dm, ncomp, ng);

  // remplir les cellules valides locales
  for (int li = 0; li < mf.local_size(); ++li) {
    Fab2D& F = mf.fab(li);
    const Box2D b = F.box();
    for (int c = 0; c < ncomp; ++c)
      for (int j = b.lo[1]; j <= b.hi[1]; ++j)
        for (int i = b.lo[0]; i <= b.hi[0]; ++i)
          F(i, j, c) = val(i, j, c);
  }

  fill_boundary(mf, dom, Periodicity{true, true});

  // verifier toutes les cellules (valides + fantomes) contre la valeur repliee
  for (int li = 0; li < mf.local_size(); ++li) {
    const Fab2D& F = mf.fab(li);
    const Box2D g = F.box().grow(ng);
    for (int c = 0; c < ncomp; ++c)
      for (int j = g.lo[1]; j <= g.hi[1]; ++j)
        for (int i = g.lo[0]; i <= g.hi[0]; ++i)
            if (std::fabs(F(i, j, c) - val(i, j, c)) > 1e-12)
              ++fails;
  }

  // The halo is deeper than both complete periodic extents, including a one-cell x axis. With four
  // one-cell boxes this also exercises repeated periodic images across MPI ranks and deep corners.
  {
    constexpr int deep_ng = 6;
    const Box2D deep_dom = Box2D::from_extents(1, 4);
    const BoxArray deep_ba = BoxArray::from_domain(deep_dom, 1);
    const DistributionMapping deep_dm = make_sfc_distribution(deep_ba, np);
    MultiFab deep(deep_ba, deep_dm, ncomp, deep_ng);
    auto deep_wrap = [](int x, int extent) { return ((x % extent) + extent) % extent; };
    auto deep_val = [&](int i, int j, int c) {
      return 11.0 * deep_wrap(i, deep_dom.nx()) + deep_wrap(j, deep_dom.ny()) + 100.0 * c;
    };
    for (int li = 0; li < deep.local_size(); ++li) {
      Fab2D& F = deep.fab(li);
      const Box2D b = F.box();
      for (int c = 0; c < ncomp; ++c)
        for (int j = b.lo[1]; j <= b.hi[1]; ++j)
          for (int i = b.lo[0]; i <= b.hi[0]; ++i)
            F(i, j, c) = deep_val(i, j, c);
    }
    fill_boundary(deep, deep_dom, Periodicity{true, true});
    for (int li = 0; li < deep.local_size(); ++li) {
      const Fab2D& F = deep.fab(li);
      const Box2D grown = F.box().grow(deep_ng);
      for (int c = 0; c < ncomp; ++c)
        for (int j = grown.lo[1]; j <= grown.hi[1]; ++j)
          for (int i = grown.lo[0]; i <= grown.hi[0]; ++i)
            if (std::fabs(F(i, j, c) - deep_val(i, j, c)) > 1e-12)
              ++fails;
    }
  }

  // Schedule construction itself is rank-local and may reject before buffer sizing.  Give only the
  // last rank an invalid periodic extent: every peer must receive the collective rejection, nobody
  // may enter a later collective/post, and no local ghost copy may become visible.
  {
    constexpr Real untouched = Real(-901.25);
    MultiFab rejected(ba, dm, ncomp, ng);
    rejected.set_val(untouched);
    for (int li = 0; li < rejected.local_size(); ++li) {
      Fab2D& field = rejected.fab(li);
      const Box2D valid = field.box();
      for (int c = 0; c < ncomp; ++c)
        for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
          for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
            field(i, j, c) = Real(10 + me + c);
    }
    const Box2D rank_domain = me == np - 1 ? Box2D{{0, 0}, {-1, L - 1}} : dom;
    bool rejected_everywhere = false;
    try {
      HaloExchange exchange = fill_boundary_begin(rejected, rank_domain, Periodicity{true, true});
      fill_boundary_end(rejected, exchange);
    } catch (const std::runtime_error&) {
      rejected_everywhere = true;
    } catch (...) {
    }
    if (!rejected_everywhere)
      ++fails;
    device_fence();
    for (int li = 0; li < rejected.local_size(); ++li) {
      const Fab2D& field = rejected.fab(li);
      const Box2D valid = field.box();
      const Box2D grown = field.grown_box();
      for (int c = 0; c < ncomp; ++c)
        for (int j = grown.lo[1]; j <= grown.hi[1]; ++j)
          for (int i = grown.lo[0]; i <= grown.hi[0]; ++i)
            if (!valid.contains(i, j) && field(i, j, c) != untouched)
              ++fails;
    }
  }

  // A rank-local allocation/preparation failure must be observed by every peer before any
  // point-to-point request can be posted.  Exercise the same collective guard used by the real
  // pinned-buffer allocation path, with the last rank as the sole failing participant.
  {
    bool rejected_everywhere = false;
    try {
      detail::collectively_prepare_before_halo_post(world_communicator_view(), [&] {
        if (me == np - 1)
          throw std::bad_alloc();
      });
    } catch (const std::bad_alloc&) {
      rejected_everywhere = true;
    } catch (...) {
    }
    if (!rejected_everywhere)
      ++fails;
  }

  {
    bool rejected_everywhere = false;
    try {
      detail::collectively_prepare_before_halo_post(world_communicator_view(), [&] {
        if (me == np - 1)
          throw std::runtime_error("injected halo preparation failure");
      });
    } catch (const std::runtime_error&) {
      rejected_everywhere = true;
    } catch (...) {
    }
    if (!rejected_everywhere)
      ++fails;
  }

  const long gfails = all_reduce_sum(fails);
  if (me == 0) {
    if (gfails == 0)
      std::printf("OK test_mpi_fillboundary (np=%d, boxes=%d)\n", np, ba.size());
    else
      std::printf("FAIL test_mpi_fillboundary : %ld cellules fausses (np=%d)\n", gfails, np);
  }
  comm_finalize();
  return gfails == 0 ? 0 : 1;
}

TEST(test_mpi_fillboundary, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_mpi_fillboundary, "test_mpi_fillboundary"), 0);
}
