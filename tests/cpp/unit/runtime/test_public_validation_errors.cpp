#include <gtest/gtest.h>

#include <pops/coupling/source/coupled_source_program.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/layout/refinement.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/linear/krylov_solver.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/parallel/comm.hpp>

#include <exception>
#include <string>
#include <vector>

namespace {

// Verifie que @p f leve et que le message porte TOUS les @p needles (contrat de message d'erreur,
// pas seulement le type de l'exception -- raison pour laquelle on garde ce helper plutot qu'un
// EXPECT_THROW nu, qui ne verifierait pas le contenu).
template <class F>
::testing::AssertionResult ThrowsWithMessage(F&& f, const std::vector<std::string>& needles) {
  try {
    f();
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    for (const std::string& needle : needles) {
      if (msg.find(needle) == std::string::npos) {
        return ::testing::AssertionFailure()
               << "message missing '" << needle << "': " << msg;
      }
    }
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << "expected an exception, none was thrown";
}

}  // namespace

using namespace pops;

TEST(PublicValidationErrors, Fab2DRejectsZeroComponents) {
  const Box2D valid = Box2D::from_extents(2, 2);
  EXPECT_TRUE(ThrowsWithMessage([&] { Fab2D bad(valid, /*ncomp=*/0, /*ng=*/0); },
                                {"pops validation error", "Fab2D", "ncomp >= 1", "ncomp=0"}))
      << "Fab2D rejects zero components in release";
}

TEST(PublicValidationErrors, Fab2DRejectsNegativeGhostWidth) {
  const Box2D valid = Box2D::from_extents(2, 2);
  EXPECT_TRUE(ThrowsWithMessage(
      [&] { Fab2D bad(valid, /*ncomp=*/1, /*ng=*/-1); },
      {"pops validation error", "Fab2D", "ghost width ng >= 0", "ng=-1"}))
      << "Fab2D rejects negative ghost width in release";
}

TEST(PublicValidationErrors, Fab2DHostAccessRejectsOutOfBoundsIndex) {
  const Box2D valid = Box2D::from_extents(2, 2);
  EXPECT_TRUE(ThrowsWithMessage(
      [&] {
        Fab2D fab(valid, /*ncomp=*/1, /*ng=*/0);
        (void)fab(2, 0, 0);
      },
      {"pops/mesh/storage/fab2d.hpp", "Fab2D::operator()", "expected", "received", "i=2"}))
      << "Fab2D host access rejects out-of-bounds index in release";
}

TEST(PublicValidationErrors, MultiFabRejectsZeroComponents) {
  const Box2D valid = Box2D::from_extents(2, 2);
  const BoxArray ba(std::vector<Box2D>{valid});
  const DistributionMapping dm(ba.size(), n_ranks());
  EXPECT_TRUE(ThrowsWithMessage([&] { MultiFab bad(ba, dm, /*ncomp=*/0, /*ngrow=*/0); },
                                {"MultiFab", "ncomp >= 1", "ncomp=0"}))
      << "MultiFab rejects zero components in release";
}

TEST(PublicValidationErrors, MultiFabRejectsNegativeGhostWidth) {
  const Box2D valid = Box2D::from_extents(2, 2);
  const BoxArray ba(std::vector<Box2D>{valid});
  const DistributionMapping dm(ba.size(), n_ranks());
  EXPECT_TRUE(ThrowsWithMessage([&] { MultiFab bad(ba, dm, /*ncomp=*/1, /*ngrow=*/-2); },
                                {"MultiFab", "ghost width ngrow >= 0", "ngrow=-2"}))
      << "MultiFab rejects negative ghost width in release";
}

TEST(PublicValidationErrors, MultiFabRejectsDistributionMappingSizeMismatch) {
  EXPECT_TRUE(ThrowsWithMessage(
      [&] {
        const BoxArray two_boxes(
            std::vector<Box2D>{Box2D{{0, 0}, {0, 0}}, Box2D{{1, 0}, {1, 0}}});
        const DistributionMapping short_dm(std::vector<int>{0});
        MultiFab bad(two_boxes, short_dm, /*ncomp=*/1, /*ngrow=*/0);
      },
      {"MultiFab", "DistributionMapping size equals BoxArray size", "box_array.size=2",
       "dmap.size=1"}))
      << "MultiFab rejects dmap/box-array size mismatch in release";
}

TEST(PublicValidationErrors, MultiFabRejectsInvalidOwnerRank) {
  const Box2D valid = Box2D::from_extents(2, 2);
  const BoxArray ba(std::vector<Box2D>{valid});
  EXPECT_TRUE(ThrowsWithMessage(
      [&] {
        const DistributionMapping bad_owner(std::vector<int>{n_ranks()});
        MultiFab bad(ba, bad_owner, /*ncomp=*/1, /*ngrow=*/0);
      },
      {"MultiFab", "owner rank", "n_ranks=" + std::to_string(n_ranks())}))
      << "MultiFab rejects invalid owner rank in release";
}

TEST(PublicValidationErrors, MultiFabRejectsInvalidLocalFabIndex) {
  const Box2D valid = Box2D::from_extents(2, 2);
  const BoxArray ba(std::vector<Box2D>{valid});
  const DistributionMapping dm(ba.size(), n_ranks());
  EXPECT_TRUE(ThrowsWithMessage(
      [&] {
        MultiFab mf(ba, dm, /*ncomp=*/1, /*ngrow=*/0);
        (void)mf.fab(1);
      },
      {"MultiFab::fab", "local index", "li=1"}))
      << "MultiFab rejects invalid local fab index in release";
}

TEST(PublicValidationErrors, CoarsenRejectsZeroRatio) {
  const Box2D fdom = Box2D::from_extents(4, 4);
  const BoxArray fba = BoxArray::from_domain(fdom, 4);
  EXPECT_TRUE(ThrowsWithMessage([&] { (void)coarsen(fba, 0); },
                                {"refinement.hpp: coarsen", "ratio r >= 1", "r=0"}))
      << "coarsen rejects zero ratio in release";
}

TEST(PublicValidationErrors, AverageDownRejectsWrongScratchLayout) {
  const Box2D cdom = Box2D::from_extents(2, 2);
  const Box2D fdom = Box2D::from_extents(4, 4);
  const BoxArray cba = BoxArray::from_domain(cdom, 2);
  const BoxArray fba = BoxArray::from_domain(fdom, 4);
  const DistributionMapping cdm(cba.size(), n_ranks());
  const DistributionMapping fdm(fba.size(), n_ranks());
  EXPECT_TRUE(ThrowsWithMessage(
      [&] {
        MultiFab fine(fba, fdm, /*ncomp=*/1, /*ngrow=*/0);
        MultiFab coarse(cba, cdm, /*ncomp=*/1, /*ngrow=*/0);
        MultiFab wrong_scratch(fba, fdm, /*ncomp=*/1, /*ngrow=*/0);
        average_down(fine, coarse, 2, wrong_scratch);
      },
      {"average_down(scratch)", "scratch MultiFab layout", "scratch.boxes=1"}))
      << "average_down rejects wrong scratch layout in release";
}

TEST(PublicValidationErrors, CsProgramStackValidationRejectsUnderflow) {
  EXPECT_TRUE(ThrowsWithMessage(
      [&] {
        CsProgram pg;
        pg.len = 1;
        pg.op[0] = static_cast<int>(CsOp::Add);
        validate_cs_program_stack(pg, "test CsProgram");
      },
      {"test CsProgram", "well-formed postfix stack program", "stack underflow"}))
      << "CsProgram stack validation rejects underflow in release";
}

TEST(PublicValidationErrors, CsProgramStackValidationRejectsUnusedResult) {
  EXPECT_TRUE(ThrowsWithMessage(
      [&] {
        CsProgram pg;
        pg.len = 2;
        pg.op[0] = static_cast<int>(CsOp::PushReg);
        pg.op[1] = static_cast<int>(CsOp::PushReg);
        validate_cs_program_stack(pg, "test CsProgram");
      },
      {"test CsProgram", "exactly one result", "final stack_depth=2"}))
      << "CsProgram stack validation rejects unused result in release";
}

TEST(PublicValidationErrors, TensorKrylovSolverRejectsAliasedOperatorAndPreconditioner) {
  EXPECT_TRUE(ThrowsWithMessage(
      [&] {
        Geometry geom{Box2D::from_extents(4, 4), 0.0, 1.0, 0.0, 1.0};
        BCRec bc;
        GeometricMG mg(geom, BoxArray::from_domain(geom.domain, 4), bc);
        TensorKrylovSolver solver(mg, mg);
        (void)solver;
      },
      {"TensorKrylovSolver", "op and precond are distinct", "alias"}))
      << "TensorKrylovSolver rejects aliased operator/preconditioner in release";
}
