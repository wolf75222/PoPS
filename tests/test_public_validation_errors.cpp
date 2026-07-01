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

#include <cstdio>
#include <exception>
#include <string>
#include <vector>

namespace {
template <class F>
bool throws_with(F&& f, const std::vector<std::string>& needles) {
  try {
    f();
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    for (const std::string& needle : needles) {
      if (msg.find(needle) == std::string::npos) {
        std::printf("  message missing '%s': %s\n", needle.c_str(), msg.c_str());
        return false;
      }
    }
    return true;
  }
  return false;
}

void chk(bool cond, const char* label, int& fails) {
  std::printf("%s %s\n", cond ? "OK" : "FAIL", label);
  if (!cond)
    ++fails;
}
}  // namespace

int main() {
  using namespace pops;
  int fails = 0;

  const Box2D valid = Box2D::from_extents(2, 2);
  const BoxArray ba(std::vector<Box2D>{valid});
  const DistributionMapping dm(ba.size(), n_ranks());

  chk(throws_with([&] { Fab2D bad(valid, /*ncomp=*/0, /*ng=*/0); },
                  {"pops validation error", "Fab2D", "ncomp >= 1", "ncomp=0"}),
      "Fab2D rejects zero components in release", fails);

  chk(throws_with([&] { Fab2D bad(valid, /*ncomp=*/1, /*ng=*/-1); },
                  {"pops validation error", "Fab2D", "ghost width ng >= 0", "ng=-1"}),
      "Fab2D rejects negative ghost width in release", fails);

  chk(throws_with(
          [&] {
            Fab2D fab(valid, /*ncomp=*/1, /*ng=*/0);
            (void)fab(2, 0, 0);
          },
          {"pops/mesh/storage/fab2d.hpp", "Fab2D::operator()", "expected", "received",
           "i=2"}),
      "Fab2D host access rejects out-of-bounds index in release", fails);

  chk(throws_with([&] { MultiFab bad(ba, dm, /*ncomp=*/0, /*ngrow=*/0); },
                  {"MultiFab", "ncomp >= 1", "ncomp=0"}),
      "MultiFab rejects zero components in release", fails);

  chk(throws_with([&] { MultiFab bad(ba, dm, /*ncomp=*/1, /*ngrow=*/-2); },
                  {"MultiFab", "ghost width ngrow >= 0", "ngrow=-2"}),
      "MultiFab rejects negative ghost width in release", fails);

  chk(throws_with(
          [&] {
            const BoxArray two_boxes(
                std::vector<Box2D>{Box2D{{0, 0}, {0, 0}}, Box2D{{1, 0}, {1, 0}}});
            const DistributionMapping short_dm(std::vector<int>{0});
            MultiFab bad(two_boxes, short_dm, /*ncomp=*/1, /*ngrow=*/0);
          },
          {"MultiFab", "DistributionMapping size equals BoxArray size", "box_array.size=2",
           "dmap.size=1"}),
      "MultiFab rejects dmap/box-array size mismatch in release", fails);

  chk(throws_with(
          [&] {
            const DistributionMapping bad_owner(std::vector<int>{n_ranks()});
            MultiFab bad(ba, bad_owner, /*ncomp=*/1, /*ngrow=*/0);
          },
          {"MultiFab", "owner rank", "n_ranks=" + std::to_string(n_ranks())}),
      "MultiFab rejects invalid owner rank in release", fails);

  chk(throws_with(
          [&] {
            MultiFab mf(ba, dm, /*ncomp=*/1, /*ngrow=*/0);
            (void)mf.fab(1);
          },
          {"MultiFab::fab", "local index", "li=1"}),
      "MultiFab rejects invalid local fab index in release", fails);

  const Box2D cdom = Box2D::from_extents(2, 2);
  const Box2D fdom = Box2D::from_extents(4, 4);
  const BoxArray cba = BoxArray::from_domain(cdom, 2);
  const BoxArray fba = BoxArray::from_domain(fdom, 4);
  const DistributionMapping cdm(cba.size(), n_ranks());
  const DistributionMapping fdm(fba.size(), n_ranks());

  chk(throws_with([&] { (void)coarsen(fba, 0); },
                  {"refinement.hpp: coarsen", "ratio r >= 1", "r=0"}),
      "coarsen rejects zero ratio in release", fails);

  chk(throws_with(
          [&] {
            MultiFab fine(fba, fdm, /*ncomp=*/1, /*ngrow=*/0);
            MultiFab coarse(cba, cdm, /*ncomp=*/1, /*ngrow=*/0);
            MultiFab wrong_scratch(fba, fdm, /*ncomp=*/1, /*ngrow=*/0);
            average_down(fine, coarse, 2, wrong_scratch);
          },
          {"average_down(scratch)", "scratch MultiFab layout", "scratch.boxes=1"}),
      "average_down rejects wrong scratch layout in release", fails);

  chk(throws_with(
          [&] {
            CsProgram pg;
            pg.len = 1;
            pg.op[0] = static_cast<int>(CsOp::Add);
            validate_cs_program_stack(pg, "test CsProgram");
          },
          {"test CsProgram", "well-formed postfix stack program", "stack underflow"}),
      "CsProgram stack validation rejects underflow in release", fails);

  chk(throws_with(
          [&] {
            CsProgram pg;
            pg.len = 2;
            pg.op[0] = static_cast<int>(CsOp::PushReg);
            pg.op[1] = static_cast<int>(CsOp::PushReg);
            validate_cs_program_stack(pg, "test CsProgram");
          },
          {"test CsProgram", "exactly one result", "final stack_depth=2"}),
      "CsProgram stack validation rejects unused result in release", fails);

  chk(throws_with(
          [&] {
            Geometry geom{Box2D::from_extents(4, 4), 0.0, 1.0, 0.0, 1.0};
            BCRec bc;
            GeometricMG mg(geom, BoxArray::from_domain(geom.domain, 4), bc);
            TensorKrylovSolver solver(mg, mg);
            (void)solver;
          },
          {"TensorKrylovSolver", "op and precond are distinct", "alias"}),
      "TensorKrylovSolver rejects aliased operator/preconditioner in release", fails);

  if (fails == 0)
    std::printf("OK test_public_validation_errors\n");
  return fails == 0 ? 0 : 1;
}
