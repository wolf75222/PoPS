// Geometry : pas d'espace dx/dy, centres de cellule, raffinement.

#include <gtest/gtest.h>

#include <pops/mesh/index/box2d.hpp>
#include <pops/mesh/geometry/geometry.hpp>

#include <cmath>

using namespace pops;

namespace {
bool close(Real x, Real y) {
  return std::fabs(x - y) < 1e-12;
}
}  // namespace

TEST(test_geometry, spacing_and_cell_centers) {
  Geometry g{Box2D::from_extents(4, 2), 0.0, 1.0, 0.0, 1.0};
  EXPECT_TRUE(close(g.dx(), 0.25)) << "dx";
  EXPECT_TRUE(close(g.dy(), 0.5)) << "dy";
  EXPECT_TRUE(close(g.x_cell(0), 0.125)) << "x_cell0";
  EXPECT_TRUE(close(g.x_cell(3), 0.875)) << "x_cell3";
  EXPECT_TRUE(close(g.x_cell(-1), -0.125)) << "x_cell_ghost";
  EXPECT_TRUE(close(g.y_cell(0), 0.25)) << "y_cell0";
}

TEST(test_geometry, refine) {
  Geometry g{Box2D::from_extents(4, 2), 0.0, 1.0, 0.0, 1.0};
  Geometry gf = g.refine(2);
  EXPECT_EQ(gf.domain, Box2D::from_extents(8, 4)) << "refine_domain";
  EXPECT_TRUE(close(gf.dx(), 0.125)) << "refine_dx";
  EXPECT_TRUE(close(gf.xhi, 1.0)) << "refine_extent";
}
