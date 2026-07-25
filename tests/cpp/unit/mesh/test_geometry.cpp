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

TEST(test_geometry, nonzero_index_origin_preserves_physical_extent) {
  const Box2D domain{{-3, 7}, {0, 8}};
  Geometry g{domain, -2.0, 2.0, 10.0, 12.0};

  EXPECT_TRUE(close(g.dx(), 1.0));
  EXPECT_TRUE(close(g.dy(), 1.0));
  EXPECT_TRUE(close(g.x_cell(-3), -1.5));
  EXPECT_TRUE(close(g.x_cell(0), 1.5));
  EXPECT_TRUE(close(g.x_cell(-4), -2.5));
  EXPECT_TRUE(close(g.y_cell(7), 10.5));
  EXPECT_TRUE(close(g.y_cell(8), 11.5));

  const Geometry gf = g.refine(2);
  EXPECT_TRUE(close(gf.x_cell(-6), -1.75));
  EXPECT_TRUE(close(gf.x_cell(1), 1.75));
  EXPECT_TRUE(close(gf.y_cell(14), 10.25));
}

TEST(test_geometry, polar_nonzero_index_origin_preserves_physical_extent) {
  const Box2D domain{{4, -2}, {7, 5}};
  PolarGeometry g{domain, 2.0, 6.0};

  EXPECT_TRUE(close(g.dr(), 1.0));
  EXPECT_TRUE(close(g.r_face(4), 2.0));
  EXPECT_TRUE(close(g.r_face(8), 6.0));
  EXPECT_TRUE(close(g.r_cell(4), 2.5));
  EXPECT_TRUE(close(g.theta_face(-2), 0.0));
  EXPECT_TRUE(close(g.theta_cell(-2), Real(0.5) * g.dtheta()));

  const PolarGeometry gf = g.refine(2);
  EXPECT_TRUE(close(gf.r_face(8), 2.0));
  EXPECT_TRUE(close(gf.r_cell(8), 2.25));
  EXPECT_TRUE(close(gf.theta_face(-4), 0.0));
}
