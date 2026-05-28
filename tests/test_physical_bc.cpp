// CL physiques : Foextrap (gradient nul), Dirichlet (reflexion autour de la
// valeur de face), et remplissage correct des coins par composition
// faces-x puis faces-y.

#include <adc/mesh/box_array.hpp>
#include <adc/mesh/distribution_mapping.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/for_each.hpp>
#include <adc/mesh/multifab.hpp>
#include <adc/mesh/physical_bc.hpp>

#include <cstdio>

using namespace adc;

static double g(int i, int j) { return i + 10.0 * j; }

int main() {
  int fails = 0;
  auto chk = [&](bool c, const char* w) {
    if (!c) {
      std::printf("FAIL %s\n", w);
      ++fails;
    }
  };

  Box2D dom = Box2D::from_extents(4, 4);  // [0..3] x [0..3]
  BoxArray ba = BoxArray::from_domain(dom, 4);
  MultiFab mf(ba, DistributionMapping(ba.size(), n_ranks()), 1, 1);

  Array4 a = mf.fab(0).array();
  for_each_cell(dom, [a](int i, int j) { a(i, j, 0) = g(i, j); });

  // melange : xlo foextrap, xhi dirichlet(0), ylo foextrap, yhi dirichlet(0)
  BCRec bc;
  bc.xlo = BCType::Foextrap;
  bc.xhi = BCType::Dirichlet;
  bc.xhi_val = 0.0;
  bc.ylo = BCType::Foextrap;
  bc.yhi = BCType::Dirichlet;
  bc.yhi_val = 0.0;

  fill_physical_bc(mf, dom, bc);
  const Fab2D& f = mf.fab(0);

  // foextrap xlo : ghost = cellule interne la plus proche
  chk(f(-1, 2, 0) == g(0, 2), "foextrap_xlo");
  // dirichlet xhi val 0 : ghost = -interne miroir
  chk(f(4, 2, 0) == -g(3, 2), "dirichlet_xhi");
  // foextrap ylo
  chk(f(2, -1, 0) == g(2, 0), "foextrap_ylo");
  // dirichlet yhi
  chk(f(2, 4, 0) == -g(2, 3), "dirichlet_yhi");

  // coin bas-gauche : face-y foextrap sur i etendu, lit le ghost-x deja rempli
  // ghost(-1,-1) = a(-1, 0) = g(0,0)
  chk(f(-1, -1, 0) == g(0, 0), "corner_foextrap_foextrap");
  // coin bas-droit : ylo foextrap lit le ghost xhi dirichlet a i=4
  // ghost(4,-1) = a(4,0) = -g(3,0)
  chk(f(4, -1, 0) == -g(3, 0), "corner_dirichlet_foextrap");

  if (fails == 0) std::printf("OK test_physical_bc\n");
  return fails == 0 ? 0 : 1;
}
