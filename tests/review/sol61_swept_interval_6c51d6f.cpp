#include "swept_interval-6c51d6f.hpp"
#include <array>
#include <cassert>
#include <iostream>

using pops::Real;
template<class F> void rejected(F operation) {
  bool caught = false;
  try { operation(); } catch (const std::invalid_argument&) { caught = true; }
  assert(caught);
}
int main() {
  constexpr Real eps = 64*std::numeric_limits<Real>::epsilon();
  // Both face directions, expansion, contraction, and pure translation.
  for (const auto displacement : {std::array<Real,2>{-.05,.1},
                                  std::array<Real,2>{.1,-.05},
                                  std::array<Real,2>{.1,.1},
                                  std::array<Real,2>{-.1,-.1}}) {
    auto g = pops::SweptInterval::prepare(0,.5,displacement[0],.5+displacement[1],
                                         displacement[0],displacement[1],eps);
    assert(std::abs(g.gcl_residual()) < eps);
    Real amount = g.updated_amount(2*g.old_measure(),2,2,.3,.3,0);
    assert(std::abs(amount/g.new_measure()-2) < 4*eps);
  }
  // A shared face cancels globally using one face density and one orientation.
  const Real old[] = {0,.25,.7,1};
  const Real shift[] = {0,.04,-.03,0};
  Real inventory = 0;
  for (int cell=0; cell<3; ++cell) {
    auto g = pops::SweptInterval::prepare(old[cell],old[cell+1],
        old[cell]+shift[cell],old[cell+1]+shift[cell+1],shift[cell],shift[cell+1],eps);
    inventory += g.updated_amount(3*g.old_measure(),3,3,.2,.2,0);
  }
  assert(std::abs(inventory-3) < 8*eps);
  // Q=1; physical net=.3; mesh amount=.3; integrated source=.2.
  auto g = pops::SweptInterval::prepare(0,.5,0,.6,0,.1,eps);
  assert(std::abs(g.updated_amount(1,2,3,.1,.4,.2)-1.2) < 4*eps);
  rejected([&]{pops::SweptInterval::prepare(0,.5,0,.6,0,.05,eps);});
  rejected([&]{pops::SweptInterval::prepare(0,1,0,0,0,-1,eps);});
  rejected([&]{pops::SweptInterval::prepare(-1e308,1e308,0,1,0,0,eps);});
  for (Real bad : {std::numeric_limits<Real>::quiet_NaN(),
                   std::numeric_limits<Real>::infinity(),
                   -std::numeric_limits<Real>::infinity()}) {
    rejected([&]{pops::SweptInterval::prepare(bad,1,0,1,0,0,eps);});
    rejected([&]{pops::SweptInterval::prepare(0,1,0,1,bad,0,eps);});
    rejected([&]{pops::SweptInterval::prepare(0,1,0,1,0,0,bad);});
  }
  assert(std::isnan(g.updated_amount(std::numeric_limits<Real>::quiet_NaN(),2,3,0,0,0)));
  std::cout << "ALE host arithmetic/GCL/orientation/amount/rejection assertions passed\n";
}
