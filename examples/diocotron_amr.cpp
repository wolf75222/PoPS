// Diocotron sur AMR (couplage decouple) : transport sous-cycle avec reflux sur
// une hierarchie 2 niveaux, Poisson resolu sur la grille grossiere uniforme
// (phi est lisse), aux = grad phi interpole vers le fin.
//
// Couche de cisaillement periodique (bande de charge perturbee mode m) qui
// s'enroule en tourbillons KH/diocotron ; une box fine interieure raffine la
// zone centrale ou les tourbillons se forment.
//
// A chaque pas : average_down (sync grossier), Poisson grossier (multigrille),
// aux = grad phi (grossier + injection vers fin), advance 2-niveaux sous-cycle.
//
// Run : ./build/bin/diocotron_amr /tmp/dio_amr [nc] [nsteps]

#include <adc/elliptic/geometric_mg.hpp>
#include <adc/integrator/amr_reflux.hpp>
#include <adc/mesh/box2d.hpp>
#include <adc/mesh/box_array.hpp>
#include <adc/mesh/fab2d.hpp>
#include <adc/mesh/geometry.hpp>
#include <adc/mesh/physical_bc.hpp>
#include <adc/model/diocotron.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace adc;

static constexpr double kPi = 3.14159265358979323846;

// remplissage periodique multi-composantes d'un Fab2D
static void fill_periodic_mc(Fab2D& F, const Box2D& dom) {
  const int ng = F.n_ghost(), nx = dom.nx(), ny = dom.ny(), nc = F.ncomp();
  for (int c = 0; c < nc; ++c) {
    for (int j = dom.lo[1]; j <= dom.hi[1]; ++j)
      for (int g = 1; g <= ng; ++g) {
        F(dom.lo[0] - g, j, c) = F(dom.hi[0] - g + 1, j, c);
        F(dom.hi[0] + g, j, c) = F(dom.lo[0] + g - 1, j, c);
      }
    for (int i = dom.lo[0] - ng; i <= dom.hi[0] + ng; ++i)
      for (int g = 1; g <= ng; ++g) {
        F(i, dom.lo[1] - g, c) = F(i, dom.hi[1] - g + 1, c);
        F(i, dom.hi[1] + g, c) = F(i, dom.lo[1] + g - 1, c);
      }
  }
}

int main(int argc, char** argv) {
  const std::string out = (argc > 1) ? argv[1] : "dio_amr";
  const int nc = (argc > 2) ? std::atoi(argv[2]) : 128;
  const int nsteps = (argc > 3) ? std::atoi(argv[3]) : 500;
  std::filesystem::create_directories(out);

  Box2D dom = Box2D::from_extents(nc, nc);
  Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const double dxc = geom.dx(), dyc = geom.dy();
  BoxArray ba(std::vector<Box2D>{dom});  // une box grossiere

  // box fine (recalculee dynamiquement par tagging), ratio 2
  int CI0 = nc / 8, CI1 = 7 * nc / 8 - 1;
  int CJ0 = 7 * nc / 16, CJ1 = 9 * nc / 16 - 1;
  Box2D fbox{{2 * CI0, 2 * CJ0}, {2 * CI1 + 1, 2 * CJ1 + 1}};

  Diocotron model;
  model.B0 = 1.0;
  model.alpha = 1.0;
  const double A = 1.0, w = 0.05;
  const int m = 2;  // mode instable (kw < 1) ; m>=4 est stable
  const double eta = 0.02;
  auto ne0 = [&](double x, double y) {
    const double y0 = 0.5 + eta * std::cos(2 * kPi * m * x);
    return 1.0 + A * std::exp(-((y - y0) * (y - y0)) / (w * w));
  };

  Fab2D Uc(dom, 1, 1), Uf(fbox, 1, 1);
  for (int j = -1; j <= nc; ++j)
    for (int i = -1; i <= nc; ++i)
      if (Uc.grown_box().contains(i, j))
        Uc(i, j) = ne0((i + 0.5) * dxc, (j + 0.5) * dyc);
  const double dxf = dxc / 2, dyf = dyc / 2;
  for (int j = fbox.lo[1]; j <= fbox.hi[1]; ++j)
    for (int i = fbox.lo[0]; i <= fbox.hi[0]; ++i)
      Uf(i, j) = ne0((i + 0.5) * dxf, (j + 0.5) * dyf);
  average_down_fab(Uf, Uc, CI0, CI1, CJ0, CJ1);

  double mean = 0;
  for (int j = 0; j < nc; ++j)
    for (int i = 0; i < nc; ++i) mean += Uc(i, j);
  mean /= double(nc) * nc;
  model.n_i0 = mean;

  BCRec bc;  // periodique (Poisson + transport)
  GeometricMG mg(geom, ba, bc);

  Fab2D auxc(dom, 3, 1), auxf(fbox, 3, 1);

  auto compute_aux = [&]() {
    // source de Poisson = alpha (n_e - n_i0) sur le grossier
    Array4 f = mg.rhs().fab(0).array();
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i)
        f(i, j) = model.alpha * (Uc(i, j) - model.n_i0);
    mg.solve(1e-8, 30);
    // aux_c = grad phi (phi a ses ghosts remplis par le solve)
    const ConstArray4 p = mg.phi().fab(0).const_array();
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) {
        auxc(i, j, 0) = p(i, j);
        auxc(i, j, 1) = (p(i + 1, j) - p(i - 1, j)) / (2 * dxc);
        auxc(i, j, 2) = (p(i, j + 1) - p(i, j - 1)) / (2 * dyc);
      }
    fill_periodic_mc(auxc, dom);
    // aux_f : injection depuis le grossier (valides + ghosts)
    const ConstArray4 ac = auxc.const_array();
    Array4 af = auxf.array();
    const Box2D g = auxf.grown_box();
    auto crsn = [](int x) { return x >= 0 ? x / 2 : -((-x + 1) / 2); };
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i)
        for (int c = 0; c < 3; ++c) af(i, j, c) = ac(crsn(i), crsn(j), c);
  };

  auto vmax = [&]() {
    double v = 0;
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i)
        v = std::max(v, std::hypot(auxc(i, j, 1), auxc(i, j, 2)) / model.B0);
    return std::max(v, 1e-12);
  };
  std::ofstream boxes(out + "/boxes.csv");
  boxes << "frame,CI0,CI1,CJ0,CJ1,nc\n";
  auto dump = [&](int frame) {
    char name[64];
    std::snprintf(name, sizeof(name), "/dens_%04d.txt", frame);
    std::ofstream f(out + name);
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) f << Uc(i, j) << (i + 1 < nc ? ' ' : '\n');
    boxes << frame << ',' << CI0 << ',' << CI1 << ',' << CJ0 << ',' << CJ1 << ','
          << nc << '\n';
  };
  auto mass = [&]() {
    double M = 0;
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i) M += Uc(i, j) * dxc * dyc;
    return M;
  };

  // regrid dynamique : la box fine est recalculee comme le bounding box des
  // cellules taguees (densite au-dessus du fond), elargi d'un buffer et clippe a
  // l'interieur. L'etat fin est transfere (ancien fin la ou il existe, sinon
  // injection depuis le grossier synchronise).
  auto regrid = [&]() {
    int i0 = nc, i1 = -1, j0 = nc, j1 = -1;
    for (int j = 0; j < nc; ++j)
      for (int i = 0; i < nc; ++i)
        if (Uc(i, j) > model.n_i0 + 0.12) {
          i0 = std::min(i0, i);
          i1 = std::max(i1, i);
          j0 = std::min(j0, j);
          j1 = std::max(j1, j);
        }
    if (i1 < i0) return;
    const int buf = 4;
    const int nCI0 = std::max(2, i0 - buf), nCI1 = std::min(nc - 3, i1 + buf);
    const int nCJ0 = std::max(2, j0 - buf), nCJ1 = std::min(nc - 3, j1 + buf);
    if (nCI0 == CI0 && nCI1 == CI1 && nCJ0 == CJ0 && nCJ1 == CJ1) return;
    Box2D nf{{2 * nCI0, 2 * nCJ0}, {2 * nCI1 + 1, 2 * nCJ1 + 1}};
    Fab2D Ufn(nf, 1, 1), auxfn(nf, 3, 1);
    const ConstArray4 c = Uc.const_array();
    const ConstArray4 ofo = Uf.const_array();
    const Box2D oldfb = Uf.box();
    Array4 a = Ufn.array();
    auto crsn = [](int x) { return x >= 0 ? x / 2 : -((-x + 1) / 2); };
    for (int j = nf.lo[1]; j <= nf.hi[1]; ++j)
      for (int i = nf.lo[0]; i <= nf.hi[0]; ++i)
        a(i, j) = oldfb.contains(i, j) ? ofo(i, j) : c(crsn(i), crsn(j));
    Uf = std::move(Ufn);
    auxf = std::move(auxfn);
    CI0 = nCI0;
    CI1 = nCI1;
    CJ0 = nCJ0;
    CJ1 = nCJ1;
  };

  compute_aux();
  const double M0 = mass();
  double dt = 0.4 * dxc / vmax();
  const int snap = std::max(1, nsteps / 30);
  std::printf("diocotron AMR (regrid dynamique) nc=%d dt=%.2e\n", nc, dt);

  int frame = 0;
  for (int s = 0; s <= nsteps; ++s) {
    if (s % snap == 0) {
      dump(frame++);
      std::printf("  s=%4d  fine=[%d..%d]x[%d..%d]  drift=%.2e\n", s, CI0, CI1,
                  CJ0, CJ1, std::fabs(mass() - M0));
    }
    if (s == nsteps) break;
    average_down_fab(Uf, Uc, CI0, CI1, CJ0, CJ1);  // sync grossier
    if (s > 0 && s % 20 == 0) regrid();            // remaillage dynamique
    compute_aux();
    amr_step_2level(model, Uc, dom, dxc, dyc, Uf, CI0, CI1, CJ0, CJ1, auxc,
                    auxf, dt);
    if (s % 20 == 0) dt = 0.4 * dxc / vmax();
  }
  std::printf("ecrit %s + %d instantanes ; drift final=%.2e\n", out.c_str(),
              frame, std::fabs(mass() - M0));
  return 0;
}
