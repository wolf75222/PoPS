// ADC-324 regression : le patch fin SEED du chemin AMR compile (build_amr_compiled, partage par
// add_compiled_model ET le bloc natif add_block mono-bloc) n'est alloue QUE quand le raffinement est
// reellement configure (set_refinement avec un seuil fini). Sans set_refinement, refine_threshold
// reste au sentinel 1e30 "pas de raffinement" : la hierarchie est alors MONO-NIVEAU (n_patches()==0),
// comme le chemin amr-schur, donc le transport grossier se distribue proprement sous MPI. Avant ce
// correctif un patch fin central (une SEULE boite non decoupee sur la dmap grossiere -> rang 0)
// persistait (n_patches()==1) meme sans raffinement : a np=4 le rang 0 portait ses boites grossieres
// PLUS tout le patch fin, et le flux aux vitesses exactes ne scalait pas (cf. ADC-324, mesure ROMEO).
//
// Quand le raffinement EST configure (set_refinement(seuil fini)), le seed est alloue et le regrid de
// build chope + distribue exactement comme avant : n_patches()>=1, chemin raffine INCHANGE (la parite
// bit-a-bit du chemin raffine est verrouillee par test_amr_compiled_model / test_amr_riemann_native).
#include <gtest/gtest.h>

#include <pops/physics/bricks/bricks.hpp>  // CompositeModel, GravityForce, GravityCoupling
#include <pops/physics/fluids/euler.hpp>   // Euler (= CompressibleFlux)
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/config/model_spec.hpp>

#include <array>
#include <cmath>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

static std::vector<double> bubble(int n) {  // bulle de densite lisse (pic 1.5 > 1.2), periodique
  std::vector<double> rho(static_cast<std::size_t>(n) * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const double x = (i + 0.5) / n - 0.5, y = (j + 0.5) / n - 0.5;
      rho[static_cast<std::size_t>(j) * n + i] = 1.0 + 0.5 * std::exp(-(x * x + y * y) / 0.02);
    }
  return rho;
}

using Model = CompositeModel<Euler, GravityForce, GravityCoupling>;

namespace {

using PatchRectangle = std::array<double, 4>;

bool same_patch_boxes(const std::vector<PatchBox>& lhs, const std::vector<PatchBox>& rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (std::size_t k = 0; k < lhs.size(); ++k) {
    const PatchBox& a = lhs[k];
    const PatchBox& b = rhs[k];
    if (a.level != b.level || a.ilo != b.ilo || a.jlo != b.jlo || a.ihi != b.ihi ||
        a.jhi != b.jhi)
      return false;
  }
  return true;
}

// Native mirror of the public Python patch_rectangles conversion. Keeping the conversion in the test
// makes the C++ mono-block hook responsible only for its typed, index-space PatchBox contract while
// still proving that every returned box maps one-to-one to a valid physical rectangle.
std::vector<PatchRectangle> physical_rectangles(const std::vector<PatchBox>& boxes, int n,
                                                double length) {
  std::vector<PatchRectangle> rectangles;
  rectangles.reserve(boxes.size());
  for (const PatchBox& box : boxes) {
    // ldexp avoids an undefined integer shift if a broken provider ever returns a nonsensical level;
    // the test below then reports that contract violation explicitly.
    const double dx = length / std::ldexp(static_cast<double>(n), box.level);
    rectangles.push_back(PatchRectangle{box.ilo * dx, box.jlo * dx,
                                        (box.ihi - box.ilo + 1) * dx,
                                        (box.jhi - box.jlo + 1) * dx});
  }
  return rectangles;
}

}  // namespace

TEST(test_amr_seed_no_refine, Runs) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  const int n = 64;
  const std::vector<double> rho = bubble(n);

  AmrSystemConfig cfg;
  cfg.n = n;
  cfg.L = 1.0;
  cfg.periodic = true;

  // (A) SANS set_refinement (refine_threshold == 1e30) sur la config amr_scale (regrid_every=0) :
  //     hierarchie MONO-NIVEAU, aucun patch fin seed -> n_patches() == 0.
  {
    AmrSystemConfig c = cfg;
    c.regrid_every = 0;
    AmrSystem A(c);
    add_compiled_model(A, "gas", Model{Euler{1.4}, GravityForce{}, GravityCoupling{-1.0, 1.0, 1.0}},
                       "minmod", "rusanov", "conservative", "explicit", /*gamma=*/1.4);
    A.set_poisson("charge_density", "geometric_mg");
    A.set_density("gas", rho);
    EXPECT_EQ(A.n_patches(), 0) << "no set_refinement -> n_patches()==0 (compile, mono-niveau)";
    // le mono-niveau reste un solveur valide : il avance et conserve la masse (FV periodique).
    const double m0 = A.mass();
    for (int s = 0; s < 8; ++s)
      A.step(1e-3);
    const std::vector<double> d = A.density();
    double nrm = 0;
    for (double v : d)
      nrm = std::fmax(nrm, std::fabs(v));
    EXPECT_TRUE(!d.empty() && nrm > 1e-6) << "no set_refinement : densite non triviale apres pas";
    EXPECT_TRUE(std::fabs(A.mass() - m0) < 1e-9 * (std::fabs(m0) + 1.0))
        << "no set_refinement : masse conservee (mono-niveau)";
    EXPECT_EQ(A.n_patches(), 0) << "no set_refinement : reste mono-niveau apres pas";
  }

  // (B) AVEC set_refinement(1.2) (seuil fini, bulle a 1.5 > 1.2) : le seed est alloue et le regrid de
  //     build chope -> n_patches() >= 1 ; le raffinement reste actif au fil des pas (regrid_every>0).
  {
    AmrSystemConfig c = cfg;
    c.regrid_every = 4;
    AmrSystem B(c);
    add_compiled_model(B, "gas", Model{Euler{1.4}, GravityForce{}, GravityCoupling{-1.0, 1.0, 1.0}},
                       "minmod", "rusanov", "conservative", "explicit", /*gamma=*/1.4);
    B.set_poisson("charge_density", "geometric_mg");
    B.set_refinement(1.2);
    B.set_density("gas", rho);
    EXPECT_GE(B.n_patches(), 1) << "set_refinement(1.2) -> n_patches()>=1 (seed alloue + regrid)";
    for (int s = 0; s < 8; ++s)
      B.step(1e-3);
    EXPECT_GE(B.n_patches(), 1) << "set_refinement(1.2) : raffinement actif apres pas";
  }

  // (C) chemin NATIF (add_block via ModelSpec) : il PARTAGE build_amr_compiled, donc la meme garde
  //     s'applique -> sans set_refinement, mono-niveau (n_patches()==0).
  {
    AmrSystemConfig c = cfg;
    c.regrid_every = 0;
    AmrSystem C(c);
    ModelSpec spec;
    spec.transport = "compressible";
    spec.source = "gravity";
    spec.elliptic = "gravity";
    spec.gamma = 1.4;
    spec.sign = -1.0;
    spec.four_pi_G = 1.0;
    spec.rho0 = 1.0;
    C.add_block("gas", spec, "minmod", "rusanov", "conservative", "explicit", 1);
    C.set_poisson("charge_density", "geometric_mg");
    C.set_density("gas", rho);
    EXPECT_EQ(C.n_patches(), 0) << "no set_refinement -> n_patches()==0 (natif, builder partage)";
  }

  // (D) GEOMETRIE MONO-BLOC NATIVE : add_block materialise AmrCouplerMP (et non AmrRuntime). Le hook
  //     patch_boxes() doit exposer exactement les boites fines du coupleur, dans l'ordre utilise par
  //     la conversion publique patch_rectangles(). Les lectures repetees sont idempotentes et ne
  //     modifient ni l'etat conservatif ni l'horloge.
  {
    AmrSystemConfig c = cfg;
    c.regrid_every = 4;
    AmrSystem D(c);
    ModelSpec spec;
    spec.transport = "compressible";
    spec.source = "gravity";
    spec.elliptic = "gravity";
    spec.gamma = 1.4;
    spec.sign = -1.0;
    spec.four_pi_G = 1.0;
    spec.rho0 = 1.0;
    D.add_block("gas", spec, "minmod", "rusanov", "conservative", "explicit", 1);
    D.set_poisson("charge_density", "geometric_mg");
    D.set_refinement(1.2);
    D.set_density("gas", rho);

    const std::vector<PatchBox> boxes_first = D.patch_boxes();  // lazy-builds the native coupler
    ASSERT_FALSE(boxes_first.empty()) << "native_monoblock_refinement_has_fine_geometry";
    EXPECT_EQ(boxes_first.size(), static_cast<std::size_t>(D.n_patches()))
        << "patch_boxes_parallel_to_native_patch_count";
    EXPECT_EQ(D.n_blocks(), 1) << "geometry_route_is_native_monoblock";

    const std::vector<PatchRectangle> rectangles_first = physical_rectangles(boxes_first, n, c.L);
    ASSERT_EQ(rectangles_first.size(), boxes_first.size())
        << "one_physical_rectangle_per_native_patch_box";
    for (std::size_t k = 0; k < boxes_first.size(); ++k) {
      const PatchBox& box = boxes_first[k];
      ASSERT_GE(box.level, 1);
      ASSERT_LT(box.level, 30);  // keeps the ratio-2 index-space shift defined below
      const int level_cells = n << box.level;
      EXPECT_LE(0, box.ilo);
      EXPECT_LE(box.ilo, box.ihi);
      EXPECT_LT(box.ihi, level_cells);
      EXPECT_LE(0, box.jlo);
      EXPECT_LE(box.jlo, box.jhi);
      EXPECT_LT(box.jhi, level_cells);

      const double dx = c.L / static_cast<double>(level_cells);
      const PatchRectangle& rectangle = rectangles_first[k];
      EXPECT_DOUBLE_EQ(rectangle[0], box.ilo * dx);
      EXPECT_DOUBLE_EQ(rectangle[1], box.jlo * dx);
      EXPECT_DOUBLE_EQ(rectangle[2], (box.ihi - box.ilo + 1) * dx);
      EXPECT_DOUBLE_EQ(rectangle[3], (box.jhi - box.jlo + 1) * dx);
      EXPECT_GE(rectangle[0], 0.0);
      EXPECT_GE(rectangle[1], 0.0);
      EXPECT_GT(rectangle[2], 0.0);
      EXPECT_GT(rectangle[3], 0.0);
      EXPECT_LE(rectangle[0] + rectangle[2], c.L);
      EXPECT_LE(rectangle[1] + rectangle[3], c.L);
    }

    const std::vector<double> state_before = D.level_state(0);
    const std::vector<double> density_before = D.density("gas");
    const double mass_before = D.mass("gas");
    const double time_before = D.time();
    const int macro_step_before = D.macro_step();
    const int levels_before = D.n_levels();

    const std::vector<PatchBox> boxes_second = D.patch_boxes();
    const std::vector<PatchRectangle> rectangles_second =
        physical_rectangles(boxes_second, n, c.L);
    EXPECT_TRUE(same_patch_boxes(boxes_first, boxes_second))
        << "native_patch_geometry_read_is_idempotent";
    EXPECT_EQ(rectangles_second, rectangles_first)
        << "physical_rectangle_conversion_is_parallel_and_idempotent";
    EXPECT_EQ(D.level_state(0), state_before) << "patch_geometry_read_does_not_mutate_state";
    EXPECT_EQ(D.density("gas"), density_before) << "patch_geometry_read_does_not_mutate_density";
    EXPECT_DOUBLE_EQ(D.mass("gas"), mass_before) << "patch_geometry_read_does_not_mutate_mass";
    EXPECT_DOUBLE_EQ(D.time(), time_before) << "patch_geometry_read_does_not_advance_time";
    EXPECT_EQ(D.macro_step(), macro_step_before) << "patch_geometry_read_does_not_advance_clock";
    EXPECT_EQ(D.n_levels(), levels_before) << "patch_geometry_read_does_not_mutate_hierarchy";
  }
}
