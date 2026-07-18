// ADC-299 + ADC-290 : la CONFIGURATION et le MODELE sont valides EN AMONT, sans defaut silencieux.
//
// ADC-299 (validation de config avant construction interne) : une SystemConfig / AmrSystemConfig
//   invalide (n <= 0, L <= 0, regrid_every < 0, coarse_max_grid < 0) est REJETEE avant que l'Impl
//   n'alloue quoi que ce soit. Pour System c'est crucial : Impl(c) derive la geometrie, le BoxArray,
//   le DistributionMapping et alloue l'aux MultiFab -- tous dimensionnes par c.n -- AVANT l'ancien
//   check_geometry. Un n=0 / L=0 ne plantait pas : il construisait une grille degeneree silencieuse
//   (boite vide, dx = L/0 = +inf ou dx negatif) qui ne se manifestait que loin en aval. On asserte le
//   refus immediat (pas un etat degenere) ET qu'une config valide construit toujours.
//
// ADC-290 (modele explicite, pas de retombee physique silencieuse) : un ModelSpec dont transport ou
//   elliptic n'est pas pose ECHOUE clairement -- AUCUN fallback vers compressible / charge. On verifie
//   le contrat directement (detail::validate_model_spec), via la surface utilisateur
//   System::add_block / AmrSystem::add_block (le message clair precede le routage par chaine), et que
//   le message NOMME le champ manquant (lisibilite). Un modele COMPLET reste accepte.
//
// Tests de CONTRAT (aucun calcul) : throws cibles + cas valides qui construisent. Compile
// python/system.cpp et python/amr_system.cpp (objets runtime splices, cf. tests/CMakeLists.txt).

#include <gtest/gtest.h>

#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/factory/model_factory.hpp>  // detail::validate_model_spec (contrat de completude)
#include <pops/runtime/config/model_spec.hpp>
#include <pops/runtime/system.hpp>

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace {

template <class T>
concept HasPublicFreezeRestore = requires(T& value) { value.pops_freeze_restore(false); };

static_assert(!HasPublicFreezeRestore<ModelSpec>,
              "ModelSpec.freeze() must be irreversible through its public C++ API");

// true si @p f leve un std::runtime_error DONT le message contient @p frag : on ne se contente pas du
// refus, on verifie que c'est le BON refus (le champ manquant nomme), donc un message lisible.
template <class F>
bool raises_with(F&& f, const std::string& frag) {
  try {
    f();
  } catch (const std::runtime_error& e) {
    return std::string(e.what()).find(frag) != std::string::npos;
  } catch (...) {
    return false;
  }
  return false;
}

// true si @p f leve un std::runtime_error (le refus attendu, sans egard au message).
template <class F>
bool raises(F&& f) {
  return raises_with(std::forward<F>(f), "");
}

// Modele natif COMPLET (diocotron scalaire) : transport + source + elliptic poses explicitement.
ModelSpec exb_charge() {
  ModelSpec s;
  s.transport = "exb";
  s.source = "none";
  s.elliptic = "charge";
  return s;
}

#if defined(POPS_HAS_KOKKOS)
// Every TEST in this binary constructs System / AmrSystem instances (Kokkos-dependent), so Kokkos
// is initialized once for the whole process via a GoogleTest global environment (ScopeGuard itself
// aborts if constructed while already initialized, so it cannot live inside each TEST).
class KokkosEnvironment : public ::testing::Environment {
 public:
  void SetUp() override { guard_.emplace(); }
  void TearDown() override { guard_.reset(); }

 private:
  std::optional<Kokkos::ScopeGuard> guard_;
};

::testing::Environment* const kKokkosEnv =
    ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
#endif

}  // namespace

// ================================================================================================
// ADC-299 : SystemConfig invalide rejetee AVANT la construction de Impl (allocation geom/ba/dm/aux).
// ================================================================================================
TEST(ConfigModelValidation, SystemConfigInvalidRejectedBeforeImpl) {
  EXPECT_TRUE(raises_with([&] { System s(SystemConfig{0, 1.0, false}); }, "n >= 1"))
      << "System(n=0) rejete avant Impl (n >= 1)";
  EXPECT_TRUE(raises_with([&] { System s(SystemConfig{-4, 1.0, false}); }, "n >= 1"))
      << "System(n<0) rejete";
  EXPECT_TRUE(raises_with([&] { System s(SystemConfig{16, 0.0, false}); }, "L > 0"))
      << "System(L=0) rejete (L > 0)";
  EXPECT_TRUE(raises_with([&] { System s(SystemConfig{16, -1.0, false}); }, "L > 0"))
      << "System(L<0) rejete";
  // Une config valide CONSTRUIT toujours (le garde-fou ne sur-rejette pas).
  {
    bool ok = false;
    try {
      System s(SystemConfig{16, 1.0, false});
      ok = (s.nx() == 16);
    } catch (...) {
      ok = false;
    }
    EXPECT_TRUE(ok) << "System config valide construit (nx == 16)";
  }
}

TEST(ConfigModelValidation, SystemRejectsAmbiguousDiffusionCoefficientShapes) {
  const std::vector<double> scalar(8 * 8, 2.0);
  const std::vector<double> diagonal_x(8 * 8, 3.0);
  const std::vector<double> diagonal_y(8 * 8, 4.0);

  System scalar_first(SystemConfig{8, 1.0, false});
  scalar_first.set_epsilon_field(scalar);
  EXPECT_TRUE(raises_with(
      [&] { scalar_first.set_epsilon_anisotropic_field(diagonal_x, diagonal_y); },
      "cannot be combined"));

  System diagonal_first(SystemConfig{8, 1.0, false});
  diagonal_first.set_epsilon_anisotropic_field(diagonal_x, diagonal_y);
  EXPECT_TRUE(raises_with([&] { diagonal_first.set_epsilon_field(scalar); },
                          "cannot be combined"));

  // Reconfiguration within the same exact shape is intentional and remains available while the
  // System is assembling; only a physically ambiguous shape change is rejected.
  EXPECT_NO_THROW(scalar_first.set_epsilon_field(std::vector<double>(8 * 8, 5.0)));
  EXPECT_NO_THROW(diagonal_first.set_epsilon_anisotropic_field(
      std::vector<double>(8 * 8, 6.0), std::vector<double>(8 * 8, 7.0)));
}

// ================================================================================================
// ADC-299 : AmrSystemConfig invalide rejetee AVANT Impl (parite avec System).
// ================================================================================================
TEST(ConfigModelValidation, AmrSystemConfigInvalidRejectedBeforeImpl) {
  EXPECT_TRUE(raises_with([&] { AmrSystem a(AmrSystemConfig{0}); }, "n >= 1"))
      << "AmrSystem(n=0) rejete (n >= 1)";
  {
    AmrSystemConfig c;
    c.n = 32;
    c.L = 0.0;
    EXPECT_TRUE(raises_with([&] { AmrSystem a(c); }, "L > 0")) << "AmrSystem(L=0) rejete (L > 0)";
  }
  {
    AmrSystemConfig c;
    c.n = 32;
    c.regrid_every = -1;
    EXPECT_TRUE(raises_with([&] { AmrSystem a(c); }, "regrid_every"))
        << "AmrSystem(regrid_every<0) rejete";
  }
  {
    AmrSystemConfig c;
    c.n = 32;
    c.coarse_max_grid = -1;
    EXPECT_TRUE(raises_with([&] { AmrSystem a(c); }, "coarse_max_grid"))
        << "AmrSystem(coarse_max_grid<0) rejete";
  }
  {
    bool ok = false;
    try {
      AmrSystem a(AmrSystemConfig{32});
      ok = (a.nx() == 32);
    } catch (...) {
      ok = false;
    }
    EXPECT_TRUE(ok) << "AmrSystem config valide construit (nx == 32)";
  }
}

// ================================================================================================
// ADC-290 (a) : le contrat direct validate_model_spec nomme le champ manquant.
// ================================================================================================
TEST(ConfigModelValidation, ValidateModelSpecNamesMissingField) {
  EXPECT_TRUE(raises_with([&] { detail::validate_model_spec(ModelSpec{}); }, "transport"))
      << "validate_model_spec : transport non pose rejete, message nomme 'transport'";
  {
    ModelSpec only_tr;
    only_tr.transport = "exb";  // elliptic encore vide
    EXPECT_TRUE(raises_with([&] { detail::validate_model_spec(only_tr); }, "elliptic"))
        << "validate_model_spec : elliptic non pose rejete, message nomme 'elliptic'";
  }
  {
    ModelSpec no_src;
    no_src.transport = "exb";
    no_src.elliptic = "charge";
    no_src.source = "";  // source explicitement videe
    EXPECT_TRUE(raises_with([&] { detail::validate_model_spec(no_src); }, "source"))
        << "validate_model_spec : source vide rejetee, message nomme 'source'";
  }
  // Un modele COMPLET passe le contrat (le garde-fou ne sur-rejette pas).
  EXPECT_TRUE(!raises([&] { detail::validate_model_spec(exb_charge()); }))
      << "validate_model_spec : modele complet (exb/none/charge) accepte";
}

TEST(ConfigModelValidation, ModelSpecFreezeIsIrreversibleForNativeCallers) {
  ModelSpec spec;
  spec.transport = "exb";
  spec.gamma = 1.25;
  EXPECT_FALSE(spec.frozen());
  spec.freeze();
  EXPECT_TRUE(spec.frozen());
  EXPECT_THROW(spec.require_mutable("transport"), std::runtime_error);
  EXPECT_THROW(spec.transport = "compressible", std::runtime_error);
  EXPECT_THROW(spec.gamma = 1.4, std::runtime_error);

  ModelSpec frozen_copy = spec;
  EXPECT_TRUE(frozen_copy.frozen());
  EXPECT_THROW(frozen_copy.source = "gravity", std::runtime_error);

  ModelSpec mutable_target;
  mutable_target = spec;
  EXPECT_TRUE(mutable_target.frozen());
  EXPECT_THROW(mutable_target.q = 2.0, std::runtime_error);

  ModelSpec move_target;
  move_target.transport = std::move(spec.transport);
  EXPECT_EQ(spec.transport.get(), "exb") << "a frozen source proxy is never drained by move";
  EXPECT_TRUE(spec.frozen());
  EXPECT_THROW(spec.require_mutable("transport"), std::runtime_error);
  EXPECT_THROW(spec.transport = "compressible", std::runtime_error);
  EXPECT_THROW(spec.gamma = 1.4, std::runtime_error);
}

TEST(ConfigModelValidation, ModelSpecCopyAndMoveRebindEveryProxyToItsDestination) {
  ModelSpec copy_source;
  copy_source.transport = "exb";
  copy_source.gamma = 1.1;
  ModelSpec copy_constructed(copy_source);
  copy_constructed.freeze();
  EXPECT_NO_THROW(copy_source.transport = "compressible");
  EXPECT_NO_THROW(copy_source.gamma = 1.2);
  EXPECT_THROW(copy_constructed.transport = "isothermal", std::runtime_error);
  EXPECT_THROW(copy_constructed.gamma = 1.3, std::runtime_error);

  ModelSpec move_source;
  move_source.transport = "exb";
  move_source.gamma = 1.1;
  ModelSpec move_constructed(std::move(move_source));
  move_constructed.freeze();
  EXPECT_NO_THROW(move_source.transport = "compressible");
  EXPECT_NO_THROW(move_source.gamma = 1.2);
  EXPECT_THROW(move_constructed.transport = "isothermal", std::runtime_error);
  EXPECT_THROW(move_constructed.gamma = 1.3, std::runtime_error);

  ModelSpec copy_assignment_source;
  copy_assignment_source.transport = "exb";
  copy_assignment_source.gamma = 1.1;
  ModelSpec copy_assigned;
  copy_assigned = copy_assignment_source;
  copy_assigned.freeze();
  EXPECT_NO_THROW(copy_assignment_source.transport = "compressible");
  EXPECT_NO_THROW(copy_assignment_source.gamma = 1.2);
  EXPECT_THROW(copy_assigned.transport = "isothermal", std::runtime_error);
  EXPECT_THROW(copy_assigned.gamma = 1.3, std::runtime_error);

  ModelSpec move_assignment_source;
  move_assignment_source.transport = "exb";
  move_assignment_source.gamma = 1.1;
  ModelSpec move_assigned;
  move_assigned = std::move(move_assignment_source);
  move_assigned.freeze();
  EXPECT_NO_THROW(move_assignment_source.transport = "compressible");
  EXPECT_NO_THROW(move_assignment_source.gamma = 1.2);
  EXPECT_THROW(move_assigned.transport = "isothermal", std::runtime_error);
  EXPECT_THROW(move_assigned.gamma = 1.3, std::runtime_error);
}

// ================================================================================================
// ADC-290 (b)/(c) : la surface utilisateur (System / AmrSystem add_block) applique le meme contrat.
// ================================================================================================
TEST(ConfigModelValidation, AddBlockAppliesContractBeforeStringRouting) {
  // Surface utilisateur : le contrat s'applique a l'entree de System::add_block, AVANT le routage
  // par chaine sur model.transport (qui dirait sinon "unknown transport ''"). Le defaut-construit
  // ne devient JAMAIS un Euler silencieux.
  EXPECT_TRUE(raises_with(
      [&] {
        System s(SystemConfig{16, 1.0, false});
        s.add_block("m", ModelSpec{});
      },
      "transport"))
      << "System::add_block(ModelSpec incomplet) rejete -- pas de transport 'compressible' "
         "silencieux";
  // Un modele complet s'installe (chemin natif ExB scalaire complet, sans lever).
  EXPECT_TRUE(!raises([&] {
    System s(SystemConfig{16, 1.0, false});
    s.add_block("ne", exb_charge());
  })) << "System::add_block(modele complet) accepte";

  // Meme contrat a l'entree de AmrSystem::add_block (parite). add_block est paresseux : le refus
  // tombe au contrat, sans declencher le build de la hierarchie.
  EXPECT_TRUE(raises_with(
      [&] {
        AmrSystem a(AmrSystemConfig{16});
        a.add_block("m", ModelSpec{});
      },
      "transport"))
      << "AmrSystem::add_block(ModelSpec incomplet) rejete -- pas de fallback silencieux";
}

// ================================================================================================
// ADC-331 : completude du routage. Chaque tag builtin de la registry (model_registry.hpp) DOIT etre
// route par le dispatch -- une ligne de table sans branche if = une derive registry/dispatch
// (validate_transport / validate_elliptic acceptent le tag, mais la chaine if/else tombe sur le
// garde "valid in registry but not routed"). Visiteur no-op : on verifie seulement que le dispatch
// ATTEINT une branche, sans construire le CompositeModel complet. Pendant runtime du static_assert
// de non-derive n_vars (model_factory.hpp) : ici on verrouille l'EXHAUSTIVITE du routage.
// ================================================================================================
TEST(ConfigModelValidation, EveryBuiltinRegistryTagIsRouted) {
  bool all_tr = true;
  for (const TransportTag& t : kTransports) {
    ModelSpec s;
    s.transport = t.name;
    bool routed = false;
    try {
      detail::dispatch_transport(s, [&](auto) { routed = true; });
    } catch (...) {
      routed = false;
    }
    all_tr = all_tr && routed;
  }
  EXPECT_TRUE(all_tr)
      << "ADC-331 : tout transport builtin de la registry est route (pas de derive)";

  bool all_el = true;
  for (const EllipticTag& t : kElliptics) {
    ModelSpec s;
    s.elliptic = t.name;
    bool routed = false;
    try {
      detail::dispatch_elliptic(s, [&](auto) { routed = true; });
    } catch (...) {
      routed = false;
    }
    all_el = all_el && routed;
  }
  EXPECT_TRUE(all_el) << "ADC-331 : tout elliptic builtin de la registry est route (pas de derive)";

  // dispatch_source est templatise sur NV et garde les forces fluides derriere `if constexpr
  // (NV >= 3)` ; a NV=4 (Euler) les sept orthographes builtin routent. Une ligne kSources ajoutee
  // sans branche dans dispatch_source ferait echouer ce passage (meme garde de derive que pour
  // transport / elliptic, etendue a l'axe source que le if/else NV-dependant rend moins evident).
  bool all_src = true;
  for (const SourceTag& t : kSources) {
    ModelSpec s;
    s.source = t.name;
    bool routed = false;
    try {
      detail::dispatch_source<4>(s, [&](auto) { routed = true; });
    } catch (...) {
      routed = false;
    }
    all_src = all_src && routed;
  }
  EXPECT_TRUE(all_src)
      << "ADC-331 : tout source builtin de la registry route a NV=4 (pas de derive)";
}
