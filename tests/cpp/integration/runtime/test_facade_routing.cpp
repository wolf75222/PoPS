// Exact-rank analytic embedded-boundary routing (staircase + cut-cell) in System::step.
//
// On valide (vraies assertions, pas de no-op) :
//   (a) NO-MASK : an analytic LevelSet with mode='none' is BYTE-IDENTIQUE to no EB.
//   (b) ROUTING-LIVE (staircase) : mode='staircase' produces a different state on the same init,
//       and mass on the active region is
//       conservee a la machine (aucun flux ne franchit la frontiere du masque) -> propre au schema masque.
//   (c) CUTCELL : mode='cutcell' runs with finite state, differs from Cartesian; an enclosing
//       LevelSet is BIT-IDENTIQUE to Cartesian.
//
// Model: exact-ranked scalar advection. Constant divergence-free velocity isolates EB routing and
// proves conservation without a two-dimensional geometric authority.

#include <gtest/gtest.h>

#include <pops/mesh/geometry/geometry.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/program/program_context.hpp>
#include <pops/runtime/system.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace pops {

template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_exact_system_block(
    CompiledSystemBlockPreparation<Dim, Model> request) {
  return prepare_generated_system_block(std::move(request));
}

}  // namespace pops

namespace {

constexpr int kTestDimension = kNativeDimension;
using NativeSystem = System<kTestDimension>;
using NativeSystemConfig = SystemConfig<kTestDimension>;
using NativeField = MultiFab<kTestDimension>;
using NativeGeometry = Geometry<kTestDimension>;
using NativeIndex = Index<kTestDimension>;
using NativeBox = Box<kTestDimension>;
using NativeExtent = Extent<kTestDimension>;
constexpr int kCompressibleComponents = kTestDimension + 2;
using ScalarModel = nd::ScalarAdvection<kTestDimension>;
using CompressibleModel = nd::IdealGasEuler<kTestDimension>;

void install_execution_lane(NativeSystem& system, std::string identity) {
  system.install_prepared_boundary_execution_lane(
      std::make_shared<ExecutionLane>(ExecutionLane::world(std::move(identity))));
}

NativeSystemConfig native_config(int cells, Real length, bool periodic) {
  NativeSystemConfig config;
  NativeIndex lower{};
  NativeIndex upper{};
  for (int axis = 0; axis < kTestDimension; ++axis) {
    config.shape[axis] = cells;
    config.lower[axis] = Real(0);
    config.upper[axis] = length;
    config.periodicity[static_cast<std::size_t>(axis)] = periodic;
    lower[axis] = 0;
    upper[axis] = cells - 1;
  }
  config.boxes = {NativeBox{lower, upper}};
  return config;
}

NativeGeometry native_geometry(const NativeSystemConfig& config) {
  return NativeGeometry::from_bounds(NativeBox::from_extents(config.shape), config.lower,
                                     config.upper);
}

std::size_t cell_count(const NativeExtent& extent) {
  std::size_t count = 1;
  for (int axis = 0; axis < kTestDimension; ++axis)
    count *= static_cast<std::size_t>(extent[axis]);
  return count;
}

NativeIndex index_from_linear(std::size_t linear, const NativeBox& box) {
  NativeIndex index{};
  for (int axis = 0; axis < kTestDimension; ++axis) {
    const std::size_t axis_extent = static_cast<std::size_t>(box.length(axis));
    index[axis] = box.lo[axis] + static_cast<int>(linear % axis_extent);
    linear /= axis_extent;
  }
  return index;
}

void install_forward_euler_program(NativeSystem& system) {
  std::vector<int> block_map(static_cast<std::size_t>(system.n_blocks()));
  std::iota(block_map.begin(), block_map.end(), 0);
  system.set_program_block_map(block_map);

  auto context = runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("test.clock.macro");
  context->install([context](double dt) {
    context->begin_step(dt);
    context->set_stage_time(0, 1);
    (void)consume_solve_outcome(context->solve_fields());

    std::vector<NativeField*> states;
    std::vector<NativeField*> next_states;
    states.reserve(static_cast<std::size_t>(context->n_blocks()));
    next_states.reserve(static_cast<std::size_t>(context->n_blocks()));
    for (int block = 0; block < context->n_blocks(); ++block) {
      NativeField& state = context->state(block);
      NativeField& residual = context->rhs_scratch(1000 + block, 0, state);
      NativeField& next = context->scratch_state(2000 + block, 0, state);
      context->rhs_into(block, state, residual, 3000 + block);
      context->lincomb(next, Real(1), state, Real(dt), residual);
      states.push_back(&state);
      next_states.push_back(&next);
    }
    for (std::size_t block = 0; block < states.size(); ++block)
      context->lincomb(*states[block], Real(0), *states[block], Real(1), *next_states[block]);
  });
  system.set_program_block_map(block_map);
}

#if defined(POPS_HAS_KOKKOS)
Kokkos::ScopeGuard& kokkos_scope() {
  static Kokkos::ScopeGuard guard;
  return guard;
}
#endif

// Densite initiale : coquille lisse, perturbee suivant toutes les coordonnees exactes pour casser
// les symetries. La linearisation garde l'axe 0 contigu, comme MultiFab<Dim>.
std::vector<double> ring_density(const NativeSystemConfig& config) {
  const NativeBox domain = NativeBox::from_extents(config.shape);
  const NativeGeometry geometry = native_geometry(config);
  std::vector<double> rho(static_cast<std::size_t>(domain.numPts()), 1e-3);
  constexpr double two_pi = 2.0 * 3.14159265358979323846;
  for (std::size_t linear = 0; linear < rho.size(); ++linear) {
    const auto position = geometry.cell_center(index_from_linear(linear, domain));
    double radius_squared = 0.0;
    double phase = 0.0;
    for (int axis = 0; axis < kTestDimension; ++axis) {
      const double length = config.upper[axis] - config.lower[axis];
      const double center = 0.5 * (config.lower[axis] + config.upper[axis]);
      const double offset = position[axis] - center;
      radius_squared += offset * offset;
      phase += static_cast<double>(axis + 1) * two_pi * offset / length;
    }
    const double length = config.upper[0] - config.lower[0];
    const double radius = std::sqrt(radius_squared);
    const double r0 = 0.18 * length;
    const double width = 0.05 * length;
    const double gaussian = std::exp(-((radius - r0) * (radius - r0)) / (2.0 * width * width));
    rho[linear] = 1e-3 + gaussian * (1.0 + 0.3 * std::sin(phase));
  }
  return rho;
}

std::vector<double> periodic_seam_density(const NativeSystemConfig& config) {
  const NativeBox domain = NativeBox::from_extents(config.shape);
  const NativeGeometry geometry = native_geometry(config);
  std::vector<double> rho(static_cast<std::size_t>(domain.numPts()));
  const double two_pi = 2.0 * std::acos(-1.0);
  for (std::size_t linear = 0; linear < rho.size(); ++linear) {
    const auto position = geometry.cell_center(index_from_linear(linear, domain));
    double harmonic = std::cos(two_pi * position[0]);
    for (int axis = 1; axis < kTestDimension; ++axis)
      harmonic *= std::sin(two_pi * position[axis]);
    rho[linear] = 1.0 + 0.15 * harmonic;
  }
  return rho;
}

ScalarModel scalar_transport_model() {
  RealVector<kTestDimension> velocity{};
  for (int axis = 0; axis < kTestDimension; ++axis)
    velocity[axis] = Real(0.25) / Real(axis + 1);
  return ScalarModel::prepare(velocity);
}

void add_periodic_transport(NativeSystem& system) {
  system.install_block_state_route("n", "test:facade-routing/n/state");
  system.seal_auxiliary_providers();
  add_compiled_model(system, "n", scalar_transport_model(), "none", "rusanov", "conservative",
                     "explicit");
  system.set_poisson("composite", "cartesian_cg", "periodic");
}

void add_compressible(NativeSystem& system) {
  system.install_block_state_route("gas", "test:facade-routing/gas/state");
  system.seal_auxiliary_providers();
  add_compiled_model(system, "gas", CompressibleModel::prepare(Real(1.4)), "none", "rusanov",
                     "conservative", "explicit", 1.4);
}

// Construit un System d'advection scalaire exact-rank pret a stepper. Le domaine/mode est pose par
// l'appelant. First-order reconstruction is the native embedded-boundary provider supported here.
void build_transport(NativeSystem& s) {
  // First-order reconstruction is the native embedded-boundary provider supported by this facade.
  // Higher-order stencils require geometry-aware neighbor reconstruction and are rejected rather
  // than reading inactive cells. The same provider is used in every mode so this test isolates only
  // residual routing.
  s.install_block_state_route("n", "test:facade-routing/n/state");
  s.seal_auxiliary_providers();
  add_compiled_model(s, "n", scalar_transport_model(), "none", "rusanov", "conservative",
                     "explicit");
  // Le Program conserve son solve de champ exact-rank ; le modele scalaire fournit un RHS nul et
  // la vitesse transportee est entierement preparee dans la loi de conservation.
  s.set_poisson("composite", "cartesian_cg", "dirichlet");
  install_forward_euler_program(s);
}

// max|diff| composante a composante entre deux champs de meme taille.
double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k)
    d = std::fmax(d, std::fabs(a[k] - b[k]));
  return d;
}

bool all_finite(const std::vector<double>& a) {
  for (double v : a)
    if (!std::isfinite(v))
      return false;
  return true;
}

// Build phi = hypot(x0-c, ..., x{Dim-1}-c) - radius directly in the native postfix VM.  The
// expression consumes every coordinate of the compiled rank: no fixed-axis adapter owns the
// geometry used by these runtime assertions.
void install_centered_ball(NativeSystem& system, double radius, const std::string& mode,
                           double kappa_min = 0.0, double face_open_eps = 0.0,
                           double cut_theta_min = 0.0) {
  static constexpr const char* coordinates[] = {"x", "y", "z"};
  std::vector<std::string> opcodes;
  std::vector<double> literals;
  for (int axis = 0; axis < kTestDimension; ++axis) {
    opcodes.emplace_back(coordinates[axis]);
    literals.push_back(0.0);
    opcodes.emplace_back("constant");
    literals.push_back(0.5);
    opcodes.emplace_back("sub");
    literals.push_back(0.0);
    if (axis > 0) {
      opcodes.emplace_back("hypot");
      literals.push_back(0.0);
    }
  }
  opcodes.emplace_back("constant");
  literals.push_back(radius);
  opcodes.emplace_back("sub");
  literals.push_back(0.0);
  system.set_analytic_level_set(opcodes, literals, mode, kappa_min, face_open_eps, cut_theta_min);
}

}  // namespace

TEST(FacadeRouting, AnalyticEmbeddedBoundaryRoutingBehavesAcrossNoneStaircaseAndCutcell) {
#if defined(POPS_HAS_KOKKOS)
  (void)kokkos_scope();
#endif
  const int n = 48;
  const double L = 1.0;
  const double radius = 0.30 * L;  // smaller than the box: genuine inactive cells
  const double dt = 2e-4;          // pas court, advection sous-CFL
  const int n_steps = 12;
  const NativeSystemConfig config = native_config(n, L, true);
  const NativeGeometry geometry = native_geometry(config);
  const std::vector<double> rho0 = ring_density(config);

  // ----------------------------------------------------------------------
  // (a) NO-MASK: a prepared analytic geometry in mode='none' == no EB (byte for byte).
  // ----------------------------------------------------------------------
  std::vector<double>
      ref_state;  // etat de reference (chemin plein cartesien), reutilise par (b)/(c)/(d)
  {
    NativeSystem base(config);
    install_execution_lane(base, "pops.test.facade-routing.embedded.base");
    build_transport(base);
    base.set_density("n", rho0);
    for (int k = 0; k < n_steps; ++k)
      base.step(dt);
    ref_state = base.get_state("n");

    NativeSystem none(config);
    install_execution_lane(none, "pops.test.facade-routing.embedded.none");
    build_transport(none);
    none.set_density("n", rho0);
    install_centered_ball(none, radius, "none");
    for (int k = 0; k < n_steps; ++k)
      none.step(dt);
    const std::vector<double> none_state = none.get_state("n");

    const double d = max_abs_diff(ref_state, none_state);
    // Equality is byte-for-byte: mode none follows exactly the Cartesian residual path.
    EXPECT_TRUE(d == 0.0) << "(a) analytic mode='none' is BIT-IDENTIQUE to Cartesian: "
                             "max|diff| = "
                          << d << " (attendu 0)";
    EXPECT_TRUE(all_finite(ref_state) && ref_state.size() == cell_count(config.shape))
        << "(a) etat de reference fini et de taille shape.product (le pas plein a bien tourne)";
  }

  // ----------------------------------------------------------------------
  // (b) staircase: different from Cartesian + active mass conserved to roundoff.
  // ----------------------------------------------------------------------
  {
    NativeSystem sc(config);
    install_execution_lane(sc, "pops.test.facade-routing.embedded.staircase");
    build_transport(sc);
    sc.set_density("n", rho0);
    install_centered_ball(sc, radius, "staircase");

    // Masse initiale sur les cellules ACTIVES (masque 0/1 du System) AVANT les pas.
    const std::vector<double> mask = sc.embedded_boundary_mask();
    const std::vector<double> dens0 = sc.density("n");
    double cell_measure = 1.0;
    for (int axis = 0; axis < kTestDimension; ++axis)
      cell_measure *= geometry.spacing(axis);
    int n_active = 0, n_inactive = 0;
    double mass0 = 0.0;
    for (std::size_t k = 0; k < mask.size(); ++k) {
      if (mask[k] >= 0.5) {
        ++n_active;
        mass0 += dens0[k] * cell_measure;
      } else
        ++n_inactive;
    }
    ASSERT_TRUE(n_active > 0 && n_inactive > 0)
        << "(b) the analytic LevelSet partitions active and inactive cells";

    for (int k = 0; k < n_steps; ++k)
      sc.step(dt);
    const std::vector<double> sc_state = sc.get_state("n");

    // Masse active APRES les pas (meme masque : le disque est statique).
    const std::vector<double> dens1 = sc.density("n");
    double mass1 = 0.0;
    for (std::size_t k = 0; k < mask.size(); ++k)
      if (mask[k] >= 0.5)
        mass1 += dens1[k] * cell_measure;

    const double d_vs_square = max_abs_diff(ref_state, sc_state);
    const double rel_drift = std::fabs(mass1 - mass0) / std::fabs(mass0);

    // The masked operator closes faces at the implicit boundary, so it must differ from Cartesian.
    EXPECT_TRUE(d_vs_square > 1e-10)
        << "(b) staircase differs from Cartesian: max|diff| = " << d_vs_square << " (attendu > 0)";
    EXPECT_TRUE(all_finite(sc_state)) << "(b) staircase state is finite";
    // La masse sur les cellules actives est conservee a la machine (flux normal nul aux faces
    // active/inactive). Borne juste au-dessus du bruit flottant des sommes telescopiques de flux.
    EXPECT_TRUE(rel_drift < 1e-12)
        << "(b) masse sur les cellules actives conservee a la machine (schema masque conservatif) "
           ": drift = "
        << rel_drift;
  }

  // ----------------------------------------------------------------------
  // (c) cut-cell: finite and different; enclosing LevelSet == Cartesian (bit for bit).
  // ----------------------------------------------------------------------
  {
    // (c1) cutting geometry: finite state + differs from Cartesian.
    NativeSystem cc(config);
    install_execution_lane(cc, "pops.test.facade-routing.embedded.cutcell");
    build_transport(cc);
    cc.set_density("n", rho0);
    install_centered_ball(cc, radius, "cutcell");
    for (int k = 0; k < n_steps; ++k)
      cc.step(dt);
    const std::vector<double> cc_state = cc.get_state("n");
    const double d_vs_square = max_abs_diff(ref_state, cc_state);
    EXPECT_TRUE(all_finite(cc_state)) << "(c1) cut-cell state is finite";
    EXPECT_TRUE(d_vs_square > 1e-10)
        << "(c1) cut-cell differs from Cartesian: max|diff| = " << d_vs_square << " (attendu > 0)";

    // (c2) enclosing level set: every cell active and no cut face, hence EB == Cartesian.
    const double enclosing_radius = 10.0 * L;
    NativeSystem sq(config);  // reference 1 pas plein
    install_execution_lane(sq, "pops.test.facade-routing.embedded.square");
    build_transport(sq);
    sq.set_density("n", rho0);
    sq.step(dt);
    const std::vector<double> sq1 = sq.get_state("n");

    NativeSystem eb(config);
    install_execution_lane(eb, "pops.test.facade-routing.embedded.enclosing");
    build_transport(eb);
    eb.set_density("n", rho0);
    install_centered_ball(eb, enclosing_radius, "cutcell");
    eb.step(dt);
    const std::vector<double> eb1 = eb.get_state("n");

    const double d_enclosing = max_abs_diff(sq1, eb1);
    EXPECT_TRUE(d_enclosing == 0.0)
        << "(c2) uncut cut-cell EB is BIT-IDENTIQUE to Cartesian: max|diff| = " << d_enclosing
        << " (attendu 0)";
  }
}

TEST(FacadeRouting, GenericAnalyticLevelSetInstallsAfterBlockConstruction) {
#if defined(POPS_HAS_KOKKOS)
  (void)kokkos_scope();
#endif
  const int n = 24;
  const double L = 1.0;
  const double radius = 0.31;
  const NativeSystemConfig config = native_config(n, L, true);
  const std::vector<double> rho0 = ring_density(config);

  // The transport closure is deliberately built before geometry installation. The native program
  // owner must therefore accept the generic LevelSet without a geometry-specific authoring order.
  NativeSystem analytic(config);
  install_execution_lane(analytic, "pops.test.facade-routing.generic-level-set");
  build_transport(analytic);
  analytic.set_density("n", rho0);
  install_centered_ball(analytic, radius, "cutcell");
  const auto mask = analytic.embedded_boundary_mask();
  EXPECT_GT(std::count(mask.begin(), mask.end(), 1.0), 0);
  EXPECT_GT(std::count(mask.begin(), mask.end(), 0.0), 0);
  analytic.step(2e-4);
  EXPECT_TRUE(all_finite(analytic.get_state("n")));
}

TEST(FacadeRouting, AnalyticLevelSetReplacementIsTransactionalOnNonFiniteValues) {
#if defined(POPS_HAS_KOKKOS)
  (void)kokkos_scope();
#endif
  NativeSystem system(native_config(20, Real(1), false));
  install_execution_lane(system, "pops.test.facade-routing.transactional-level-set");
  system.set_analytic_level_set({"x", "constant", "sub"}, {0.0, 0.5, 0.0}, "staircase", 0.2, 1e-5,
                                0.1);
  const std::vector<double> original = system.embedded_boundary_mask();

  // (x - x) / 0 is structurally valid but non-finite at every sampled cell. Rejection must happen
  // before publishing either the new program, the new mask, the thresholds, or the routing mode.
  const std::vector<std::string> invalid_ops{"x", "x", "sub", "constant", "div"};
  const std::vector<double> invalid_literals{0.0, 0.0, 0.0, 0.0, 0.0};
  EXPECT_THROW(
      system.set_analytic_level_set(invalid_ops, invalid_literals, "cutcell", 0.3, 2e-5, 0.2),
      std::domain_error);
  EXPECT_EQ(original, system.embedded_boundary_mask());
}

TEST(FacadeRouting, PeriodicAnalyticLevelSetUsesTopologyAtTheSeam) {
#if defined(POPS_HAS_KOKKOS)
  (void)kokkos_scope();
#endif
  const int n = 24;
  const NativeSystemConfig config = native_config(n, Real(1), true);
  const std::vector<double> rho0 = periodic_seam_density(config);

  // The valid-cell expression x - 1/4 describes the same non-circular half-plane in both systems.
  // The reference spells out the low-side periodic extension only to make this regression observable:
  // a correct topology fill replaces that extension with the opposite valid cells and both prepared
  // metric fields become bit-identical. Direct evaluation at the fictitious x<0 ghost does not.
  NativeSystem topology(config);
  install_execution_lane(topology, "pops.test.facade-routing.periodic-topology");
  add_periodic_transport(topology);
  topology.set_density("n", rho0);
  topology.set_analytic_level_set({"x", "constant", "sub"}, {0.0, 0.25, 0.0}, "cutcell");
  install_forward_euler_program(topology);

  NativeSystem explicit_wrap(config);
  install_execution_lane(explicit_wrap, "pops.test.facade-routing.periodic-explicit-wrap");
  add_periodic_transport(explicit_wrap);
  explicit_wrap.set_density("n", rho0);
  explicit_wrap.set_analytic_level_set(
      {"x", "constant", "lt", "x", "constant", "add", "constant", "sub", "x", "constant", "sub",
       "where"},
      {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0}, "cutcell");
  install_forward_euler_program(explicit_wrap);

  ASSERT_EQ(topology.embedded_boundary_mask(), explicit_wrap.embedded_boundary_mask());
  topology.step(2e-4);
  explicit_wrap.step(2e-4);
  EXPECT_EQ(topology.get_state("n"), explicit_wrap.get_state("n"));
  EXPECT_GT(max_abs_diff(topology.get_state("n"), rho0), 0.0);
}

TEST(FacadeRouting, PrimitiveMaterializationFailsClosedWithoutMutatingAcceptedState) {
#if defined(POPS_HAS_KOKKOS)
  (void)kokkos_scope();
#endif
  constexpr int n = 4;
  const NativeSystemConfig config = native_config(n, Real(1), true);
  NativeSystem system(config);
  install_execution_lane(system, "pops.test.facade-routing.primitive-materialization");
  add_compressible(system);

  // All components are finite, but Euler conservative -> primitive is undefined at rho=0.
  // This exercises the real runtime registry and its prepared conversion, not a test-only callback.
  const std::vector<double> accepted(
      static_cast<std::size_t>(kCompressibleComponents) * cell_count(config.shape), 0.0);
  system.set_state("gas", accepted);

  bool rejected = false;
  try {
    (void)system.get_primitive_state("gas");
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    rejected = message.find("batch variable recovery failed") != std::string::npos;
  }
  EXPECT_TRUE(rejected);
  EXPECT_EQ(system.get_state("gas"), accepted)
      << "failed diagnostic recovery must not mutate the accepted conservative state";
}

TEST(FacadeRouting, PreparedBlockInstallationRefusesMissingBatchAuthorityWithoutPublication) {
#if defined(POPS_HAS_KOKKOS)
  (void)kokkos_scope();
#endif
  constexpr int n = 4;
  NativeSystem system(native_config(n, Real(1), true));
  install_execution_lane(system, "pops.test.facade-routing.prepared-block-refusal");
  system.install_block_state_route("gas", "test:facade-routing/gas/state");
  system.seal_auxiliary_providers();
  auto prepared = prepare_compiled_system_block<kTestDimension>(
      system, "gas", CompressibleModel::prepare(Real(1.4)), "none", "rusanov", "conservative",
      "explicit", 1.4, 1, true, 1);
  prepared.batch_conservative_to_primitive = {};

  bool rejected = false;
  try {
    system.install_prepared_block(std::move(prepared));
  } catch (const std::invalid_argument& error) {
    rejected = std::string(error.what()).find("complete exact-ranked execution contract") !=
               std::string::npos;
  }
  EXPECT_TRUE(rejected);
  EXPECT_EQ(system.n_blocks(), 0)
      << "missing prepared batch authority must not publish a partial block";
}

TEST(FacadeRouting, PrimitiveInputRequiresPreparedRecoveryBeforeConservativePublication) {
#if defined(POPS_HAS_KOKKOS)
  (void)kokkos_scope();
#endif
  constexpr int n = 4;
  const NativeSystemConfig config = native_config(n, Real(1), true);
  const std::size_t cells = cell_count(config.shape);
  NativeSystem system(config);
  install_execution_lane(system, "pops.test.facade-routing.primitive-input");
  add_compressible(system);

  std::vector<double> accepted(static_cast<std::size_t>(kCompressibleComponents) * cells, 0.0);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    accepted[cell] = 1.0;
    accepted[static_cast<std::size_t>(kCompressibleComponents - 1) * cells + cell] = 2.5;
  }
  system.set_state("gas", accepted);

  std::vector<double> inadmissible_primitive(
      static_cast<std::size_t>(kCompressibleComponents) * cells, 0.0);
  for (std::size_t cell = 0; cell < cells; ++cell)
    inadmissible_primitive[static_cast<std::size_t>(kCompressibleComponents - 1) * cells + cell] =
        1.0;
  EXPECT_THROW(system.set_primitive_state("gas", inadmissible_primitive), std::runtime_error);
  EXPECT_EQ(system.get_state("gas"), accepted)
      << "failed forward conversion validation must not publish a partial conservative state";

  std::vector<double> admissible_primitive(
      static_cast<std::size_t>(kCompressibleComponents) * cells, 0.0);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    admissible_primitive[cell] = 1.0;
    for (int axis = 0; axis < kTestDimension; ++axis)
      admissible_primitive[static_cast<std::size_t>(axis + 1) * cells + cell] =
          axis % 2 == 0 ? 0.2 : -0.1;
    admissible_primitive[static_cast<std::size_t>(kCompressibleComponents - 1) * cells + cell] =
        1.0;
  }
  EXPECT_NO_THROW(system.set_primitive_state("gas", admissible_primitive));
  const std::vector<double> recovered = system.get_primitive_state("gas");
  ASSERT_EQ(recovered.size(), admissible_primitive.size());
  for (std::size_t value = 0; value < recovered.size(); ++value)
    EXPECT_NEAR(recovered[value], admissible_primitive[value], 1e-12);
}
