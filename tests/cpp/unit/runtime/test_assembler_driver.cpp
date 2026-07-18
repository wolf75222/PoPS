// Scission Assembleur / Driver (retour tuteur sec.8.2 B). Un SystemAssembler ASSEMBLE les
// champs (Poisson de systeme + aux + residu de bloc) SANS avancer ; un SystemDriver AVANCE
// (et possede un assembleur). "advance un coupleur" devient "advance un driver".

#include <gtest/gtest.h>

#include <pops/core/model/coupled_system.hpp>
#include <pops/core/state/state.hpp>
#include <pops/coupling/amr/amr_coupler_mp.hpp>
#include <pops/coupling/single/coupler.hpp>
#include <pops/coupling/system/system_coupler.hpp>  // SystemAssembler, SystemDriver, SystemCoupler
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <cmath>
#include <cstdio>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

using namespace pops;

struct Scalar {
  using State = StateVec<1>;
  using Aux = pops::Aux;
  static constexpr int n_vars = 1;
  POPS_HD State flux(const State&, const Aux&, int) const { return State{}; }
  POPS_HD Real max_wave_speed(const State&, const Aux&, int) const { return Real(0); }
  POPS_HD State source(const State&, const Aux&) const { return State{}; }
  POPS_HD Real elliptic_rhs(const State& u) const { return u[0]; }
};

using Blk = EquationBlock<Scalar, FirstOrder, ExplicitTime<SSPRK2, 1>>;

namespace {

struct FactoryProbe {
  int constructions = 0;
  int solves = 0;
  FieldDistribution distribution = FieldDistribution::Replicated;
  std::vector<int> mapping;
};

// Deliberately does not expose the constructor used by GeometricMG. This proves that a uniform
// coupler depends on the factory protocol, not on a coincidentally compatible backend constructor.
class FactoryOnlyElliptic {
 public:
  FactoryOnlyElliptic(const Geometry& geom, const BoxArray& boxes,
                      const DistributionMapping& mapping, const BCRec& boundary,
                      ActiveRegionProvider2D active, FieldDistribution distribution,
                      FactoryProbe& probe)
      : geom_(geom),
        dm_(mapping),
        rhs_(boxes, dm_, 1, 0),
        phi_(boxes, dm_, 1, 1),
        distribution_(distribution),
        probe_(&probe),
        operator_contract_(make_materialized_elliptic_operator_contract(
            operator_identity(), geom_, boundary, active, distribution_, rhs_, phi_)) {}

  static constexpr EllipticOperatorIdentity operator_identity() noexcept {
    return {"pops.test.factory-only-operator", 1};
  }

  static EllipticOperatorContract expected_operator_contract(const EllipticBuildRequest& request) {
    return make_expected_elliptic_operator_contract(operator_identity(), request);
  }

  MultiFab& rhs() { return rhs_; }
  MultiFab& phi() { return phi_; }
  void solve() {
    ++probe_->solves;
    phi_.set_val(Real(0));
  }
  Real residual() const { return Real(0); }
  const Geometry& geom() const { return geom_; }
  FieldDistribution field_distribution() const noexcept { return distribution_; }
  const EllipticOperatorContract& prepared_operator_contract() const noexcept {
    return operator_contract_;
  }

 private:
  Geometry geom_;
  DistributionMapping dm_;
  MultiFab rhs_;
  MultiFab phi_;
  FieldDistribution distribution_;
  FactoryProbe* probe_;
  EllipticOperatorContract operator_contract_;
};

struct FactoryOnlyEllipticBuilder {
  FactoryProbe* probe;
  std::string contract{"pops.test.factory-only-elliptic@1"};

  [[nodiscard]] std::string_view collective_contract() const noexcept { return contract; }

  [[nodiscard]] EllipticOperatorContract expected_operator_contract(
      const EllipticBuildRequest& request) const {
    return FactoryOnlyElliptic::expected_operator_contract(request);
  }

  [[nodiscard]] FieldDistribution materialized_distribution(
      const EllipticBuildRequest& request) const noexcept {
    return request.distribution;
  }

  [[nodiscard]] bool supports(const EllipticBuildRequest&) const noexcept { return true; }

  EllipticFactoryBuildResult<FactoryOnlyElliptic> build(
      EllipticBuildRequest request) const noexcept {
    return capture_local_elliptic_factory_build<FactoryOnlyElliptic>([this, request = std::move(
                                                                                request)] {
      ++probe->constructions;
      probe->distribution = request.distribution;
      probe->mapping = request.mapping.ranks();
      return FactoryOnlyElliptic(request.geometry, request.boxes, request.mapping, request.boundary,
                                 std::move(request.active), request.distribution, *probe);
    });
  }
};

struct WrongDistributionEllipticBuilder {
  FactoryProbe* probe;
  std::string contract{"pops.test.wrong-distribution-elliptic@1"};

  [[nodiscard]] std::string_view collective_contract() const noexcept { return contract; }

  [[nodiscard]] EllipticOperatorContract expected_operator_contract(
      const EllipticBuildRequest& request) const {
    return FactoryOnlyElliptic::expected_operator_contract(request);
  }

  [[nodiscard]] FieldDistribution materialized_distribution(
      const EllipticBuildRequest&) const noexcept {
    return FieldDistribution::Distributed;
  }

  [[nodiscard]] bool supports(const EllipticBuildRequest&) const noexcept { return true; }

  EllipticFactoryBuildResult<FactoryOnlyElliptic> build(
      EllipticBuildRequest request) const noexcept {
    return capture_local_elliptic_factory_build<FactoryOnlyElliptic>([this, request = std::move(
                                                                                request)] {
      ++probe->constructions;
      return FactoryOnlyElliptic(request.geometry, request.boxes, request.mapping, request.boundary,
                                 std::move(request.active), request.distribution, *probe);
    });
  }
};

static_assert(EllipticSolver<FactoryOnlyElliptic>);
static_assert(EllipticFactory<FactoryOnlyEllipticBuilder, FactoryOnlyElliptic>);
static_assert(EllipticFactory<WrongDistributionEllipticBuilder, FactoryOnlyElliptic>);
static_assert(!std::constructible_from<FactoryOnlyElliptic, const Geometry&, const BoxArray&,
                                       const DistributionMapping&, const BCRec&,
                                       ActiveRegionProvider2D, FieldDistribution>);

}  // namespace

// SystemCoupler reste un alias du Driver (compat).
static_assert(std::is_same_v<SystemCoupler<CoupledSystem<Blk, Blk>, ChargeDensityRhs>,
                             SystemDriver<CoupledSystem<Blk, Blk>, ChargeDensityRhs>>);

TEST(AssemblerDriver, SplitAssemblerSolvesFieldsAndDriverAdvances) {
  const int n = 16;
  const Box2D dom = Box2D::from_extents(n, n);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const BoxArray ba(std::vector<Box2D>{dom});
  const DistributionMapping dm(1, n_ranks());
  BCRec bc;

  // charge a moyenne nulle : n0 = 1 + 0.25*signe(i<n/2), n1 = 1 -> f = -0.5*signe.
  MultiFab U0(ba, dm, 1, 2), U1(ba, dm, 1, 2);
  {
    Array4 a0 = U0.fab(0).array(), a1 = U1.fab(0).array();
    const Box2D g = U0.fab(0).grown_box();
    for (int j = g.lo[1]; j <= g.hi[1]; ++j)
      for (int i = g.lo[0]; i <= g.hi[0]; ++i) {
        a0(i, j, 0) = Real(1) + (i < n / 2 ? Real(0.25) : Real(-0.25));
        a1(i, j, 0) = Real(1);
      }
  }
  Blk b0{"a", Scalar{}, U0, bc}, b1{"b", Scalar{}, U1, bc};
  CoupledSystem system{b0, b1};
  ChargeDensityRhs charge{{{Real(-1), 0}, {Real(1), 0}}};

  // --- ASSEMBLEUR seul : resout les champs, expose phi/aux, ne fait AUCUN pas. ---
  SystemAssembler assembler(system, geom, ba, bc, charge);
  assembler.solve_fields();
  EXPECT_TRUE(norm_inf(assembler.phi()) > Real(1e-6)) << "assembler_phi_nonzero";
  // residu d'un bloc : Scalar -> flux et source nuls -> R = 0 (l'evaluateur tourne).
  MultiFab R(ba, dm, 1, 0);
  assembler.block_residual<NoSlope, RusanovFlux>(assembler.system().block<0>(),
                                                 assembler.system().block<0>().U(), R,
                                                 /*recompute_aux=*/false);
  EXPECT_TRUE(norm_inf(R) < Real(1e-14)) << "assembler_block_residual_zero";

  // --- DRIVER : avance (et possede un assembleur). ---
  MultiFab V0(ba, dm, 1, 2), V1(ba, dm, 1, 2);
  V0.set_val(Real(1));
  V1.set_val(Real(1));
  Blk d0{"a", Scalar{}, V0, bc}, d1{"b", Scalar{}, V1, bc};
  CoupledSystem dsys{d0, d1};
  SystemDriver driver(dsys, geom, ba, bc, charge);
  driver.step(Real(0.1));  // blocs explicites, flux/source nuls -> etat inchange, mais tourne
  EXPECT_TRUE(std::fabs(sum(V0, 0) - Real(1) * n * n) < Real(1e-12)) << "driver_step_runs";
  EXPECT_TRUE(norm_inf(driver.phi()) < Real(1e-9)) << "driver_phi_zero_for_neutral_balance";
}

TEST(AssemblerDriver, UniformCouplersUseDistributionAwareCustomFactory) {
  const int n = 8;
  const Box2D dom = Box2D::from_extents(n, n);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const BoxArray ba(std::vector<Box2D>{dom});
  const DistributionMapping dm(ba.size(), n_ranks());
  BCRec bc;

  MultiFab single_state(ba, dm, 1, 2);
  single_state.set_val(Real(0));
  FactoryProbe single_probe;
  Coupler<Scalar, FactoryOnlyElliptic> single(Scalar{}, geom, ba, bc, bc, {},
                                              ScalarFieldProvider2D{},
                                              FactoryOnlyEllipticBuilder{&single_probe});
  single.solve_fields(single_state);
  EXPECT_EQ(single_probe.constructions, 1);
  EXPECT_EQ(single_probe.solves, 1);
  EXPECT_EQ(single_probe.distribution, FieldDistribution::Distributed);
  EXPECT_EQ(single_probe.mapping, dm.ranks());

  MultiFab U0(ba, dm, 1, 2), U1(ba, dm, 1, 2);
  U0.set_val(Real(0));
  U1.set_val(Real(0));
  Blk b0{"a", Scalar{}, U0, bc}, b1{"b", Scalar{}, U1, bc};
  using System = CoupledSystem<Blk, Blk>;
  ChargeDensityRhs charge{{{Real(-1), 0}, {Real(1), 0}}};

  FactoryProbe assembler_probe;
  SystemAssembler<System, ChargeDensityRhs, FactoryOnlyElliptic> assembler(
      System{b0, b1}, geom, ba, bc, charge, {}, ScalarFieldProvider2D{},
      FactoryOnlyEllipticBuilder{&assembler_probe});
  assembler.solve_fields();
  EXPECT_EQ(assembler_probe.constructions, 1);
  EXPECT_EQ(assembler_probe.solves, 1);
  EXPECT_EQ(assembler_probe.distribution, FieldDistribution::Distributed);
  EXPECT_EQ(assembler_probe.mapping, dm.ranks());

  MultiFab V0(ba, dm, 1, 2), V1(ba, dm, 1, 2);
  V0.set_val(Real(0));
  V1.set_val(Real(0));
  Blk d0{"a", Scalar{}, V0, bc}, d1{"b", Scalar{}, V1, bc};
  FactoryProbe driver_probe;
  SystemDriver<System, ChargeDensityRhs, FactoryOnlyElliptic> driver(
      System{d0, d1}, geom, ba, bc, charge, {}, ScalarFieldProvider2D{},
      FactoryOnlyEllipticBuilder{&driver_probe});
  driver.solve_fields();
  EXPECT_EQ(driver_probe.constructions, 1);
  EXPECT_EQ(driver_probe.solves, 1);
  EXPECT_EQ(driver_probe.distribution, FieldDistribution::Distributed);
  EXPECT_EQ(driver_probe.mapping, dm.ranks());
}

TEST(AssemblerDriver, RejectsBackendThatIgnoresRequestedDistribution) {
  const Box2D domain = Box2D::from_extents(8, 8);
  const Geometry geometry{domain, 0.0, 1.0, 0.0, 1.0};
  const BoxArray boxes(std::vector<Box2D>{domain});
  const DistributionMapping mapping(std::vector<int>{my_rank()});
  FactoryProbe probe;

  EXPECT_THROW((void)make_elliptic_solver<FactoryOnlyElliptic>(
                   {geometry, boxes, mapping, BCRec{}, {}, FieldDistribution::Replicated},
                   WrongDistributionEllipticBuilder{&probe}),
               std::invalid_argument);
}

TEST(AssemblerDriver, ExactMappingReachesUniformAndAmrFactories) {
  const Box2D dom = Box2D::from_extents(8, 8);
  const Geometry geom{dom, 0.0, 1.0, 0.0, 1.0};
  const BoxArray ba = BoxArray::from_domain(dom, 4);
  std::vector<int> owners(static_cast<std::size_t>(ba.size()), 0);
  for (int box = 0; box < ba.size(); ++box)
    owners[static_cast<std::size_t>(box)] = n_ranks() > 1 ? (box + 1) % n_ranks() : 0;
  const DistributionMapping mapping(owners);
  if (n_ranks() > 1)
    EXPECT_NE(mapping.ranks(), DistributionMapping(ba.size(), n_ranks()).ranks());
  BCRec bc;

  FactoryProbe uniform_probe;
  Coupler<Scalar, FactoryOnlyElliptic> uniform(Scalar{}, geom, ba, mapping, bc, bc, {},
                                               ScalarFieldProvider2D{},
                                               FactoryOnlyEllipticBuilder{&uniform_probe});
  EXPECT_EQ(uniform_probe.mapping, mapping.ranks());
  EXPECT_EQ(uniform_probe.distribution, FieldDistribution::Distributed);

  MultiFab first(ba, mapping, 1, 2), second(ba, mapping, 1, 2);
  Blk first_block{"first", Scalar{}, first, bc}, second_block{"second", Scalar{}, second, bc};
  using System = CoupledSystem<Blk, Blk>;
  FactoryProbe system_probe;
  SystemAssembler<System, ChargeDensityRhs, FactoryOnlyElliptic> system(
      System{first_block, second_block}, geom, ba, bc,
      ChargeDensityRhs{{{Real(-1), 0}, {Real(1), 0}}}, {}, ScalarFieldProvider2D{},
      FactoryOnlyEllipticBuilder{&system_probe});
  EXPECT_EQ(system_probe.mapping, mapping.ranks());
  EXPECT_EQ(system_probe.distribution, FieldDistribution::Distributed);

  MultiFab coarse(ba, mapping, 1, 2);
  std::vector<AmrLevelMP> levels;
  levels.push_back(AmrLevelMP{std::move(coarse), nullptr, geom.dx(), geom.dy()});
  FactoryProbe amr_probe;
  AmrCouplerMP<Scalar, FactoryOnlyElliptic> amr(Scalar{}, geom, ba, bc, std::move(levels), {},
                                                /*replicated_coarse=*/false,
                                                FactoryOnlyEllipticBuilder{&amr_probe});
  EXPECT_EQ(amr_probe.mapping, mapping.ranks());
  EXPECT_EQ(amr_probe.distribution, FieldDistribution::Distributed);

  const DistributionMapping replicated_mapping(
      std::vector<int>(static_cast<std::size_t>(ba.size()), my_rank()));
  MultiFab replicated_coarse(ba, replicated_mapping, 1, 2);
  std::vector<AmrLevelMP> replicated_levels;
  replicated_levels.push_back(
      AmrLevelMP{std::move(replicated_coarse), nullptr, geom.dx(), geom.dy()});
  FactoryProbe replicated_probe;
  AmrCouplerMP<Scalar, FactoryOnlyElliptic> replicated_amr(
      Scalar{}, geom, ba, bc, std::move(replicated_levels), {},
      /*replicated_coarse=*/true, FactoryOnlyEllipticBuilder{&replicated_probe});
  EXPECT_EQ(replicated_probe.mapping, replicated_mapping.ranks());
  EXPECT_EQ(replicated_probe.distribution, FieldDistribution::Replicated);
}
