#include <gtest/gtest.h>

#include <pops/runtime/amr/tensor_composite_fac.hpp>

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Profiling.hpp>

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace {
using namespace pops;
using namespace pops::runtime::program::tensor_fac;
using Field = MultiFab<2>;

struct Configuration {
  Index<2> origin{0, 0};
  Extent<2> ratio{2, 2};
  bool replicated = false;
  bool periodic_seam = false;
  bool island = false;
  bool split_fine = false;
  bool conducting = false;
  bool mapped = false;
  int levels = 2;
  CoarseCorrectionMethod coarse_method = CoarseCorrectionMethod::gauss_seidel;
};

// The normal-band case is the sealed old-code counterexample: 8x4 coarse cells,
// coarse footprint i=3..4, and ratio two. MPI2 has a real empty fine owner.
// Variants alter one ownership/geometric mechanism at a time.
struct InterfaceProblem {
  const ExecutionLane& lane;
  Configuration configuration;
  std::vector<Geometry<2>> geometry;
  std::vector<PhysicalBoundaryConditions<2>> boundary;
  std::vector<mesh::BoxArray<2>> layout;
  std::vector<mesh::Distribution<2>> distribution;
  std::vector<std::unique_ptr<Field>> initial, rhs, solution;
  std::vector<std::array<std::unique_ptr<Field>, 4>> coefficient;
  std::vector<LevelBinding<2, Field::memory_space>> bindings;
  std::vector<amr::RefinementRatio<2>> ratios;
  elliptic::nd::CartesianTensorStencilOptions options;
  std::unique_ptr<FullTensorCompositeFac<2>> solver;

  InterfaceProblem(const ExecutionLane& execution_lane, InterfaceCoupling coupling,
                   Configuration config = {})
      : lane(execution_lane), configuration(config),
        layout(config.levels), distribution(config.levels), initial(config.levels),
        rhs(config.levels), solution(config.levels), coefficient(config.levels), bindings(config.levels),
        ratios(config.levels - 1, amr::RefinementRatio<2>{std::array<int, 2>{
            static_cast<int>(config.ratio[0]), static_cast<int>(config.ratio[1])}}),
        options{config.conducting ? 1u : 3u, config.conducting ? 2u : 0u, true} {
    const mesh::RankSpace<2> ranks{Index<2>{2, -3}, Extent<2>{lane.size(), 1}};
    const Index<2> local_rank = ranks.coordinate(static_cast<std::size_t>(lane.rank()));
    const Box<2> domain{config.origin, Index<2>{config.origin[0] + 7, config.origin[1] + 3}};
    geometry.push_back(Geometry<2>::from_bounds(domain, RealVector<2>{0, 0},
        RealVector<2>{1, config.mapped ? Real(2) * std::acos(Real(-1)) : Real(1)}));
    for (int level = 1; level < config.levels; ++level)
      geometry.push_back(geometry.back().refine(config.ratio));
    for (int level = 0; level < config.levels; ++level) {
      std::array<PhysicalBoundaryFace, 4> faces{};
      faces[0] = {PhysicalBoundaryKind::neumann, Real(0)};
      faces[1] = {config.conducting ? PhysicalBoundaryKind::dirichlet
                                  : PhysicalBoundaryKind::neumann, Real(0)};
      boundary.emplace_back(
          BoundaryTopology<2>::axis_periodic(std::array<bool, 2>{false, true}), faces,
          RealVector<2>{geometry[level].spacing(0), geometry[level].spacing(1)});
    }
    std::vector<Box<2>> coarse_boxes;
    std::vector<Index<2>> owners;
    const int partitions = config.replicated ? 1 : lane.size();
    for (int rank = 0; rank < partitions; ++rank) {
      coarse_boxes.push_back({Index<2>{domain.lo[0] + rank * 8 / partitions, domain.lo[1]},
                             Index<2>{domain.lo[0] + (rank + 1) * 8 / partitions - 1,
                                      domain.hi[1]}});
      owners.push_back(ranks.coordinate(static_cast<std::size_t>(rank)));
    }
    layout[0] = mesh::BoxArray<2>(std::move(coarse_boxes));
    distribution[0] = config.replicated
        ? mesh::Distribution<2>::replicated(layout[0], ranks)
        : mesh::Distribution<2>::partitioned(layout[0], ranks, std::move(owners));
    Box<2> footprint = domain;
    if (config.periodic_seam) {
      footprint.hi[1] = footprint.lo[1];
    } else {
      footprint.lo[0] += config.levels == 3 ? 2 : 3;
      footprint.hi[0] = footprint.lo[0] + (config.levels == 3 ? 3 : 1);
    }
    if (config.island) {
      ++footprint.lo[1];
      --footprint.hi[1];
    }
    std::vector<Box<2>> fine_boxes;
    owners.clear();
    if (config.split_fine) {
      // Divide at a coarse face: this is an actual same-level fine interface.
      const int tangent = config.periodic_seam ? 0 : 1;
      Box<2> first = footprint, second = footprint;
      first.hi[tangent] = footprint.lo[tangent] + static_cast<int>(footprint.length(tangent) / 2) - 1;
      second.lo[tangent] = first.hi[tangent] + 1;
      fine_boxes = {refine(first, config.ratio), refine(second, config.ratio)};
      owners = {ranks.coordinate(0), ranks.coordinate(static_cast<std::size_t>(lane.size() - 1))};
    } else {
      fine_boxes = {refine(footprint, config.ratio)};
      owners = {ranks.coordinate(static_cast<std::size_t>(lane.size() - 1))};
    }
    layout[1] = mesh::BoxArray<2>(std::move(fine_boxes));
    distribution[1] = mesh::Distribution<2>::partitioned(layout[1], ranks, std::move(owners));
    if (config.levels == 3) {
      Box<2> inner = layout[1][0];
      inner.lo[0] += 2;
      inner.hi[0] -= 2;
      layout[2] = mesh::BoxArray<2>(std::vector<Box<2>>{refine(inner, config.ratio)});
      distribution[2] = mesh::Distribution<2>::partitioned(layout[2], ranks,
          std::vector<Index<2>>{ranks.coordinate(0)});
    }
    for (int level = 0; level < config.levels; ++level) {
      initial[level] = std::make_unique<Field>(layout[level], distribution[level], local_rank,
                                              1, Extent<2>{1, 1});
      rhs[level] = std::make_unique<Field>(layout[level], distribution[level], local_rank, 1, Extent<2>{});
      solution[level] = std::make_unique<Field>(layout[level], distribution[level], local_rank,
                                               1, Extent<2>{1, 1});
      for (int entry = 0; entry < 4; ++entry) {
        coefficient[level][entry] = std::make_unique<Field>(layout[level], distribution[level],
                                                            local_rank, 1, Extent<2>{1, 1});
        coefficient[level][entry]->set_val(entry == 0 || entry == 3 ? Real(1) : Real(0));
      }
      bindings[level] = {&geometry[level], &boundary[level],
          {coefficient[level][0].get(), coefficient[level][1].get(),
           coefficient[level][2].get(), coefficient[level][3].get()},
          rhs[level].get(), initial[level].get(), solution[level].get()};
    }
    solver = std::make_unique<FullTensorCompositeFac<2>>(
        bindings, ratios, lane, options, config.coarse_method, 64,
        CoarsePreconditionerKind::diagonal, coupling);
  }

  static std::size_t offset(const Box<2>& box, int i, int j) {
    return static_cast<std::size_t>(i - box.lo[0]) +
        static_cast<std::size_t>(j - box.lo[1]) * static_cast<std::size_t>(box.length(0));
  }

  template <class Function>
  void fill(Field& field, const Geometry<2>& geom, Function function) {
    for (std::size_t local = 0; local < field.local_size(); ++local) {
      auto& fab = field.fab(local);
      auto values = fab.create_host_mirror();
      fab.copy_to_host(values);
      const auto box = fab.grown_box();
      for (int j = fab.box().lo[1]; j <= fab.box().hi[1]; ++j)
        for (int i = fab.box().lo[0]; i <= fab.box().hi[0]; ++i) {
          const Real x = geom.lower()[0] + (Real(i - geom.domain().lo[0]) + Real(0.5)) * geom.spacing(0);
          const Real y = geom.lower()[1] + (Real(j - geom.domain().lo[1]) + Real(0.5)) * geom.spacing(1);
          values(offset(box, i, j)) = function(x, y);
        }
      fab.copy_from_host(values);
    }
  }

  void set_state(int state, Real alpha = Real(1), Real beta = Real(0)) {
    for (int level = 0; level < configuration.levels; ++level)
      fill(*solution[level], geometry[level], [&](Real x, Real y) {
        if (state == 0) return level == 0 ? Real(0) : Real(1);
        if (state == 1) return Real(1);
        if (state == 4) return x * y;
        const Real theta = Real(2) * std::acos(Real(-1)) * y / geometry[level].upper()[1];
        if (state == 5) return (Real(1) - x * x) * std::cos(theta);
        const Real a = (Real(1) + Real(0.2) * level) *
            (x * x * std::cos(theta) + Real(0.4) * x * std::sin(Real(2) * theta));
        const Real b = (Real(1) - Real(0.1) * level) *
            (std::cos(Real(3) * theta) + x * std::sin(theta));
        return alpha * a + beta * b;
      });
  }

  void set_tensor(bool cross_terms) {
    for (int level = 0; level < configuration.levels; ++level)
      for (int entry = 0; entry < 4; ++entry)
        fill(*coefficient[level][entry], geometry[level], [&](Real x, Real y) {
          if (configuration.mapped) {
            if (entry == 0) return x;
            if (entry == 3) return Real(1) / x;
            return cross_terms ? (entry == 1 ? Real(0.125) : Real(-0.05)) * x * std::cos(y) : Real(0);
          }
          if (entry == 0 || entry == 3) return Real(1);
          return cross_terms ? (entry == 1 ? Real(0.3) : Real(-0.2)) : Real(0);
        });
  }

  bool covered(int level, const Index<2>& cell) const {
    if (level + 1 == configuration.levels) return false;
    for (const Box<2>& fine : layout[level + 1].boxes())
      if (coarsen(fine, configuration.ratio).contains(cell)) return true;
    return false;
  }

  std::vector<Real> image(int level) const {
    std::vector<Real> result(static_cast<std::size_t>(geometry[level].domain().numPts()), Real(0));
    if (!(level == 0 && configuration.replicated && lane.rank() != 0)) {
      const auto& residual = solver->current_residual(static_cast<std::size_t>(level));
      for (std::size_t local = 0; local < residual.local_size(); ++local) {
        const auto& fab = residual.fab(local);
        auto values = fab.create_host_mirror();
        fab.copy_to_host(values);
        for (int j = fab.box().lo[1]; j <= fab.box().hi[1]; ++j)
          for (int i = fab.box().lo[0]; i <= fab.box().hi[0]; ++i) {
            if (covered(level, Index<2>{i, j})) continue;
            result[offset(geometry[level].domain(), i, j)] = -values(offset(fab.grown_box(), i, j));
          }
      }
    }
    all_reduce_sum_inplace(result.data(), result.size(), lane.communicator());
    return result;
  }

  std::vector<Real> integral() const {
    std::vector<Real> result(static_cast<std::size_t>(configuration.levels + 1), Real(0));
    for (int level = 0; level < configuration.levels; ++level) {
      const Real volume = geometry[level].spacing(0) * geometry[level].spacing(1);
      for (Real value : image(level)) {
        result[static_cast<std::size_t>(level)] += volume * value;
        result.back() += volume * std::abs(value);
      }
    }
    return result;
  }

  void manufacture_rhs() {
    solver->evaluate_current_residual(lane);
    for (int level = 0; level < configuration.levels; ++level) {
      const auto& residual = solver->current_residual(static_cast<std::size_t>(level));
      for (std::size_t local = 0; local < rhs[level]->local_size(); ++local) {
        const auto input = std::as_const(residual.fab(local)).view();
        const auto output = rhs[level]->fab(local).view();
        for_each_cell(rhs[level]->box(local), KOKKOS_LAMBDA(const Index<2>& cell) {
          output(cell, 0) = -input(cell, 0);
        });
      }
      runtime::program::tensor_fac::detail::copy_valid<2>(*initial[level], *solution[level]);
    }
    Kokkos::fence();
  }
};

TEST(TensorFacConservativeInterface, ClosedJumpHasTwelveLegacyImbalanceAndZeroFineFluxBalance) {
  const auto lane = ExecutionLane::duplicate_world_collectively("tests.tensor-fac.flux-jump");
  if (lane.size() != 1 && lane.size() != 2) GTEST_SKIP();
  for (const bool replicated : {false, true}) {
    Configuration config;
    config.replicated = replicated;
    InterfaceProblem legacy(lane, InterfaceCoupling::level_stencil, config);
    InterfaceProblem conservative(lane, InterfaceCoupling::fine_flux, config);
    EXPECT_NE(legacy.solver->exact_prepared_contract(), conservative.solver->exact_prepared_contract());
    legacy.set_state(0);
    conservative.set_state(0);
    legacy.solver->evaluate_current_residual(lane);
    conservative.solver->evaluate_current_residual(lane);
    const auto old_integral = legacy.integral(), new_integral = conservative.integral();
    EXPECT_DOUBLE_EQ(old_integral[0], -16);
    EXPECT_DOUBLE_EQ(old_integral[1], 28);
    EXPECT_DOUBLE_EQ(old_integral[0] + old_integral[1], 12);
    EXPECT_DOUBLE_EQ(new_integral[0], -28);
    EXPECT_DOUBLE_EQ(new_integral[1], 28);
    EXPECT_DOUBLE_EQ(new_integral[0] + new_integral[1], 0);
    EXPECT_EQ(all_reduce_max(conservative.solution[1]->local_size() == 0 ? 1L : 0L, lane),
              lane.size() == 2 ? 1 : 0);
    conservative.set_state(1);
    conservative.solver->evaluate_current_residual(lane);
    EXPECT_DOUBLE_EQ(conservative.integral()[2], 0);
  }
}

TEST(TensorFacConservativeInterface, PeriodicOriginsAnisotropicChildrenAndMappedCrossTermsConserve) {
  const auto lane = ExecutionLane::duplicate_world_collectively("tests.tensor-fac.flux-geometry");
  if (lane.size() != 1 && lane.size() != 2) GTEST_SKIP();
  for (int shape = 0; shape < 3; ++shape)
    for (const bool replicated : {false, true}) {
      Configuration config;
      config.origin = Index<2>{-4, 3};
      config.ratio = Extent<2>{2, 3};
      config.replicated = replicated;
      config.periodic_seam = shape == 1;
      config.island = shape == 2;
      config.split_fine = true;
      config.mapped = true;
      InterfaceProblem problem(lane, InterfaceCoupling::fine_flux, config);
      problem.set_tensor(true);
      problem.set_state(2);
      problem.solver->evaluate_current_residual(lane);
      const auto integral = problem.integral();
      EXPECT_TRUE(std::isfinite(integral[2]));
      EXPECT_NEAR(integral[0] + integral[1], Real(0),
                  Real(512) * std::numeric_limits<Real>::epsilon() * (Real(1) + integral[2]));
    }
}

TEST(TensorFacConservativeInterface, RetainsNonsymmetricCrossDerivativeAndIsLinear) {
  const auto lane = ExecutionLane::duplicate_world_collectively("tests.tensor-fac.flux-linearity");
  if (lane.size() != 1 && lane.size() != 2) GTEST_SKIP();
  InterfaceProblem problem(lane, InterfaceCoupling::fine_flux);
  problem.set_state(4);
  problem.set_tensor(true);
  problem.solver->evaluate_current_residual(lane);
  const auto full = problem.image(1);
  problem.set_tensor(false);
  problem.solver->evaluate_current_residual(lane);
  const auto diagonal = problem.image(1);
  // At an interior fine cell, u=x*y gives -div(A grad u)=-(A01+A10)=-0.1.
  const auto index = InterfaceProblem::offset(problem.geometry[1].domain(), 7, 3);
  EXPECT_NEAR(full[index], Real(-0.1), Real(128) * std::numeric_limits<Real>::epsilon());
  EXPECT_NEAR(diagonal[index], Real(0), Real(128) * std::numeric_limits<Real>::epsilon());
  problem.set_tensor(true);
  problem.set_state(2, Real(1), Real(0));
  problem.solver->evaluate_current_residual(lane);
  const std::array<std::vector<Real>, 2> a{problem.image(0), problem.image(1)};
  problem.set_state(2, Real(0), Real(1));
  problem.solver->evaluate_current_residual(lane);
  const std::array<std::vector<Real>, 2> b{problem.image(0), problem.image(1)};
  problem.set_state(2, Real(0.75), Real(-0.25));
  problem.solver->evaluate_current_residual(lane);
  for (int level = 0; level < 2; ++level) {
    const auto combined = problem.image(level);
    for (std::size_t cell = 0; cell < combined.size(); ++cell) {
      const Real expected = Real(0.75) * a[level][cell] - Real(0.25) * b[level][cell];
      EXPECT_NEAR(combined[cell], expected,
          Real(512) * std::numeric_limits<Real>::epsilon() *
              (Real(1) + std::abs(a[level][cell]) + std::abs(b[level][cell])));
    }
  }
}

TEST(TensorFacConservativeInterface, ManufacturedSelectedImageIsTheSolveFixedPoint) {
  const auto lane = ExecutionLane::duplicate_world_collectively("tests.tensor-fac.flux-fixed-point");
  if (lane.size() != 1 && lane.size() != 2) GTEST_SKIP();
  Configuration config;
  config.conducting = true;
  InterfaceProblem problem(lane, InterfaceCoupling::fine_flux, config);
  problem.set_tensor(true);
  problem.set_state(2);
  problem.manufacture_rhs();
  Controls controls;
  controls.interface_coupling = InterfaceCoupling::fine_flux;
  controls.absolute_tolerance = Real(1e-12);
  const auto report = problem.solver->solve(controls, lane);
  ASSERT_TRUE(report.solved()) << report.reason;
  EXPECT_EQ(report.iters, 0);
  EXPECT_LE(report.residual_norm, controls.absolute_tolerance);
  problem.solver->evaluate_current_residual(lane);
  EXPECT_LE(problem.integral()[2], controls.absolute_tolerance);
}

TEST(TensorFacConservativeInterface, ThreeLevelsConvergeToTheirSelectedConservativeImage) {
  const auto lane = ExecutionLane::duplicate_world_collectively("tests.tensor-fac.flux-three-level");
  if (lane.size() != 1 && lane.size() != 2) GTEST_SKIP();
  Configuration config;
  config.levels = 3;
  config.conducting = true;
  config.coarse_method = CoarseCorrectionMethod::gmres;
  InterfaceProblem problem(lane, InterfaceCoupling::fine_flux, config);
  problem.set_tensor(true);
  problem.set_state(5);
  problem.manufacture_rhs();
  for (auto& initial : problem.initial) initial->set_val(Real(0));
  // These outer controls match the authored Hoffart case. The small Cartesian root uses the
  // ordinary fixed diagonal GMRES preconditioner, with its unchanged true-stencil verifier.
  Controls controls;
  controls.interface_coupling = InterfaceCoupling::fine_flux;
  controls.coarse_method = CoarseCorrectionMethod::gmres;
  controls.relative_tolerance = Real(1e-10);
  controls.absolute_tolerance = Real(1e-12);
  controls.maximum_iterations = 300;
  controls.fine_sweeps = 64;
  controls.coarse_cycles = 512;
  controls.correction_damping = Real(0.5);
  const auto report = problem.solver->solve(controls, lane);
  ASSERT_TRUE(report.solved()) << report.reason << " residual=" << report.residual_norm
                              << " iterations=" << report.iters;
  EXPECT_GT(report.iters, 0);
  const Real stop = std::max(controls.absolute_tolerance,
      controls.relative_tolerance * static_cast<Real>(report.reference_residual_norm));
  problem.solver->evaluate_current_residual(lane);
  for (int level = 0; level < config.levels; ++level)
    for (const Real residual : problem.image(level))
      EXPECT_LE(std::abs(residual), stop);
}

TEST(TensorFacConservativeInterface, ConstructionAuthenticatesCouplingAcrossRanks) {
  const auto lane = ExecutionLane::duplicate_world_collectively("tests.tensor-fac.flux-construction");
  if (lane.size() != 1 && lane.size() != 2) GTEST_SKIP();
  const auto invalid = lane.rank() == 0 ? static_cast<InterfaceCoupling>(2)
                                      : InterfaceCoupling::fine_flux;
  EXPECT_ANY_THROW(InterfaceProblem(lane, invalid));
  if (lane.size() == 2) {
    const auto asymmetric = lane.rank() == 0 ? InterfaceCoupling::level_stencil
                                           : InterfaceCoupling::fine_flux;
    EXPECT_THROW(InterfaceProblem(lane, asymmetric), std::invalid_argument);
  }
  EXPECT_NO_THROW(InterfaceProblem(lane, InterfaceCoupling::fine_flux));
}

TEST(TensorFacConservativeInterface, RejectsAsymmetricControlsReplicasAndInvalidCoefficientsCollectively) {
  const auto lane = ExecutionLane::duplicate_world_collectively("tests.tensor-fac.flux-refusal");
  if (lane.size() != 1 && lane.size() != 2) GTEST_SKIP();
  Configuration config;
  config.replicated = true;
  InterfaceProblem problem(lane, InterfaceCoupling::fine_flux, config);
  problem.set_state(1);
  EXPECT_NO_THROW(problem.solver->evaluate_current_residual(lane));
  Controls controls;
  controls.interface_coupling = lane.rank() == 0 ? InterfaceCoupling::level_stencil
                                              : InterfaceCoupling::fine_flux;
  EXPECT_THROW(problem.solver->solve(controls, lane), std::invalid_argument);
  controls.interface_coupling = InterfaceCoupling::fine_flux;
  controls.fine_sweeps = lane.rank() == 0 ? 0 : 4;
  EXPECT_THROW(problem.solver->solve(controls, lane), std::invalid_argument);
  if (lane.size() == 2) {
    controls.fine_sweeps = lane.rank() == 0 ? 4 : 6;
    EXPECT_THROW(problem.solver->solve(controls, lane), std::invalid_argument);
    for (Field* field : {problem.solution[0].get(), problem.rhs[0].get(), problem.coefficient[0][1].get()}) {
      field->set_val(lane.rank() == 0 ? Real(0) : Real(0.125));
      EXPECT_THROW(problem.solver->evaluate_current_residual(lane), std::runtime_error);
      field->set_val(Real(0));
      EXPECT_NO_THROW(problem.solver->evaluate_current_residual(lane));
    }
  }
  // A fine owner alone supplies invalid input; empty owners must reach the same refusal.
  problem.coefficient[1][0]->set_val(std::numeric_limits<Real>::quiet_NaN());
  EXPECT_THROW(problem.solver->evaluate_current_residual(lane), std::runtime_error);
  problem.coefficient[1][0]->set_val(Real(1));
  EXPECT_NO_THROW(problem.solver->evaluate_current_residual(lane));
  problem.coefficient[1][1]->set_val(Real(2));
  EXPECT_THROW(problem.solver->evaluate_current_residual(lane), std::runtime_error);
  problem.coefficient[1][1]->set_val(Real(0));
  EXPECT_NO_THROW(problem.solver->evaluate_current_residual(lane));
  problem.solution[1]->set_val(std::numeric_limits<Real>::infinity());
  EXPECT_THROW(problem.solver->evaluate_current_residual(lane), std::runtime_error);
  problem.solution[1]->set_val(Real(0));
  EXPECT_NO_THROW(problem.solver->evaluate_current_residual(lane));
}

std::atomic<std::size_t> data_allocations{0};
Kokkos::Tools::allocateDataFunction previous_allocate = nullptr;

void count_allocation(Kokkos::Tools::SpaceHandle space, const char* label, const void* pointer,
                      const std::uint64_t size) {
  ++data_allocations;
  if (previous_allocate) previous_allocate(space, label, pointer, size);
}

struct AllocationObserver {
  AllocationObserver() {
    previous_allocate = Kokkos::Tools::Experimental::get_callbacks().allocate_data;
    data_allocations.store(0);
    Kokkos::Tools::Experimental::set_allocate_data_callback(count_allocation);
  }
  ~AllocationObserver() {
    Kokkos::Tools::Experimental::set_allocate_data_callback(previous_allocate);
    previous_allocate = nullptr;
  }
};

TEST(TensorFacConservativeInterface, RepeatedResidualApplicationsReusePreparedDeviceStorage) {
  const auto lane = ExecutionLane::duplicate_world_collectively("tests.tensor-fac.flux-storage");
  if (lane.size() != 1 && lane.size() != 2) GTEST_SKIP();
  Configuration config;
  config.replicated = true;
  InterfaceProblem problem(lane, InterfaceCoupling::fine_flux, config);
  problem.set_state(2);
  problem.set_tensor(true);
  problem.solver->evaluate_current_residual(lane);
  {
    AllocationObserver observer;
    Kokkos::View<Real*> calibration("tensor_flux_allocation_witness", 1);
    Kokkos::fence();
    ASSERT_GT(data_allocations.load(), 0u) << "allocation observer must witness a real allocation";
    data_allocations.store(0);
    problem.solver->evaluate_current_residual(lane);
    problem.solver->evaluate_current_residual(lane);
    Kokkos::fence();
    EXPECT_EQ(data_allocations.load(), 0u);
  }
}
}  // namespace
