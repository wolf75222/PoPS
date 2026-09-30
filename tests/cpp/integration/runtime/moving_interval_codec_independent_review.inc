// Independent codec injections adapted to 2df5090 + 2a741ef. Root owns compilation/reception.
#include <pops/runtime/program/moving_interval_checkpoint.hpp>

namespace {
struct IndependentMovingCodecFixture {
  using Geometry = pops::runtime::program::MovingIntervalGeometry<1>;
  pops::MultiFab<1> state;
  Geometry geometry, declaration;
  pops::runtime::program::AcceptedExchangeLedger ledger;

  explicit IndependentMovingCodecFixture(pops::Real tolerance = pops::Real(1e-13)) {
    const pops::Box<1> box{pops::Index<1>{0}, pops::Index<1>{1}};
    const pops::mesh::BoxArray<1> layout(std::vector<pops::Box<1>>{box});
    const auto distribution = pops::mesh::Distribution<1>::replicated(
        layout, pops::mesh::RankSpace<1>{pops::Index<1>{0}, pops::Extent<1>{1}});
    state = pops::MultiFab<1>(layout, distribution, pops::Index<1>{0}, 1, pops::Extent<1>{0});
    state.set_val(pops::Real(2));
    geometry.runtime_block = 0;
    geometry.physical_frame = "review-fixed-interval";
    geometry.clock_authority = "review-clock";
    geometry.geometry_tolerance_authority = tolerance;
    geometry.measures = state;
    geometry.measures.set_val(pops::Real(1));
    geometry.coordinates.emplace_back(box, 1);
    geometry.swept_volumes.emplace_back(box, 1);
    auto coordinates = geometry.coordinates[0].template field<0>().create_host_mirror();
    for (std::size_t face = 0; face < 3; ++face)
      coordinates(face) = static_cast<pops::Real>(face);
    geometry.coordinates[0].template field<0>().copy_from_host(coordinates);
    geometry.swept_volumes[0].set_val(pops::Real(0));
    declaration = geometry;
    geometry.generation = 1;
    auto& receipt = geometry.last_receipt.emplace();
    receipt.point.clock = "review-clock";
    receipt.point.tick = 0;
    receipt.point.level = receipt.point.stage = receipt.point.substep = 0;
    receipt.point.stage_fraction = pops::amr::Rational(0, 1);
    receipt.point.dt = .1;
    receipt.point.physical_time = 0.;
    receipt.physical_frame = geometry.physical_frame;
    receipt.quadrature_identity = "endpoint-swept@1";
    receipt.geometry_tolerance = tolerance;
    receipt.previous_state = state;
    receipt.previous_measures = geometry.measures;
    receipt.previous_coordinates = geometry.coordinates;
    receipt.integrated_source = state;
    receipt.integrated_source.set_val(pops::Real(0));
    receipt.physical_flux.emplace_back(box, 1);
    receipt.face_density.emplace_back(box, 1);
    receipt.physical_flux[0].set_val(pops::Real(0));
    receipt.face_density[0].set_val(pops::Real(2));
    pops::runtime::program::ExchangeRecord point{
        "moving-interval", "interval", "mesh", "endpoint-swept@1", 1, 1., 0., 1., 1};
    point.qualify_runtime_point(receipt.point);
    geometry.last_interval = point.evaluation_context;
    for (int cell = 0; cell < 2; ++cell) {
      stage({"source:mesh/component:0", "cell:" + std::to_string(cell), "mesh", "endpoint-swept@1",
             1, 1., 0., 1., 1});
      for (int side = 0; side < 2; ++side) {
        const auto occurrence = "cell:" + std::to_string(cell) + "/side:" + std::to_string(side);
        const int outward = side ? 1 : -1;
        stage({"amount:mesh/component:0", occurrence, "mesh", "endpoint-swept@1", outward, 1., 0.,
               1., 1});
        stage({"geometry:mesh", occurrence, "mesh", "endpoint-swept@1", outward, 1., 0., 1., 1});
      }
    }
  }
  void stage(pops::runtime::program::ExchangeRecord record) {
    record.qualify_runtime_point(geometry.last_receipt->point);
    ledger.stage(std::move(record));
  }
  auto image() const {
    return pops::runtime::program::checkpoint_moving_intervals(
        ledger, std::map<std::string, Geometry>{{"mesh", geometry}},
        [&](int) -> const pops::MultiFab<1>& { return state; });
  }
  auto read(const std::vector<std::uint8_t>& bytes, const pops::MultiFab<1>& installed) const {
    return pops::runtime::program::read_moving_checkpoint(
        bytes, ledger, std::map<std::string, Geometry>{{"mesh", declaration}},
        [&](int) -> const pops::MultiFab<1>& { return installed; });
  }
};
}  // namespace

TEST(ProgramRuntime, IndependentMovingCodecRejectsMeasureCountermodelRegardlessOfWireTolerance) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  IndependentMovingCodecFixture fixture(pops::Real(2));
  ASSERT_NO_THROW(fixture.read(fixture.image(), fixture.state));
  // Same endpoints and exact conserved amount; only the forged measure/state pair changes.
  fixture.geometry.measures.set_val(pops::Real(2));
  fixture.state.set_val(pops::Real(1));
  fixture.geometry.last_receipt->geometry_tolerance = 2;
  EXPECT_ANY_THROW(fixture.read(fixture.image(), fixture.state));
}

TEST(ProgramRuntime, IndependentMovingCodecRejectsOverflowInIndependentReynoldsBalance) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  IndependentMovingCodecFixture fixture;
  ASSERT_NO_THROW(fixture.read(fixture.image(), fixture.state));
  // Keep measures equal to endpoint differences, but make q1 overflow from finite factors.
  fixture.geometry.measures.set_val(pops::Real(2));
  fixture.geometry.last_receipt->previous_measures.set_val(pops::Real(2));
  for (auto* faces :
       {&fixture.geometry.coordinates, &fixture.geometry.last_receipt->previous_coordinates,
        &fixture.declaration.coordinates}) {
    auto host = (*faces)[0].template field<0>().create_host_mirror();
    for (std::size_t face = 0; face < 3; ++face)
      host(face) = pops::Real(2) * static_cast<pops::Real>(face);
    (*faces)[0].template field<0>().copy_from_host(host);
  }
  fixture.state.set_val(pops::Real(1e308));
  EXPECT_ANY_THROW(fixture.read(fixture.image(), fixture.state));
}

TEST(ProgramRuntime, IndependentMovingCodecAuthenticatesComponentTopologyAndCredibleCounts) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  IndependentMovingCodecFixture fixture;
  auto bytes = fixture.image();
  ASSERT_NO_THROW(fixture.read(bytes, fixture.state));
  pops::MultiFab<1> foreign(fixture.state.layout(), fixture.state.distribution(),
                            fixture.state.local_rank(), 2, fixture.state.ghosts());
  EXPECT_ANY_THROW(fixture.read(bytes, foreign));
  for (std::size_t index = 8; index < 16; ++index)
    bytes[index] = 255;  // hostile nested ledger length, before its allocation
  EXPECT_ANY_THROW(fixture.read(bytes, fixture.state));
}

TEST(ProgramRuntime, IndependentMovingCodecRejectsInventedExteriorSupportOnInternalReceipt) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  IndependentMovingCodecFixture fixture;
  fixture.ledger.declare_integral("q", .7);
  ASSERT_NO_THROW(fixture.read(fixture.image(), fixture.state));
  pops::runtime::program::AcceptedExchangeLedger altered;
  altered.declare_integral(
      "q", .7);  // preserve support metadata through the actual POPSEX02 inner image
  for (auto record : fixture.ledger.records()) {
    record.trace_axis = 0;
    record.trace_side = 1;
    record.trace_component = 0;
    record.exterior_trace = true;
    // Runtime context and its evaluation suffix remain genuine, as do all amounts.
    altered.stage(std::move(record));
  }
  fixture.ledger.swap(altered);
  EXPECT_ANY_THROW(fixture.read(fixture.image(), fixture.state));
}

TEST(ProgramRuntime, IndependentMovingCodecRequiresExactEndTimeAndMacroStep) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  IndependentMovingCodecFixture fixture;
  auto candidate = fixture.read(fixture.image(), fixture.state);
  ASSERT_NO_THROW(pops::runtime::program::require_moving_checkpoint_lifecycle(candidate, .1, 1));
  EXPECT_ANY_THROW(pops::runtime::program::require_moving_checkpoint_lifecycle(
      candidate, std::nextafter(.1, 1.), 1));
  EXPECT_ANY_THROW(pops::runtime::program::require_moving_checkpoint_lifecycle(candidate, .1, 0));
  EXPECT_ANY_THROW(pops::runtime::program::require_moving_checkpoint_lifecycle(candidate, .1, 2));
  EXPECT_ANY_THROW(pops::runtime::program::require_moving_checkpoint_lifecycle(
      candidate, std::numeric_limits<double>::infinity(), 1));
  // Keep geometry/ledger valid; only change the separate declaration used by restore.
  fixture.declaration.clock_authority = "other-clock";
  EXPECT_ANY_THROW(fixture.read(fixture.image(), fixture.state));
}

TEST(ProgramRuntime, IndependentMovingCaptureRejectsNonterminalOrNestedStaging) {
  if constexpr (kNativeDimension != 1) {
    GTEST_SKIP()
        << "System moving provider requires native Dim1; direct codec tests are ranked separately";
  }
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  System<kNativeDimension> sim(distributed_boundary_domain_config<kNativeDimension>(8));
  install_execution_lane(sim, "pops.test.independent-moving-stage-depth");
  add_gas(sim, 1.4);
  std::vector<double> initial;
  fill_ic(initial, 8, 1.4);
  sim.set_state("gas", initial);
  sim.set_program_block_map({0});
  auto context = runtime::program::make_program_execution_provider(&sim);
  context->configure_primary_clock("review-clock");
  context->initialize_moving_interval_geometry("mesh", 0, "unit-interval", Real(1e-13));
  ASSERT_NO_THROW(sim.checkpoint_program_exchanges());
  const auto before = sim.checkpoint_program_exchanges();
  sim.begin_step_transaction();
  ASSERT_EQ(sim.step_transaction_depth(), 1u);
  EXPECT_ANY_THROW(sim.checkpoint_program_exchanges());
  EXPECT_ANY_THROW(sim.checkpoint_program_exchanges(true));  // no terminal receipt exists yet
  sim.begin_step_transaction();
  ASSERT_EQ(sim.step_transaction_depth(), 2u);
  EXPECT_ANY_THROW(sim.checkpoint_program_exchanges(true));
  sim.rollback_step_transaction();
  sim.rollback_step_transaction();
  EXPECT_EQ(sim.checkpoint_program_exchanges(), before);
  sim.begin_restart_transaction();
  EXPECT_ANY_THROW(sim.checkpoint_program_exchanges(true));
  sim.rollback_restart_transaction();
  EXPECT_EQ(sim.checkpoint_program_exchanges(), before);
}
