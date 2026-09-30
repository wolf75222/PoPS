// Independent POPSEX03/04 probes against 99d651c1. Root owns native reception.
// The expected interior amounts are literal binary64 values proved by Fraction;
// neither the fixture's Reynolds balance nor its ledger uses the production helper.
namespace {
struct IndependentRelativeQuantityFixture : IndependentMovingCodecFixture {
  using Convention = pops::runtime::program::MovingRelativeAmountConvention;
  std::array<double, 3> physical{}, density{}, fused{}, ordinary{};
  double sweep = 0;

  explicit IndependentRelativeQuantityFixture(bool cab = false, bool legacy = false) {
    const auto layout = state.layout();
    const auto distribution = state.distribution();
    const auto box = state.box(0);
    state = pops::MultiFab<1>(layout, distribution, pops::Index<1>{0}, 3, pops::Extent<1>{0});
    auto& receipt = *geometry.last_receipt;
    const std::array<double, 3> factors = cab ? std::array<double, 3>{.5, 2., 1.}
                                             : std::array<double, 3>{.5, 1., 2.};
    const double p = cab ? 0x1.134759d5cf776p-9 : -0x1.d7e1ff2b2db0ap-11;
    const double u = cab ? 0x1.8009d3c5cf16ep+1 : 0x1.8004eaba28d6cp+1;
    const double s = cab ? 0x1.da88051ea9000p-15 : -0x1.da88051ea8000p-15;
    const double f = cab ? 0x1.fa10cfbb6fc95p-10 : -0x1.7ee75a8e3ebabp-11;
    const double o = cab ? 0x1.fa10cfbb6fc96p-10 : -0x1.7ee75a8e3ebaap-11;
    sweep = legacy ? 0. : s;
    geometry.checkpoint_wire_version = legacy ? 3 : 4;
    receipt.relative_amount_convention = legacy ? Convention::kLegacyExpression : Convention::kFused;
    receipt.previous_state = state;
    receipt.integrated_source = state;
    receipt.integrated_source.set_val(0.);
    receipt.physical_flux.clear(); receipt.physical_flux.emplace_back(box, 3);
    receipt.face_density.clear(); receipt.face_density.emplace_back(box, 3);
    auto previous = receipt.previous_state.fab(0).create_host_mirror();
    auto current = state.fab(0).create_host_mirror();
    auto flux = receipt.physical_flux[0].template field<0>().create_host_mirror();
    auto trace = receipt.face_density[0].template field<0>().create_host_mirror();
    for (int component = 0; component < 3; ++component) {
      physical[component] = p * factors[component];
      density[component] = u * factors[component];
      fused[component] = legacy ? physical[component] : f * factors[component];
      ordinary[component] = legacy ? physical[component] : o * factors[component];
      for (int face = 0; face < 3; ++face) {
        flux(face + 3 * component) = physical[component];
        trace(face + 3 * component) = density[component];
      }
      for (int cell = 0; cell < 2; ++cell) {
        previous(cell + 2 * component) = density[component];
        const double left = cell == 0 ? physical[component] : fused[component];
        const double right = cell == 0 ? fused[component] : physical[component];
        const double volume = cell == 0 ? .125 + sweep : .25 - (.125 + sweep);
        current(cell + 2 * component) = (density[component] * .125 + left - right) / volume;
      }
    }
    receipt.previous_state.fab(0).copy_from_host(previous);
    state.fab(0).copy_from_host(current);
    receipt.physical_flux[0].template field<0>().copy_from_host(flux);
    receipt.face_density[0].template field<0>().copy_from_host(trace);
    auto nodes = geometry.coordinates[0].template field<0>().create_host_mirror();
    nodes(0) = 0.; nodes(1) = .125; nodes(2) = .25;
    geometry.coordinates[0].template field<0>().copy_from_host(nodes);
    receipt.previous_coordinates = geometry.coordinates;
    receipt.previous_measures.set_val(.125);
    declaration = geometry;
    declaration.generation = 0; declaration.last_receipt.reset(); declaration.last_interval.clear();
    declaration.measures.set_val(.125);
    nodes(1) = .125 + sweep;
    geometry.coordinates[0].template field<0>().copy_from_host(nodes);
    auto volumes = geometry.measures.fab(0).create_host_mirror();
    volumes(0) = nodes(1) - nodes(0); volumes(1) = nodes(2) - nodes(1);
    geometry.measures.fab(0).copy_from_host(volumes);
    auto sweeps = geometry.swept_volumes[0].template field<0>().create_host_mirror();
    sweeps(0) = sweeps(2) = 0.; sweeps(1) = sweep;
    geometry.swept_volumes[0].template field<0>().copy_from_host(sweeps);
    rebuild_ledger(false);
  }

  void rebuild_ledger(bool corrupt) {
    ledger.clear();
    for (int cell = 0; cell < 2; ++cell) {
      for (int component = 0; component < 3; ++component) {
        stage({"source:mesh/component:" + std::to_string(component), "cell:" + std::to_string(cell),
               "mesh", "endpoint-swept@1", 1, 1., 0., 1., 1});
        for (int side = 0; side < 2; ++side) {
          const int face = cell + side;
          double quantity = face == 1 ? fused[component] : physical[component];
          if (corrupt && cell == 0 && side == 1 && component == 1) quantity = ordinary[component];
          const auto occurrence = "cell:" + std::to_string(cell) + "/side:" + std::to_string(side);
          stage({"amount:mesh/component:" + std::to_string(component), occurrence, "mesh",
                 "endpoint-swept@1", side ? 1 : -1, 1., quantity, 1., 1});
          if (component == 0)
            stage({"geometry:mesh", occurrence, "mesh", "endpoint-swept@1", side ? 1 : -1,
                   1., face == 1 ? sweep : 0., 1., 1});
        }
      }
    }
  }

  // Deliberately bypasses validate_local so reader injections reach the true
  // decoder. The historical layout contains no arithmetic tag whatsoever.
  std::vector<std::uint8_t> raw_image(bool historical, std::uint64_t tag = 1) const {
    using namespace pops::runtime::program::moving_checkpoint_detail;
    Writer out;
    const std::array<std::uint8_t, 8> header{'P','O','P','S','E','X','0',
                                           static_cast<std::uint8_t>(historical ? '3' : '4')};
    out.raw(header); out.bytes(ledger.checkpoint(true)); out.size(1);
    out.string("mesh"); out.i32(geometry.runtime_block); out.string(geometry.physical_frame);
    out.string(geometry.clock_authority); out.real(*geometry.geometry_tolerance_authority);
    out.u64(geometry.generation); out.string(geometry.last_interval);
    write_field(out, state); write_field(out, geometry.measures);
    write_faces(out, geometry.coordinates, state, 1);
    write_faces(out, geometry.swept_volumes, state, 1); out.u64(1);
    const auto& receipt = *geometry.last_receipt;
    write_point(out, receipt.point); out.string(receipt.physical_frame);
    out.string(receipt.quadrature_identity); out.real(receipt.geometry_tolerance);
    if (!historical) out.u64(tag);
    write_field(out, receipt.previous_state); write_field(out, receipt.previous_measures);
    write_field(out, receipt.integrated_source);
    write_faces(out, receipt.previous_coordinates, state, 1);
    write_faces(out, receipt.physical_flux, state, 3);
    write_faces(out, receipt.face_density, state, 3);
    return std::move(out).take();
  }
};
}  // namespace

TEST(ProgramRuntime, IndependentMovingRelativeFusedVectorBitsAndExactReceipt) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  for (const bool cab : {false, true}) {
    IndependentRelativeQuantityFixture fixture(cab);
    ASSERT_NO_THROW(fixture.read(fixture.image(), fixture.state));
    EXPECT_EQ(fixture.image(), fixture.raw_image(false));
    for (int component = 0; component < 3; ++component) {
      const auto actual = pops::runtime::program::moving_relative_face_amount(
          fixture.physical[component], fixture.density[component], fixture.sweep);
      EXPECT_EQ(std::bit_cast<std::uint64_t>(actual), std::bit_cast<std::uint64_t>(fixture.fused[component]));
      EXPECT_NE(std::bit_cast<std::uint64_t>(actual), std::bit_cast<std::uint64_t>(fixture.ordinary[component]));
    }
    fixture.rebuild_ledger(true);
    try {
      (void)fixture.read(fixture.raw_image(false), fixture.state);
      FAIL() << "reader accepted the one-bit ordinary quantity in a fused receipt";
    } catch (const std::invalid_argument& error) {
      EXPECT_NE(std::string(error.what()).find("receipt quantities/support differ"), std::string::npos);
    }
  }
}

TEST(ProgramRuntime, IndependentMovingRelativeUnknownTagAndVersionRefuse) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  IndependentRelativeQuantityFixture fixture;
  const auto authentic = fixture.image();
  ASSERT_NO_THROW(fixture.read(authentic, fixture.state));
  EXPECT_THROW(fixture.read(fixture.raw_image(false, 2), fixture.state), std::invalid_argument);
  EXPECT_THROW(fixture.read(fixture.raw_image(false, UINT64_MAX), fixture.state), std::invalid_argument);
  EXPECT_EQ(fixture.image(), authentic);
  fixture.geometry.checkpoint_wire_version = 3;
  EXPECT_THROW(fixture.image(), std::invalid_argument);
  fixture.geometry.checkpoint_wire_version = 5;
  EXPECT_THROW(fixture.image(), std::invalid_argument);
  EXPECT_THROW(pops::runtime::program::moving_relative_face_amount(1., 2., .1,
      static_cast<IndependentRelativeQuantityFixture::Convention>(2)), std::invalid_argument);
}

TEST(ProgramRuntime, IndependentMovingRelativeHistorical03PreservesExactBytes) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  IndependentRelativeQuantityFixture fixture(false, true);
  const auto old_bytes = fixture.raw_image(true);
  ASSERT_NO_THROW(fixture.read(old_bytes, fixture.state));
  EXPECT_EQ(fixture.image(), old_bytes);
  const auto restored = fixture.read(old_bytes, fixture.state);
  EXPECT_EQ(restored.geometry.at("mesh").checkpoint_wire_version, 3u);
  EXPECT_EQ(restored.geometry.at("mesh").last_receipt->relative_amount_convention,
            IndependentRelativeQuantityFixture::Convention::kLegacyExpression);
  const auto reexport = pops::runtime::program::checkpoint_moving_intervals(
      restored.exchanges, restored.geometry,
      [&](int block) -> const pops::MultiFab<1>& { return restored.states.at(block); });
  EXPECT_EQ(reexport, old_bytes);
  // Initial historical geometry also retains @3 without a receipt/tag.
  auto initial = fixture.declaration;
  initial.checkpoint_wire_version = 3;
  pops::runtime::program::AcceptedExchangeLedger empty;
  const auto initial_bytes = pops::runtime::program::checkpoint_moving_intervals(
      empty, std::map<std::string, IndependentMovingCodecFixture::Geometry>{{"mesh", initial}},
      [&](int) -> const pops::MultiFab<1>& { return fixture.geometry.last_receipt->previous_state; });
  EXPECT_EQ(std::string(initial_bytes.begin(), initial_bytes.begin() + 8), "POPSEX03");
}

TEST(ProgramRuntime, IndependentMovingRelativeMixedHistoryUsesPerReceiptConvention) {
#if defined(POPS_HAS_KOKKOS)
  ensure_kokkos();
#endif
  using namespace pops::runtime::program;
  IndependentRelativeQuantityFixture fused, legacy(false, true);
  ASSERT_NO_THROW(fused.read(fused.image(), fused.state));
  ASSERT_NO_THROW(legacy.read(legacy.image(), legacy.state));
  legacy.geometry.runtime_block = legacy.declaration.runtime_block = 1;
  ExchangeRecord point{"moving-interval", "interval", "legacy", "endpoint-swept@1", 1, 1., 0., 1., 1};
  point.qualify_runtime_point(legacy.geometry.last_receipt->point);
  legacy.geometry.last_interval = point.evaluation_context;
  AcceptedExchangeLedger joined = fused.ledger;
  for (const auto& old : legacy.ledger.records()) {
    auto operation = old.operation_identity;
    const auto position = operation.find("mesh");
    ASSERT_NE(position, std::string::npos);
    operation.replace(position, 4, "legacy");
    ExchangeRecord record{operation, old.occurrence_identity, "legacy", old.quadrature_identity,
                          old.orientation, old.face_measure, old.numerical_flux, old.temporal_weight,
                          old.multiplicity};
    record.qualify_runtime_point(legacy.geometry.last_receipt->point);
    joined.stage(std::move(record));
  }
  const std::map<std::string, IndependentMovingCodecFixture::Geometry> geometries{
      {"legacy", legacy.geometry}, {"mesh", fused.geometry}};
  const std::map<std::string, IndependentMovingCodecFixture::Geometry> declarations{
      {"legacy", legacy.declaration}, {"mesh", fused.declaration}};
  const auto state = [&](int block) -> const pops::MultiFab<1>& {
    return block == 0 ? fused.state : legacy.state;
  };
  std::vector<std::uint8_t> bytes;
  ASSERT_NO_THROW(bytes = checkpoint_moving_intervals(joined, geometries, state));
  EXPECT_EQ(std::string(bytes.begin(), bytes.begin() + 8), "POPSEX04");
  ASSERT_NO_THROW(read_moving_checkpoint(bytes, joined, declarations, state));
  const auto restored = read_moving_checkpoint(bytes, joined, declarations, state);
  EXPECT_EQ(restored.geometry.at("legacy").last_receipt->relative_amount_convention,
            IndependentRelativeQuantityFixture::Convention::kLegacyExpression);
  EXPECT_EQ(restored.geometry.at("mesh").last_receipt->relative_amount_convention,
            IndependentRelativeQuantityFixture::Convention::kFused);
  EXPECT_EQ(checkpoint_moving_intervals(restored.exchanges, restored.geometry,
      [&](int block) -> const pops::MultiFab<1>& { return restored.states.at(block); }), bytes);
}
