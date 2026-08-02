#include <gtest/gtest.h>

#include <pops/runtime/config/platform_manifest.hpp>

#include <string>
#include <vector>

namespace {

using pops::platform::ExecutionContext;
using pops::platform::ExecutionResource;
using pops::platform::FieldViewDescriptor;
using pops::platform::PlatformManifest;
using pops::platform::RuntimeBackendManifest;

PlatformManifest platform() {
  return pops::platform::proven_serial_platform("production", "system", "headers|clang|c++23");
}

RuntimeBackendManifest backend() {
  return pops::platform::proven_serial_backend("production", "system", "headers|clang|c++23");
}

ExecutionContext context() {
  return {backend(), {"serial", 0, false}, {"float64", 0, false}, {"host", 0, false}};
}

FieldViewDescriptor field() {
  return {"state",   2,      {16, 12},  {12, 1}, "cell",    {{0, 0}, {0, 0}},
          "float64", "host", "patch-0", "right", "borrowed"};
}

}  // namespace

TEST(PlatformManifest, EveryCompatibilityFactChangesIdentity) {
  const auto baseline = platform();
  const std::string identity = pops::platform::identity_token("platform-manifest", baseline);
  for (int variant = 0; variant < 6; ++variant) {
    auto changed = baseline;
    if (variant == 0)
      changed.backend = pops::platform::prove_text("aot", "test");
    else if (variant == 1)
      changed.target = pops::platform::prove_text("amr_system", "test");
    else if (variant == 2)
      changed.abi = pops::platform::prove_text("other|clang|c++23", "test");
    else if (variant == 3)
      changed.precision.compute = pops::platform::prove_text("float32", "test");
    else if (variant == 4)
      changed.device = pops::platform::prove_text("cuda:0", "test");
    else
      changed.communicator = pops::platform::prove_text("comm:7", "test");
    EXPECT_NE(pops::platform::identity_token("platform-manifest", changed), identity);
  }
}

TEST(PlatformManifest, UnknownIsMissingProofAndThreeDimensionsRemainRepresentable) {
  auto missing = platform();
  missing.device = pops::platform::CapabilityProof::unknown();
  EXPECT_THROW(pops::platform::validate_launch(missing, context(), {field()}),
               pops::platform::ContractError);

  auto three_d = field();
  three_d.dimension = 3;
  three_d.extents = {8, 8, 8};
  three_d.strides = {64, 8, 1};
  three_d.ghosts = {{0, 0}, {0, 0}, {0, 0}};
  EXPECT_THROW(pops::platform::validate_launch(platform(), context(), {three_d}),
               pops::platform::ContractError);
}

TEST(PlatformManifest, FieldAndCommunicatorMismatchesRefuseBeforeKernel) {
  int launches = 0;
  auto kernel = [&](const auto&, const auto&) { return ++launches; };
  const auto required = field();
  for (int variant = 0; variant < 9; ++variant) {
    auto actual = field();
    if (variant == 0)
      actual.centering = "node";
    else if (variant == 1)
      actual.scalar = "float32";
    else if (variant == 2)
      actual.extents = {15, 12};
    else if (variant == 3)
      actual.memory_space = "device";
    else if (variant == 4)
      actual.strides = {1, 16};
    else if (variant == 5)
      actual.ghosts = {{1, 0}, {0, 0}};
    else if (variant == 6)
      actual.patch = "patch-1";
    else if (variant == 7)
      actual.layout = "left";
    else
      actual.ownership = "owned";
    EXPECT_THROW(
        pops::platform::launch_checked(platform(), context(), {actual}, kernel, {required}),
        pops::platform::ContractError);
    EXPECT_EQ(launches, 0);
  }
  auto execution = context();
  execution.communicator.identity = "comm:wrong";
  EXPECT_THROW(pops::platform::launch_checked(platform(), execution, {field()}, kernel, {required}),
               pops::platform::ContractError);
  EXPECT_EQ(launches, 0);
}

TEST(PlatformManifest, FieldCapabilitiesAndNamesFailClosed) {
  int launches = 0;
  auto kernel = [&](const auto&, const auto&) { return ++launches; };

  auto missing = platform();
  missing.capabilities.erase("ownership");
  EXPECT_THROW(pops::platform::launch_checked(missing, context(), {field()}, kernel),
               pops::platform::ContractError);

  auto unsupported = platform();
  unsupported.capabilities["layouts"] = pops::platform::prove_text_set({"left"}, "test");
  auto unsupported_context = context();
  unsupported_context.backend.capabilities["layouts"] =
      pops::platform::prove_text_set({"left"}, "test");
  EXPECT_THROW(pops::platform::launch_checked(unsupported, unsupported_context, {field()}, kernel),
               pops::platform::ContractError);

  auto disabled = platform();
  disabled.capabilities["generic_field_view"] = pops::platform::prove_bool(false, "test");
  auto disabled_context = context();
  disabled_context.backend.capabilities["generic_field_view"] =
      pops::platform::prove_bool(false, "test");
  EXPECT_THROW(pops::platform::launch_checked(disabled, disabled_context, {field()}, kernel),
               pops::platform::ContractError);

  EXPECT_THROW(pops::platform::launch_checked(platform(), context(), {field(), field()}, kernel),
               pops::platform::ContractError);
  EXPECT_THROW(
      pops::platform::launch_checked(platform(), context(), {field()}, kernel, {field(), field()}),
      pops::platform::ContractError);
  EXPECT_EQ(launches, 0);
}

TEST(PlatformManifest, FieldGhostsMustLeavePositiveInterior) {
  auto hidden = field();
  hidden.ghosts = {{16, 0}, {0, 0}};
  EXPECT_THROW(pops::platform::validate_launch(platform(), context(), {hidden}),
               pops::platform::ContractError);

  hidden = field();
  hidden.ghosts = {{8, 8}, {0, 0}};
  EXPECT_THROW(pops::platform::validate_launch(platform(), context(), {hidden}),
               pops::platform::ContractError);
}

TEST(PlatformManifest, GenericTwoDimensionalDoubleRouteLaunches) {
  int launches = 0;
  EXPECT_EQ(pops::platform::launch_checked(platform(), context(), {field()},
                                           [&](const auto&, const auto& fields) {
                                             ++launches;
                                             return fields.front().extents[0];
                                           },
                                           {field()}),
            16U);
  EXPECT_EQ(launches, 1);
}
