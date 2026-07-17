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
  for (int variant = 0; variant < 5; ++variant) {
    auto actual = field();
    auto execution = context();
    if (variant == 0)
      actual.centering = "node";
    else if (variant == 1)
      actual.scalar = "float32";
    else if (variant == 2)
      actual.extents = {15, 12};
    else if (variant == 3)
      actual.memory_space = "device";
    else
      execution.communicator.identity = "comm:wrong";
    EXPECT_THROW(
        pops::platform::launch_checked(platform(), execution, {actual}, kernel, {required}),
        pops::platform::ContractError);
    EXPECT_EQ(launches, 0);
  }
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
