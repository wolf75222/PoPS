#include <gtest/gtest.h>

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/runtime/config/platform_manifest.hpp>

#include <cstddef>
#include <utility>
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
  std::vector<std::size_t> extents(static_cast<std::size_t>(pops::kNativeDimension));
  std::vector<std::ptrdiff_t> strides(static_cast<std::size_t>(pops::kNativeDimension));
  std::vector<std::pair<int, int>> ghosts(static_cast<std::size_t>(pops::kNativeDimension),
                                          {0, 0});
  std::size_t stride = 1;
  for (int axis = pops::kNativeDimension - 1; axis >= 0; --axis) {
    const std::size_t slot = static_cast<std::size_t>(axis);
    extents[slot] = static_cast<std::size_t>(16 - 2 * axis);
    strides[slot] = static_cast<std::ptrdiff_t>(stride);
    stride *= extents[slot];
  }
  return {"state", pops::kNativeDimension, std::move(extents), std::move(strides),
          "cell",  std::move(ghosts),      "float64",        "host",
          "patch-0", "right",             "borrowed"};
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

TEST(PlatformManifest, UnknownIsMissingProofAndForeignRanksRemainRepresentable) {
  auto missing = platform();
  missing.device = pops::platform::CapabilityProof::unknown();
  EXPECT_THROW(pops::platform::validate_launch(missing, context(), {field()}),
               pops::platform::ContractError);

  auto foreign_rank = field();
  foreign_rank.dimension = pops::kNativeDimension == 3 ? 2 : 3;
  foreign_rank.extents.assign(static_cast<std::size_t>(foreign_rank.dimension), 8);
  foreign_rank.strides.resize(static_cast<std::size_t>(foreign_rank.dimension));
  foreign_rank.ghosts.assign(static_cast<std::size_t>(foreign_rank.dimension), {0, 0});
  std::ptrdiff_t stride = 1;
  for (int axis = foreign_rank.dimension - 1; axis >= 0; --axis) {
    foreign_rank.strides[static_cast<std::size_t>(axis)] = stride;
    stride *= foreign_rank.extents[static_cast<std::size_t>(axis)];
  }
  EXPECT_THROW(pops::platform::validate_launch(platform(), context(), {foreign_rank}),
               pops::platform::ContractError);
}

TEST(PlatformManifest, UnknownCapabilityRefusesBeforeKernel) {
  auto missing = platform();
  missing.device = pops::platform::CapabilityProof::unknown();
  int launches = 0;
  EXPECT_THROW(pops::platform::launch_checked(missing, context(), {field()},
                                              [&](const auto&, const auto&) {
                                                ++launches;
                                                return 0;
                                              },
                                              {field()}),
               pops::platform::ContractError);
  EXPECT_EQ(launches, 0);
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
      --actual.extents.front();
    else if (variant == 3)
      actual.memory_space = "device";
    else if (variant == 4)
      ++actual.strides.front();
    else if (variant == 5)
      actual.ghosts.front() = {1, 0};
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

TEST(PlatformManifest, GenericNativeRankDoubleRouteLaunches) {
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
