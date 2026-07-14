#include <gtest/gtest.h>

#include <pops/runtime/dynamic/component_consumers.hpp>
#include <pops/runtime/dynamic/component_loader.hpp>

#include "component_abi_test_helpers.hpp"

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

namespace abi = pops::component::test_support;

enum class FluxTableFixture { Exact, HeaderOnly, ForgedEntrySize };

constexpr const char* kComponentId = "pops://test/final-flux@1.0.0";
constexpr const char* kSemanticIdentity = "semantic-final-flux";
constexpr const char* kManifestIdentity = "manifest-final-flux";

std::string component_source() {
  return R"CPP(
#include <pops/runtime/config/generated_component_abi.hpp>
#include <cstddef>

namespace {
int prepare_count = 0;
int destroy_count = 0;

int evaluate(void*, const PopsNumericalFluxRequestV1* request,
             PopsNumericalFluxResultV1* result) {
  const auto* left = static_cast<const double*>(request->left.data);
  const auto* right = static_cast<const double*>(request->right.data);
  auto* output = static_cast<double*>(result->normal_flux.data);
  const auto points = request->left.extents[0] * request->left.extents[1];
  for (std::size_t point = 0; point < points; ++point) {
    for (std::size_t component = 0; component < request->left.component_count; ++component) {
      const auto index = point * static_cast<std::size_t>(request->left.axis_strides[1]) +
                         component * static_cast<std::size_t>(request->left.component_stride);
      output[index] = 0.25 * left[index] + 0.75 * right[index];
    }
    result->stability_bounds[point] = 3.0;
    result->actions[point] = POPS_COMPONENT_CONTINUE_V1;
  }
  result->status = {sizeof(PopsComponentStatusV1), 0,
                    POPS_COMPONENT_CONTINUE_V1, nullptr};
  return 0;
}

int apply_transfer(void*, const PopsTransferRequestV1*, PopsComponentStatusV1* status) {
  *status = {sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr};
  return 0;
}

int prepare(const PopsComponentPrepareRequestV1*, void** state,
            PopsComponentStatusV1* status) {
  *state = new int(++prepare_count);
  *status = {sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr};
  return 0;
}
void destroy(void* state) {
  ++destroy_count;
  delete static_cast<int*>(state);
}

#if defined(POPS_TEST_HEADER_ONLY_FLUX_TABLE)
const PopsComponentTableHeaderV1 flux{
    sizeof(PopsComponentTableHeaderV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
    POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1, &prepare, &destroy};
#else
const PopsNumericalFluxApiV1 flux{
    {sizeof(PopsNumericalFluxApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
     POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1, &prepare, &destroy},
    &evaluate};
#endif
const PopsTransferApiV1 transfer{
    {sizeof(PopsTransferApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
     POPS_NATIVE_INTERFACE_TRANSFER_V1, 1, &prepare, &destroy},
    &apply_transfer};
const PopsComponentInterfaceEntryV1 interfaces[]{
    {POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1,
#if defined(POPS_TEST_FORGED_FLUX_ENTRY_SIZE)
     sizeof(PopsNumericalFluxApiV1), &flux},
#else
     sizeof(flux), &flux},
#endif
    {POPS_NATIVE_INTERFACE_TRANSFER_V1, 1,
     sizeof(PopsTransferApiV1), &transfer}};
const PopsComponentApiV1 component{
    sizeof(PopsComponentApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
    POPS_COMPONENT_CATALOG_SHA256_V1,
    "pops://test/final-flux@1.0.0", "semantic-final-flux",
    "manifest-final-flux", 2, interfaces};
}  // namespace

extern "C" const PopsComponentApiV1* pops_component_interface_v1() {
  return &component;
}
extern "C" int pops_test_prepare_count() { return prepare_count; }
extern "C" int pops_test_destroy_count() { return destroy_count; }
)CPP";
}

std::filesystem::path compile_component(
    FluxTableFixture fixture = FluxTableFixture::Exact) {
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto base = std::filesystem::path(POPS_TEST_TMPDIR) /
                    ("final_component_abi_" + std::to_string(stamp));
  const auto source = base.string() + ".cpp";
#if defined(__APPLE__)
  const auto library = base.string() + ".dylib";
  const std::string compiler = "/usr/bin/c++";
  const std::string shared = " -dynamiclib";
#else
  const auto library = base.string() + ".so";
  const std::string compiler = POPS_TEST_CXX;
  const std::string shared = " -shared -fPIC";
#endif
  {
    std::ofstream stream(source);
    stream << component_source();
  }
  std::string fixture_flags;
  if (fixture != FluxTableFixture::Exact)
    fixture_flags += " -DPOPS_TEST_HEADER_ONLY_FLUX_TABLE";
  if (fixture == FluxTableFixture::ForgedEntrySize)
    fixture_flags += " -DPOPS_TEST_FORGED_FLUX_ENTRY_SIZE";
  const std::string command = compiler + shared + fixture_flags +
                              " -std=" + POPS_TEST_CXX_STD +
                              " -O2 -I\"" + POPS_TEST_INCLUDE + "\" \"" + source +
                              "\" -o \"" + library + "\"";
  if (std::system(command.c_str()) != 0) {
    std::filesystem::remove(source);
    throw std::runtime_error("failed to compile exact component ABI fixture");
  }
  std::filesystem::remove(source);
  return library;
}

pops::component::ExpectedNativeComponent expected() {
  return {kComponentId, kSemanticIdentity, kManifestIdentity,
          POPS_COMPONENT_CATALOG_SHA256_V1,
          {{POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1,
            sizeof(PopsNumericalFluxApiV1)},
           {POPS_NATIVE_INTERFACE_TRANSFER_V1, 1,
            sizeof(PopsTransferApiV1)}}};
}

TEST(test_amr_native_loader, LoadsAuthenticatesAndExecutesExactFinalTable) {
  const auto library = compile_component();
  {
    auto loaded = pops::component::LoadedComponent::load(library.string(), expected());
    const auto& table = loaded.table<PopsNumericalFluxApiV1>(
        POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1);
    const std::array<double, 4> left{1.0, 3.0, 5.0, 7.0};
    const std::array<double, 4> right{2.0, 4.0, 6.0, 8.0};
    const std::array<double, 4> normals{1.0, 0.0, 1.0, 0.0};
    std::array<double, 4> flux{};
    std::array<double, 2> stability{};
    std::array<PopsComponentActionV1, 2> actions{};
    const auto execution = abi::host_execution_context();
    const PopsNumericalFluxRequestV1 request{
        sizeof(PopsNumericalFluxRequestV1),
        abi::const_field_view(left.data(), 1, 2, 2),
        abi::const_field_view(right.data(), 1, 2, 2),
        abi::const_field_view(normals.data(), 1, 2, 2),
        nullptr, abi::logical_time(), execution};
    PopsNumericalFluxResultV1 result{
        sizeof(PopsNumericalFluxResultV1),
        abi::field_view(flux.data(), 1, 2, 2),
        stability.data(), actions.data(), {}};
    void* state = loaded.prepared_state(
        POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1, execution);
    ASSERT_NE(state, nullptr);
    ASSERT_EQ(pops::component::evaluate_faces(table, state, request, result), 0);
    EXPECT_EQ(flux, (std::array<double, 4>{1.75, 3.75, 5.75, 7.75}));
    EXPECT_EQ(stability, (std::array<double, 2>{3.0, 3.0}));
    auto mismatched_context = execution;
    mismatched_context.execution_identity = "test::other-execution-context";
    EXPECT_THROW((void)loaded.prepared_state(
                     POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1,
                     mismatched_context), std::invalid_argument);
  }
  std::filesystem::remove(library);
}

TEST(test_amr_native_loader, CachesPreparedResourcesPerExactTargetAndPinsExecutionContext) {
  const auto library = compile_component();
  const auto inspection = pops::dynlib::open(library.string());
  ASSERT_TRUE(pops::dynlib::valid(inspection));
  using CounterFn = int (*)();
  const auto prepare_count = reinterpret_cast<CounterFn>(
      pops::dynlib::sym(inspection, "pops_test_prepare_count"));
  const auto destroy_count = reinterpret_cast<CounterFn>(
      pops::dynlib::sym(inspection, "pops_test_destroy_count"));
  ASSERT_NE(prepare_count, nullptr);
  ASSERT_NE(destroy_count, nullptr);
  {
    auto loaded = pops::component::LoadedComponent::load(library.string(), expected());
    const auto execution = abi::host_execution_context();
    auto anonymous_execution = execution;
    anonymous_execution.execution_identity = "";
    EXPECT_THROW((void)loaded.prepared_state(
                     POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1,
                     anonymous_execution, R"({"scheme":"shared"})",
                     R"({"identity":"target-a"})"),
                 std::invalid_argument);
    EXPECT_EQ(prepare_count(), 0);
    auto incomplete_execution = execution;
    incomplete_execution.backend_identity = nullptr;
    EXPECT_THROW((void)loaded.prepared_state(
                     POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1,
                     incomplete_execution, R"({"scheme":"shared"})",
                     R"({"identity":"target-a"})"),
                 std::invalid_argument);
    EXPECT_EQ(prepare_count(), 0);
    void* first = loaded.prepared_state(
        POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1, execution,
        R"({"scheme":"shared"})", R"({"identity":"target-a"})");
    EXPECT_EQ(prepare_count(), 1);
    EXPECT_EQ(loaded.prepared_state(
                  POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1, execution,
                  R"({"scheme":"shared"})", R"({"identity":"target-a"})"),
              first);
    EXPECT_EQ(prepare_count(), 1);
    void* second = loaded.prepared_state(
        POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1, execution,
        R"({"scheme":"shared"})", R"({"identity":"target-b"})");
    EXPECT_NE(second, first);
    EXPECT_EQ(prepare_count(), 2);

    auto mismatched_context = execution;
    mismatched_context.execution_identity = "test::other-execution-context";
    EXPECT_THROW((void)loaded.prepared_state(
                     POPS_NATIVE_INTERFACE_NUMERICAL_FLUX_V1, 1,
                     mismatched_context, R"({"scheme":"shared"})",
                     R"({"identity":"target-c"})"),
                 std::invalid_argument);
    EXPECT_EQ(prepare_count(), 2);
    EXPECT_EQ(destroy_count(), 0);
  }
  EXPECT_EQ(destroy_count(), 2);
  pops::dynlib::close(inspection);
  std::filesystem::remove(library);
}

TEST(test_amr_native_loader, RefusesIdentityInterfaceAndTableSizeMismatches) {
  const auto library = compile_component();
  auto forged = expected();
  forged.semantic_identity = "forged-semantic";
  EXPECT_THROW(
      pops::component::LoadedComponent::load(library.string(), forged),
      std::runtime_error);

  auto undeclared_export = expected();
  undeclared_export.interfaces.pop_back();
  EXPECT_THROW(
      pops::component::LoadedComponent::load(library.string(), undeclared_export),
      std::runtime_error);

  auto duplicate_expectation = expected();
  duplicate_expectation.interfaces.push_back(duplicate_expectation.interfaces.front());
  EXPECT_THROW(
      pops::component::LoadedComponent::load(library.string(), duplicate_expectation),
      std::runtime_error);

  auto missing = expected();
  missing.interfaces = {{POPS_NATIVE_INTERFACE_TRANSFER_V1, 1,
                         sizeof(PopsTransferApiV1)}};
  EXPECT_THROW(
      pops::component::LoadedComponent::load(library.string(), missing),
      std::runtime_error);

  auto truncated = expected();
  truncated.interfaces[0].minimum_table_size = sizeof(PopsNumericalFluxApiV1) + 1;
  EXPECT_THROW(
      pops::component::LoadedComponent::load(library.string(), truncated),
      std::runtime_error);
  std::filesystem::remove(library);
}

TEST(test_amr_native_loader, RefusesHonestlyReportedHeaderOnlyInterfaceTable) {
  const auto library = compile_component(FluxTableFixture::HeaderOnly);
  EXPECT_THROW(
      pops::component::LoadedComponent::load(library.string(), expected()),
      std::runtime_error);
  std::filesystem::remove(library);
}

TEST(test_amr_native_loader, RefusesHeaderOnlyTableWithForgedFullEntrySize) {
  const auto library = compile_component(FluxTableFixture::ForgedEntrySize);
  EXPECT_THROW(
      pops::component::LoadedComponent::load(library.string(), expected()),
      std::runtime_error);
  std::filesystem::remove(library);
}

}  // namespace
