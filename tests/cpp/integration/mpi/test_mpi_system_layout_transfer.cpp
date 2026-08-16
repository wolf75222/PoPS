#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include "native_dso_compiler.hpp"

#include <pops/parallel/comm.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/config/generated_component_abi.hpp>
#include <pops/runtime/dynamic/authenticated_native_file.hpp>
#include <pops/runtime/dynamic/component_loader.hpp>
#include <pops/runtime/system.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

#ifndef POPS_HAS_MPI
#error "test_mpi_system_layout_transfer requires an MPI-enabled PoPS build"
#endif
#include <mpi.h>

namespace pops {

template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_exact_system_block(
    CompiledSystemBlockPreparation<Dim, Model> request) {
  return prepare_generated_system_block(std::move(request));
}

}  // namespace pops

namespace {

constexpr char kComponentId[] = "pops://test/system-layout-transfer-mpi@1.0.0";
constexpr char kSemanticIdentity[] = "system-layout-transfer-mpi-semantic-v1";
constexpr char kManifestIdentity[] = "system-layout-transfer-mpi-manifest-v1";
constexpr char kMappingIdentity[] = "test::mapping::fine-to-coarse";
constexpr char kProviderIdentity[] = "test::provider::conservative-cell-average";
constexpr char kFineLayout[] = "test::layout::fine";
constexpr char kCoarseLayout[] = "test::layout::coarse";
constexpr char kCellAverage[] = "pops://representations/cell-average@1";
constexpr char kBeforeStep[] = "pops://synchronization/before-step@1";

// This fixture is deliberately only a Transfer provider.  The Systems and their transactions stay
// in the linked PoPS runtime; the DSO proves that the prepared consumer loads, authenticates and
// invokes the public native component ABI instead of substituting a host callback in the test.
std::string transfer_component_source(int dimension) {
  std::string source = R"CPP(
#include <pops/runtime/config/generated_component_abi.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

    namespace {
    constexpr char kComponentId[] = "pops://test/system-layout-transfer-mpi@1.0.0";
    constexpr char kSemanticIdentity[] = "system-layout-transfer-mpi-semantic-v1";
    constexpr char kManifestIdentity[] = "system-layout-transfer-mpi-manifest-v1";
    constexpr std::size_t kDimension = __POPS_DIM__;

    int fail(PopsComponentStatusV1* status, int code, const char* reason) {
      if (status != nullptr)
        *status = {sizeof(PopsComponentStatusV1), code, POPS_COMPONENT_ABORT_RUN_V1, reason};
      return code;
    }

    int apply(void*, const PopsTransferRequestV1* request, PopsComponentStatusV1* status) {
      if (request == nullptr || status == nullptr ||
          request->struct_size < sizeof(PopsTransferRequestV1))
        return fail(status, 11, "transfer request is incomplete");
      if (request->dimension != kDimension || request->source.dimension != kDimension ||
          request->destination.dimension != kDimension || request->source.data == nullptr ||
          request->destination.data == nullptr || request->refinement_ratio == nullptr)
        return fail(status, 12, "transfer field views are incomplete");
      if (request->source.scalar_type != POPS_SCALAR_FLOAT64_V1 ||
          request->destination.scalar_type != POPS_SCALAR_FLOAT64_V1 ||
          request->source.component_count != request->destination.component_count ||
          request->operation != POPS_TRANSFER_OPERATION_CONSERVATIVE_CELL_AVERAGE_V1)
        return fail(status, 13, "transfer type or operation is unsupported");

      std::size_t destination_elements = 1;
      std::size_t samples_per_destination = 1;
      for (std::size_t axis = 0; axis < kDimension; ++axis) {
        const std::int32_t ratio = request->refinement_ratio[axis];
        if (ratio <= 0 || request->source.extents[axis] !=
                              request->destination.extents[axis] * static_cast<std::size_t>(ratio))
          return fail(status, 14, "transfer refinement ratio does not match the field extents");
        destination_elements *= request->destination.extents[axis];
        samples_per_destination *= static_cast<std::size_t>(ratio);
      }

      const auto* source = static_cast<const double*>(request->source.data);
      auto* destination = static_cast<double*>(request->destination.data);
      const double scale = 1.0 / static_cast<double>(samples_per_destination);
      for (std::size_t component = 0; component < request->source.component_count; ++component) {
        for (std::size_t linear_destination = 0; linear_destination < destination_elements;
             ++linear_destination) {
          std::array<std::size_t, kDimension> destination_index{};
          std::size_t remaining_destination = linear_destination;
          std::ptrdiff_t destination_offset =
              static_cast<std::ptrdiff_t>(component) * request->destination.component_stride;
          for (std::size_t axis = 0; axis < kDimension; ++axis) {
            destination_index[axis] = remaining_destination % request->destination.extents[axis];
            remaining_destination /= request->destination.extents[axis];
            destination_offset += static_cast<std::ptrdiff_t>(destination_index[axis]) *
                                  request->destination.axis_strides[axis];
          }
          double sum = 0.0;
          for (std::size_t sample = 0; sample < samples_per_destination; ++sample) {
            std::size_t remaining_sample = sample;
            std::ptrdiff_t source_offset =
                static_cast<std::ptrdiff_t>(component) * request->source.component_stride;
            for (std::size_t axis = 0; axis < kDimension; ++axis) {
              const std::size_t ratio = static_cast<std::size_t>(request->refinement_ratio[axis]);
              const std::size_t offset = remaining_sample % ratio;
              remaining_sample /= ratio;
              source_offset +=
                  static_cast<std::ptrdiff_t>(destination_index[axis] * ratio + offset) *
                  request->source.axis_strides[axis];
            }
            sum += source[source_offset];
          }
          destination[destination_offset] = sum * scale;
        }
      }
      *status = {sizeof(PopsComponentStatusV1), 0, POPS_COMPONENT_CONTINUE_V1, nullptr};
      return 0;
    }

    const PopsTransferApiV1 transfer_table{
        {sizeof(PopsTransferApiV1), POPS_COMPONENT_PROTOCOL_ABI_V1,
         POPS_NATIVE_INTERFACE_TRANSFER_V1, 1, nullptr, nullptr},
        &apply};
    const PopsComponentInterfaceEntryV1 interface_entry{POPS_NATIVE_INTERFACE_TRANSFER_V1, 1,
                                                        sizeof(PopsTransferApiV1), &transfer_table};
    const PopsComponentApiV1 component_api{sizeof(PopsComponentApiV1),
                                           POPS_COMPONENT_PROTOCOL_ABI_V1,
                                           POPS_ABI_KEY_LITERAL,
                                           POPS_COMPONENT_CATALOG_SHA256_V1,
                                           kComponentId,
                                           kSemanticIdentity,
                                           kManifestIdentity,
                                           1,
                                           &interface_entry};
    }  // namespace

    extern "C" const PopsComponentApiV1* pops_component_interface_v1() {
      return &component_api;
    }
  )CPP";
  constexpr std::string_view marker = "__POPS_DIM__";
  source.replace(source.find(marker), marker.size(), std::to_string(dimension));
  return source;
}

template <int Dim>
struct PassiveScalar {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Law::n_vars;
  static constexpr int n_providers = 0;
  Law law{};

  [[nodiscard]] static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.mpi-system-layout-transfer.scalar", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(law.velocity()[axis]);
  }
  POPS_HD pops::nd::StateConversion<Primitive> recover(const State& state) const {
    return law.recover(state);
  }
  POPS_HD pops::nd::StateConversion<State> make_conservative(const Primitive& primitive) const {
    return law.make_conservative(primitive);
  }
  POPS_HD pops::nd::StateConversionStatus admissibility(const State& state) const {
    return law.admissibility(state);
  }
  template <int Axis>
  POPS_HD State flux(const State& state) const {
    return law.template flux<Axis>(state);
  }
  template <int Axis>
  POPS_HD pops::Real max_wave_speed(const State& state) const {
    return law.template max_wave_speed<Axis>(state);
  }
  template <int Axis>
  POPS_HD void wave_speeds(const State& state, pops::Real& lower, pops::Real& upper) const {
    law.template wave_speeds<Axis>(state, lower, upper);
  }
  POPS_HD State source(const State&, const pops::ProviderValues<0>&) const { return State{}; }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }

  static pops::VariableSet conservative_vars() {
    return {pops::VariableKind::Conservative, {"u"}, 1, {pops::VariableRole::Scalar}};
  }
  static pops::VariableSet primitive_vars() {
    return {pops::VariableKind::Primitive, {"u"}, 1, {pops::VariableRole::Scalar}};
  }
};

pops::component::ExpectedNativeComponent expected_component(const std::string& library) {
  return {kComponentId,
          kSemanticIdentity,
          kManifestIdentity,
          POPS_COMPONENT_CATALOG_SHA256_V1,
          POPS_ABI_KEY_LITERAL,
          pops::dynlib::AuthenticatedNativeFile(library).binary_identity(),
          {{POPS_NATIVE_INTERFACE_TRANSFER_V1, 1, sizeof(PopsTransferApiV1)}}};
}

pops::SystemLayoutTransferExecution transfer_execution(MPI_Comm communicator) {
  return {1,
          "test::execution::mpi-lane-host",
          POPS_MEMORY_SPACE_HOST_V1,
          "test::backend::mpi-cpu",
          "test::device::cpu:0",
          POPS_SCALAR_FLOAT64_V1,
          POPS_PRECISION_FLOAT64_V1,
          POPS_PRECISION_FLOAT64_V1,
          POPS_PRECISION_FLOAT64_V1,
          POPS_PRECISION_FLOAT64_V1,
          0,
          "test::stream::host-synchronous",
          static_cast<std::int64_t>(MPI_Comm_c2f(communicator)),
          static_cast<std::int64_t>(MPI_Type_c2f(MPI_DOUBLE)),
          "test::mpi-system-layout-transfer-lane",
          "MPI_DOUBLE"};
}

template <int Dim>
pops::SystemLayoutTransferSpec<Dim> transfer_spec() {
  std::array<std::int32_t, Dim> ratio{};
  ratio.fill(2);
  return {kMappingIdentity,
          kProviderIdentity,
          kComponentId,
          kManifestIdentity,
          kFineLayout,
          kCoarseLayout,
          "fine",
          "coarse",
          kCellAverage,
          kCellAverage,
          kBeforeStep,
          ratio,
          POPS_TRANSFER_OPERATION_CONSERVATIVE_CELL_AVERAGE_V1};
}

class ScopedMpiCommunicator {
 public:
  explicit ScopedMpiCommunicator(MPI_Comm source) {
    if (MPI_Comm_dup(source, &communicator_) != MPI_SUCCESS)
      throw std::runtime_error("MPI_Comm_dup failed for the layout-transfer test lane");
    if (MPI_Comm_set_errhandler(communicator_, MPI_ERRORS_RETURN) != MPI_SUCCESS) {
      MPI_Comm_free(&communicator_);
      throw std::runtime_error("MPI_Comm_set_errhandler failed for the layout-transfer test lane");
    }
  }

  ~ScopedMpiCommunicator() {
    if (communicator_ != MPI_COMM_NULL)
      MPI_Comm_free(&communicator_);
  }

  ScopedMpiCommunicator(const ScopedMpiCommunicator&) = delete;
  ScopedMpiCommunicator& operator=(const ScopedMpiCommunicator&) = delete;

  MPI_Comm get() const { return communicator_; }

 private:
  MPI_Comm communicator_ = MPI_COMM_NULL;
};

template <int Dim>
void install_scalar(pops::System<Dim>& system, const char* name) {
  pops::add_compiled_model(system, name, PassiveScalar<Dim>{}, "none", "rusanov", "conservative",
                           "explicit");
}

template <int Dim>
pops::SystemConfig<Dim> uniform_config(int cells_per_axis) {
  pops::SystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = cells_per_axis;
    config.lower[axis] = pops::Real(0);
    config.upper[axis] = pops::Real(1);
    config.periodicity[static_cast<std::size_t>(axis)] = true;
  }
  config.boxes = {pops::Box<Dim>::from_extents(config.shape)};
  return config;
}

template <int Dim>
std::size_t cell_count(int cells_per_axis) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(cells_per_axis);
  return result;
}

template <int Dim>
std::vector<double> conservative_cell_averages(const std::vector<double>& fine,
                                               int fine_cells_per_axis, int coarse_cells_per_axis) {
  if (fine_cells_per_axis != 2 * coarse_cells_per_axis ||
      fine.size() != cell_count<Dim>(fine_cells_per_axis))
    throw std::invalid_argument("layout-transfer fixture requires a factor-two exact-rank remap");
  std::vector<double> coarse(cell_count<Dim>(coarse_cells_per_axis));
  const std::size_t samples_per_coarse = cell_count<Dim>(2);
  for (std::size_t linear_coarse = 0; linear_coarse < coarse.size(); ++linear_coarse) {
    std::array<std::size_t, Dim> coarse_index{};
    std::size_t remaining_coarse = linear_coarse;
    for (int axis = 0; axis < Dim; ++axis) {
      coarse_index[static_cast<std::size_t>(axis)] =
          remaining_coarse % static_cast<std::size_t>(coarse_cells_per_axis);
      remaining_coarse /= static_cast<std::size_t>(coarse_cells_per_axis);
    }
    double sum = 0.0;
    for (std::size_t sample = 0; sample < samples_per_coarse; ++sample) {
      std::size_t remaining_sample = sample;
      std::size_t linear_fine = 0;
      std::size_t stride = 1;
      for (int axis = 0; axis < Dim; ++axis) {
        const std::size_t fine_coordinate =
            2 * coarse_index[static_cast<std::size_t>(axis)] + remaining_sample % 2;
        remaining_sample /= 2;
        linear_fine += fine_coordinate * stride;
        stride *= static_cast<std::size_t>(fine_cells_per_axis);
      }
      sum += fine[linear_fine];
    }
    coarse[linear_coarse] = sum / static_cast<double>(samples_per_coarse);
  }
  return coarse;
}

int run_mpi_system_layout_transfer(int argc, char** argv) {
  constexpr int Dim = pops::kNativeDimension;
  using NativeSystem = pops::System<Dim>;
  using NativeSystemConfig = pops::SystemConfig<Dim>;
  // Uniform System owns one global box on rank zero.  This is therefore a direct proof of the
  // collective contract, exact world authorities, global receipts, zero-owner participation and
  // transaction retry/rollback.  It deliberately does not claim distributed multi-box payload
  // movement; that belongs to the AMR/multi-box transfer coverage.
  pops::comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  const int rank = pops::my_rank();
  const int ranks = pops::n_ranks();
  long local_failures = 0;
  auto check = [&](bool condition, const char* where) {
    if (!condition) {
      std::fprintf(stderr, "[rank %d/%d] FAIL %s\n", rank, ranks, where);
      ++local_failures;
    }
  };
  auto phase = [&](const char* name, auto&& function) {
    long local_phase_failure = 0;
    try {
      std::forward<decltype(function)>(function)();
    } catch (const std::exception& error) {
      std::fprintf(stderr, "[rank %d/%d] %s: %s\n", rank, ranks, name, error.what());
      local_phase_failure = 1;
      ++local_failures;
    } catch (...) {
      std::fprintf(stderr, "[rank %d/%d] %s: unknown exception\n", rank, ranks, name);
      local_phase_failure = 1;
      ++local_failures;
    }
    return pops::all_reduce_sum(local_phase_failure) == 0;
  };

  const std::filesystem::path base =
      std::filesystem::path(POPS_TEST_TMPDIR) / "mpi_system_layout_transfer_component";
  const std::string source = base.string() + ".cpp";
#if defined(__APPLE__)
  const std::string library = base.string() + ".dylib";
#else
  const std::string library = base.string() + ".so";
#endif
  bool preserve_compile_failure = false;
  auto finish = [&]() {
    const long global_failures = pops::all_reduce_sum(local_failures);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      std::error_code ignored;
      if (!preserve_compile_failure) {
        std::filesystem::remove(source, ignored);
        std::filesystem::remove(library, ignored);
        std::filesystem::remove(library + ".log", ignored);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
      std::printf("%s test_mpi_system_layout_transfer np=%d\n",
                  global_failures == 0 ? "OK" : "FAIL", ranks);
    pops::comm_finalize();
    return global_failures == 0 ? 0 : 1;
  };

  check(ranks == 2, "suite requires exactly two real MPI ranks");
  if (pops::all_reduce_sum(ranks == 2 ? 0L : 1L) != 0)
    return finish();

  int package_ok = 0;
  if (rank == 0) {
    std::ofstream output(source);
    output << transfer_component_source(Dim);
    output.close();
    const auto package = pops::test::native_dso::compile_shared(source, library);
    package_ok = package.ok ? 1 : 0;
    if (!package.ok) {
      preserve_compile_failure = true;
      pops::test::native_dso::report_compile_failure("test_mpi_system_layout_transfer", package);
    }
  }
  MPI_Bcast(&preserve_compile_failure, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  MPI_Bcast(&package_ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);  // No rank may dlopen a path before rank zero has published it.
  check(package_ok == 1, "rank-zero native Transfer DSO compilation");
  if (package_ok != 1)
    return finish();

  {
    const ScopedMpiCommunicator transfer_lane(MPI_COMM_WORLD);
    int world_relation = MPI_UNEQUAL;
    check(MPI_Comm_compare(transfer_lane.get(), MPI_COMM_WORLD, &world_relation) == MPI_SUCCESS,
          "layout-transfer lane comparison succeeds");
    check(world_relation == MPI_CONGRUENT,
          "layout-transfer test executes on a distinct world-congruent communicator");

    std::shared_ptr<pops::component::LoadedComponent> component;
    bool healthy = phase("authenticated Transfer DSO load", [&] {
      component = std::make_shared<pops::component::LoadedComponent>(
          pops::component::LoadedComponent::load(library, expected_component(library)));
    });

    std::unique_ptr<NativeSystem> fine;
    std::unique_ptr<NativeSystem> coarse;
    std::shared_ptr<pops::PreparedSystemLayoutTransfer<Dim>> transfer;
    std::vector<double> fine_initial(cell_count<Dim>(4));
    for (std::size_t index = 0; index < fine_initial.size(); ++index)
      fine_initial[index] = static_cast<double>(2 * index + 1);
    const std::vector<double> coarse_initial(cell_count<Dim>(2), -4.0);
    const std::vector<double> coarse_average = conservative_cell_averages<Dim>(fine_initial, 4, 2);
    std::vector<double> fine_retry(fine_initial.size());
    std::vector<double> coarse_retry(coarse_average.size());
    for (std::size_t index = 0; index < fine_initial.size(); ++index)
      fine_retry[index] = 2.0 * fine_initial[index];
    for (std::size_t index = 0; index < coarse_average.size(); ++index)
      coarse_retry[index] = 2.0 * coarse_average[index];

    if (healthy) {
      healthy = phase("two bound native Systems", [&] {
        NativeSystemConfig fine_config = uniform_config<Dim>(4);
        NativeSystemConfig coarse_config = uniform_config<Dim>(2);
        fine = std::make_unique<NativeSystem>(fine_config);
        coarse = std::make_unique<NativeSystem>(coarse_config);
        install_scalar(*fine, "fine");
        install_scalar(*coarse, "coarse");
        fine->set_state("fine", fine_initial);
        coarse->set_state("coarse", coarse_initial);
        fine->mark_bound();
        coarse->mark_bound();
      });
    }

    if (healthy) {
      check(fine->local_boxes("fine").size() == (rank == 0 ? 1u : 0u),
            "fine System has one owner and one empty peer");
      check(coarse->local_boxes("coarse").size() == (rank == 0 ? 1u : 0u),
            "coarse System has one owner and one empty peer");
      healthy = phase("collective prepared Transfer construction", [&] {
        transfer = pops::PreparedSystemLayoutTransfer<Dim>::prepare(
            *fine, *coarse, component, transfer_spec<Dim>(),
            transfer_execution(transfer_lane.get()));
      });
    }

    auto check_receipt = [&](const pops::SystemLayoutTransferReceipt& receipt,
                             std::uint64_t generation, std::uint64_t attempt) {
      check(receipt.applied && receipt.mapping_identity == kMappingIdentity &&
                receipt.provider_identity == kProviderIdentity &&
                receipt.provider_component_identity == kComponentId &&
                receipt.provider_manifest_identity == kManifestIdentity &&
                receipt.source_layout_identity == kFineLayout &&
                receipt.target_layout_identity == kCoarseLayout && receipt.source_block == "fine" &&
                receipt.target_block == "coarse" &&
                receipt.execution_identity == "test::execution::mpi-lane-host" &&
                receipt.operation == POPS_TRANSFER_OPERATION_CONSERVATIVE_CELL_AVERAGE_V1 &&
                receipt.generation == generation && receipt.attempt == attempt &&
                receipt.source_element_count == fine_initial.size() &&
                receipt.destination_element_count == coarse_initial.size(),
            "global Transfer receipt authenticates the collective operation");
    };

    if (healthy) {
      healthy = phase("capture apply commit", [&] {
        fine->begin_step_transaction();
        coarse->begin_step_transaction();
        transfer->begin_transaction(1);
        transfer->capture(1, 1);
        const auto receipt = transfer->apply(1, 1);
        check_receipt(receipt, 1, 1);
        check(coarse->get_state("coarse") == (rank == 0 ? coarse_average : std::vector<double>{}),
              "accepted apply writes only the target owner");
        fine->commit_step_transaction();
        coarse->commit_step_transaction();
        fine->finalize_step_transaction();
        coarse->finalize_step_transaction();
        transfer->finalize_transaction(1);
      });
    }

    if (healthy) {
      healthy = phase("rejected attempt rollback and retry", [&] {
        fine->set_state("fine", fine_retry);
        fine->begin_step_transaction();
        coarse->begin_step_transaction();
        transfer->begin_transaction(2);
        transfer->capture(2, 1);
        check_receipt(transfer->apply(2, 1), 2, 1);
        check(coarse->get_state("coarse") == (rank == 0 ? coarse_retry : std::vector<double>{}),
              "first retry attempt applied its captured source");

        transfer->reject_attempt(2, 1);
        coarse->rollback_step_transaction();
        fine->rollback_step_transaction();
        check(coarse->get_state("coarse") == (rank == 0 ? coarse_average : std::vector<double>{}),
              "rejected attempt restores the accepted target");

        fine->begin_step_transaction();
        coarse->begin_step_transaction();
        transfer->capture(2, 2);
        check_receipt(transfer->apply(2, 2), 2, 2);
        fine->commit_step_transaction();
        coarse->commit_step_transaction();
        fine->finalize_step_transaction();
        coarse->finalize_step_transaction();
        transfer->finalize_transaction(2);
        check(coarse->get_state("coarse") == (rank == 0 ? coarse_retry : std::vector<double>{}),
              "retried attempt commits the new target");
      });
    }

    if (healthy) {
      healthy = phase("whole transaction rollback", [&] {
        std::vector<double> transient(fine_retry.size());
        std::vector<double> transient_average(coarse_retry.size());
        for (std::size_t index = 0; index < fine_retry.size(); ++index)
          transient[index] = fine_retry[index] + 100.0;
        for (std::size_t index = 0; index < coarse_retry.size(); ++index)
          transient_average[index] = coarse_retry[index] + 100.0;

        fine->begin_step_transaction();
        coarse->begin_step_transaction();
        transfer->begin_transaction(3);
        fine->set_state("fine", transient);
        transfer->capture(3, 1);
        check_receipt(transfer->apply(3, 1), 3, 1);
        check(
            coarse->get_state("coarse") == (rank == 0 ? transient_average : std::vector<double>{}),
            "rollback candidate reaches the target before rejection");
        coarse->rollback_step_transaction();
        fine->rollback_step_transaction();
        transfer->rollback_transaction(3);
        check(fine->get_state("fine") == (rank == 0 ? fine_retry : std::vector<double>{}),
              "whole rollback restores the source owner");
        check(coarse->get_state("coarse") == (rank == 0 ? coarse_retry : std::vector<double>{}),
              "whole rollback restores the target owner");
      });
    }

    if (!healthy && rank == 0)
      std::fprintf(stderr, "test_mpi_system_layout_transfer stopped after a collective failure\n");
    transfer.reset();
    coarse.reset();
    fine.reset();
    component.reset();
  }

  return finish();
}

}  // namespace

TEST(test_mpi_system_layout_transfer, Runs) {
  EXPECT_EQ(
      pops::test::RunTestBody(&run_mpi_system_layout_transfer, "test_mpi_system_layout_transfer"),
      0);
}
