#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include "native_dso_compiler.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/index/box.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>
#include <pops/runtime/config/generated_component_abi.hpp>
#include <pops/runtime/dynamic/component_loader.hpp>
#include <pops/runtime/system.hpp>
#include <pops/runtime/system/exact_field_marshaling.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
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

// The fixture exposes only the public Transfer ABI.  Its rank is request metadata, so the same
// DSO kernel is an axis-loop algorithm for every native Cartesian rank.  The Systems,
// transactional snapshots and remote CopyTransport stay in PoPS.
std::string transfer_component_source() {
  return R"CPP(
#include <pops/runtime/config/generated_component_abi.hpp>

#include <cstddef>
#include <cstdint>

    namespace {
    constexpr char kComponentId[] = "pops://test/system-layout-transfer-mpi@1.0.0";
    constexpr char kSemanticIdentity[] = "system-layout-transfer-mpi-semantic-v1";
    constexpr char kManifestIdentity[] = "system-layout-transfer-mpi-manifest-v1";

    int fail(PopsComponentStatusV1* status, int code, const char* reason) {
      if (status != nullptr)
        *status = {sizeof(PopsComponentStatusV1), code, POPS_COMPONENT_ABORT_RUN_V1, reason};
      return code;
    }

    std::size_t product(const std::size_t* extents, int dimension) {
      std::size_t result = 1;
      for (int axis = 0; axis < dimension; ++axis)
        result *= extents[axis];
      return result;
    }

    std::ptrdiff_t field_offset(const PopsConstFieldViewV1& field, const std::size_t* index,
                                std::size_t component, int dimension) {
      std::ptrdiff_t result = static_cast<std::ptrdiff_t>(component) * field.component_stride;
      for (int axis = 0; axis < dimension; ++axis)
        result += static_cast<std::ptrdiff_t>(index[axis]) * field.axis_strides[axis];
      return result;
    }

    std::ptrdiff_t field_offset(const PopsFieldViewV1& field, const std::size_t* index,
                                std::size_t component, int dimension) {
      std::ptrdiff_t result = static_cast<std::ptrdiff_t>(component) * field.component_stride;
      for (int axis = 0; axis < dimension; ++axis)
        result += static_cast<std::ptrdiff_t>(index[axis]) * field.axis_strides[axis];
      return result;
    }

    void decode(std::size_t linear, const std::size_t* extents, int dimension, std::size_t* index) {
      for (int axis = 0; axis < dimension; ++axis) {
        index[axis] = linear % extents[axis];
        linear /= extents[axis];
      }
    }

    int apply(void*, const PopsTransferRequestV1* request, PopsComponentStatusV1* status) {
      if (request == nullptr || status == nullptr ||
          request->struct_size < sizeof(PopsTransferRequestV1))
        return fail(status, 11, "transfer request is incomplete");
      const int dimension = request->dimension;
      if (dimension < 1 || dimension > 3 || request->source.dimension != dimension ||
          request->destination.dimension != dimension || request->source.data == nullptr ||
          request->destination.data == nullptr || request->refinement_ratio == nullptr)
        return fail(status, 12, "transfer field views are incomplete");
      if (request->source.scalar_type != POPS_SCALAR_FLOAT64_V1 ||
          request->destination.scalar_type != POPS_SCALAR_FLOAT64_V1 ||
          request->source.component_count != request->destination.component_count ||
          request->operation != POPS_TRANSFER_OPERATION_CONSERVATIVE_CELL_AVERAGE_V1)
        return fail(status, 13, "transfer type or operation is unsupported");

      std::size_t ratio[3]{1, 1, 1};
      for (int axis = 0; axis < dimension; ++axis) {
        if (request->refinement_ratio[axis] <= 0 ||
            request->source.extents[axis] !=
                request->destination.extents[axis] *
                    static_cast<std::size_t>(request->refinement_ratio[axis]))
          return fail(status, 14, "transfer refinement ratio does not match the field extents");
        ratio[axis] = static_cast<std::size_t>(request->refinement_ratio[axis]);
      }

      const auto* source = static_cast<const double*>(request->source.data);
      auto* destination = static_cast<double*>(request->destination.data);
      const std::size_t destination_cells = product(request->destination.extents, dimension);
      const std::size_t fine_cells_per_coarse = product(ratio, dimension);
      const double scale = 1.0 / static_cast<double>(fine_cells_per_coarse);
      for (std::size_t component = 0; component < request->source.component_count; ++component) {
        for (std::size_t destination_linear = 0; destination_linear < destination_cells;
             ++destination_linear) {
          std::size_t destination_index[3]{0, 0, 0};
          decode(destination_linear, request->destination.extents, dimension, destination_index);
          double sum = 0.0;
          for (std::size_t fine_linear = 0; fine_linear < fine_cells_per_coarse; ++fine_linear) {
            std::size_t fine_offset[3]{0, 0, 0};
            std::size_t source_index[3]{0, 0, 0};
            decode(fine_linear, ratio, dimension, fine_offset);
            for (int axis = 0; axis < dimension; ++axis)
              source_index[axis] = destination_index[axis] * ratio[axis] + fine_offset[axis];
            sum += source[field_offset(request->source, source_index, component, dimension)];
          }
          destination[field_offset(request->destination, destination_index, component, dimension)] =
              sum * scale;
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
}

template <int Dim>
struct ScalarFieldModel {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;

  static constexpr int dimension = Dim;
  static constexpr int n_vars = Law::n_vars;
  static constexpr int n_providers = 0;
  Law law{};

  [[nodiscard]] static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.mpi.system-layout-transfer.scalar", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(law.velocity()[axis]);
  }
  static pops::VariableSet conservative_vars() {
    return {pops::VariableKind::Conservative, {"u"}, 1, {pops::VariableRole::Scalar}};
  }
  static pops::VariableSet primitive_vars() {
    return {pops::VariableKind::Primitive, {"u"}, 1, {pops::VariableRole::Scalar}};
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
  POPS_HD State source(const State&, const pops::ProviderValues<0>&) const { return {}; }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
};

template <int Dim>
ScalarFieldModel<Dim> scalar_field_model() {
  pops::RealVector<Dim> velocity{};
  for (int axis = 0; axis < Dim; ++axis)
    velocity[axis] = pops::Real(0);
  return {pops::nd::ScalarAdvection<Dim>::prepare(velocity)};
}

pops::component::ExpectedNativeComponent expected_component() {
  return {
      kComponentId,         kSemanticIdentity,
      kManifestIdentity,    POPS_COMPONENT_CATALOG_SHA256_V1,
      POPS_ABI_KEY_LITERAL, {{POPS_NATIVE_INTERFACE_TRANSFER_V1, 1, sizeof(PopsTransferApiV1)}}};
}

template <int Dim>
pops::SystemLayoutTransferSpec<Dim> transfer_spec() {
  pops::SystemLayoutTransferSpec<Dim> spec{kMappingIdentity,
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
                                           {},
                                           POPS_TRANSFER_OPERATION_CONSERVATIVE_CELL_AVERAGE_V1};
  spec.refinement_ratio.fill(2);
  return spec;
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
void install_scalar(pops::System<Dim>& system, const char* name) {
  pops::add_compiled_model<Dim>(system, name, scalar_field_model<Dim>(), "none", "rusanov",
                                "conservative", "explicit");
}

template <int Dim>
std::size_t cell_count(const pops::Extent<Dim>& extent) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(extent[axis]);
  return result;
}

template <int Dim>
pops::SystemConfig<Dim> split_config(int cells_per_axis, bool reverse_boxes) {
  pops::SystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = cells_per_axis;
    config.lower[axis] = pops::Real(0);
    config.upper[axis] = pops::Real(1);
    config.periodicity[axis] = true;
  }
  for (int ordered = 0; ordered < 2; ++ordered) {
    const int half = reverse_boxes ? 1 - ordered : ordered;
    pops::Index<Dim> lower{};
    pops::Index<Dim> upper{};
    for (int axis = 0; axis < Dim; ++axis)
      upper[axis] = static_cast<int>(config.shape[axis]) - 1;
    lower[0] = half * static_cast<int>(config.shape[0] / 2);
    upper[0] = lower[0] + static_cast<int>(config.shape[0] / 2) - 1;
    config.boxes.emplace_back(lower, upper);
  }
  return config;
}

template <int Dim>
std::vector<double> fine_values(double scale) {
  const auto config = split_config<Dim>(4, false);
  std::vector<double> result(cell_count(config.shape));
  for (std::size_t ordinal = 0; ordinal < result.size(); ++ordinal)
    result[ordinal] = scale * static_cast<double>(ordinal + 1);
  return result;
}

template <int Dim>
std::vector<double> coarse_averages(double scale) {
  const auto fine = split_config<Dim>(4, false);
  const auto coarse = split_config<Dim>(2, true);
  const pops::Box<Dim> fine_domain = fine.index_domain();
  const pops::Box<Dim> coarse_domain = coarse.index_domain();
  std::vector<double> result(cell_count(coarse.shape), 0.0);
  pops::runtime::system::marshaling::for_each_host_index(
      coarse_domain, [&](const pops::Index<Dim>& coarse_index, std::size_t coarse_ordinal) {
        double sum = 0.0;
        const std::size_t samples = std::size_t{1} << Dim;
        for (std::size_t sample = 0; sample < samples; ++sample) {
          pops::Index<Dim> fine_index{};
          for (int axis = 0; axis < Dim; ++axis)
            fine_index[axis] = 2 * coarse_index[axis] + static_cast<int>((sample >> axis) & 1u);
          const auto fine_ordinal =
              pops::runtime::system::marshaling::domain_ordinal(fine_domain, fine_index);
          sum += scale * static_cast<double>(fine_ordinal + 1);
        }
        result[coarse_ordinal] = sum / static_cast<double>(samples);
      });
  return result;
}

template <int Dim>
std::vector<double> gather_distributed_state(pops::System<Dim>& system, const char* name,
                                             int cells_per_axis) {
  const auto config = split_config<Dim>(cells_per_axis, cells_per_axis == 2);
  const pops::Box<Dim> domain = config.index_domain();
  const std::vector<double> local = system.get_state(name);
  std::vector<double> global(cell_count(config.shape), 0.0);
  std::size_t local_offset = 0;
  for (const pops::Box<Dim>& local_box : system.local_boxes(name))
    pops::runtime::system::marshaling::for_each_host_index(
        local_box, [&](const pops::Index<Dim>& index, std::size_t) {
          if (local_offset >= local.size())
            throw std::runtime_error("local System state does not match its local patch geometry");
          global[pops::runtime::system::marshaling::domain_ordinal(domain, index)] =
              local[local_offset++];
        });
  if (local_offset != local.size())
    throw std::runtime_error("local System state has trailing values after exact patch marshaling");
  // get_state is local by contract.  This exact-rank reconstruction therefore uses one collective
  // sum over non-overlapping boxes, making the payload observable on every peer without assuming
  // a root owner or a fixed dimensional layout.
  pops::all_reduce_sum_inplace(global.data(), global.size());
  return global;
}

template <int Dim>
pops::Box<Dim> refine_by_two(const pops::Box<Dim>& box) {
  pops::Index<Dim> lower{};
  pops::Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    lower[axis] = 2 * box.lo[axis];
    upper[axis] = 2 * box.hi[axis] + 1;
  }
  return {lower, upper};
}

template <int Dim>
void check_receipt(const pops::SystemLayoutTransferReceipt& receipt, std::uint64_t generation,
                   std::uint64_t attempt, const char* where, const auto& check) {
  const auto fine = split_config<Dim>(4, false);
  const auto coarse = split_config<Dim>(2, true);
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
            receipt.source_element_count == cell_count(fine.shape) &&
            receipt.destination_element_count == cell_count(coarse.shape),
        where);
}

int run_mpi_system_layout_transfer(int argc, char** argv) {
  constexpr int Dim = pops::kNativeDimension;
  pops::comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  const int rank = pops::my_rank();
  const int ranks = pops::n_ranks();
  long local_failures = 0;
  const auto check = [&](bool condition, const char* where) {
    if (!condition) {
      std::fprintf(stderr, "[rank %d/%d] FAIL %s\n", rank, ranks, where);
      ++local_failures;
    }
  };
  const auto phase = [&](const char* name, auto&& function) {
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
  const auto finish = [&]() {
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
      std::printf("%s test_mpi_system_layout_transfer np=%d dim=%d\n",
                  global_failures == 0 ? "OK" : "FAIL", ranks, Dim);
    pops::comm_finalize();
    return global_failures == 0 ? 0 : 1;
  };

  check(ranks == 2, "suite requires exactly two real MPI ranks");
  if (pops::all_reduce_sum(ranks == 2 ? 0L : 1L) != 0)
    return finish();

  int package_ok = 0;
  if (rank == 0) {
    std::ofstream output(source);
    output << transfer_component_source();
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
  MPI_Barrier(MPI_COMM_WORLD);
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
          pops::component::LoadedComponent::load(library, expected_component()));
    });

    std::unique_ptr<pops::System<Dim>> fine;
    std::unique_ptr<pops::System<Dim>> coarse;
    std::shared_ptr<pops::PreparedSystemLayoutTransfer<Dim>> transfer;
    const std::vector<double> fine_initial = fine_values<Dim>(1.0);
    const std::vector<double> coarse_initial(cell_count(split_config<Dim>(2, true).shape), -4.0);
    const std::vector<double> coarse_average = coarse_averages<Dim>(1.0);
    const std::vector<double> fine_retry = fine_values<Dim>(2.0);
    const std::vector<double> coarse_retry = coarse_averages<Dim>(2.0);

    if (healthy) {
      healthy = phase("two split bound native Systems", [&] {
        fine = std::make_unique<pops::System<Dim>>(split_config<Dim>(4, false));
        coarse = std::make_unique<pops::System<Dim>>(split_config<Dim>(2, true));
        fine->install_block_state_route("fine", "test.mpi.system-layout-transfer/fine/state@1");
        coarse->install_block_state_route("coarse",
                                          "test.mpi.system-layout-transfer/coarse/state@1");
        install_scalar(*fine, "fine");
        install_scalar(*coarse, "coarse");
        fine->set_state("fine", fine_initial);
        coarse->set_state("coarse", coarse_initial);
        fine->mark_bound();
        coarse->mark_bound();
      });
    }

    if (healthy) {
      const std::vector<pops::Box<Dim>> fine_local = fine->local_boxes("fine");
      const std::vector<pops::Box<Dim>> coarse_local = coarse->local_boxes("coarse");
      check(fine_local.size() == 1 && coarse_local.size() == 1,
            "round-robin split gives every MPI rank one local source and target patch");
      const bool remote_capture =
          fine_local.size() == 1 && coarse_local.size() == 1 &&
          fine_local.front().intersect(refine_by_two(coarse_local.front())).empty();
      check(
          remote_capture,
          "reversed source/target layouts force each snapshot carrier patch to cross an MPI owner");
      check(pops::all_reduce_sum(remote_capture ? 1L : 0L) == ranks,
            "all ranks participate in a real cross-owner copy transport");
      healthy = phase("collective prepared Transfer construction", [&] {
        transfer = pops::PreparedSystemLayoutTransfer<Dim>::prepare(
            *fine, *coarse, component, transfer_spec<Dim>(),
            transfer_execution(transfer_lane.get()));
      });
    }

    if (healthy) {
      healthy = phase("capture apply commit through remote copy transport", [&] {
        fine->begin_step_transaction();
        coarse->begin_step_transaction();
        transfer->begin_transaction(1);
        transfer->capture(1, 1);
        check_receipt<Dim>(transfer->apply(1, 1), 1, 1,
                           "global Transfer receipt authenticates the collective operation", check);
        check(gather_distributed_state(*coarse, "coarse", 2) == coarse_average,
              "remote snapshot reaches every target owner before conservative averaging");
        fine->commit_step_transaction();
        coarse->commit_step_transaction();
        fine->finalize_step_transaction();
        coarse->finalize_step_transaction();
        transfer->finalize_transaction(1);
      });
    }

    if (healthy) {
      healthy = phase("rejected attempt rollback and remote retry", [&] {
        fine->set_state("fine", fine_retry);
        fine->begin_step_transaction();
        coarse->begin_step_transaction();
        transfer->begin_transaction(2);
        transfer->capture(2, 1);
        check_receipt<Dim>(transfer->apply(2, 1), 2, 1,
                           "first retry receipt authenticates its exact capture", check);
        check(gather_distributed_state(*coarse, "coarse", 2) == coarse_retry,
              "first retry applies remote fine data");

        transfer->reject_attempt(2, 1);
        coarse->rollback_step_transaction();
        fine->rollback_step_transaction();
        check(gather_distributed_state(*coarse, "coarse", 2) == coarse_average,
              "rejected attempt restores the accepted distributed target");

        fine->begin_step_transaction();
        coarse->begin_step_transaction();
        transfer->capture(2, 2);
        check_receipt<Dim>(transfer->apply(2, 2), 2, 2,
                           "retried receipt authenticates its exact capture", check);
        fine->commit_step_transaction();
        coarse->commit_step_transaction();
        fine->finalize_step_transaction();
        coarse->finalize_step_transaction();
        transfer->finalize_transaction(2);
        check(gather_distributed_state(*coarse, "coarse", 2) == coarse_retry,
              "retry commits the remotely copied target");
      });
    }

    if (healthy) {
      healthy = phase("whole transaction rollback", [&] {
        const std::vector<double> transient = fine_values<Dim>(2.0 + 100.0);
        const std::vector<double> transient_average = coarse_averages<Dim>(2.0 + 100.0);
        fine->begin_step_transaction();
        coarse->begin_step_transaction();
        transfer->begin_transaction(3);
        fine->set_state("fine", transient);
        transfer->capture(3, 1);
        check_receipt<Dim>(transfer->apply(3, 1), 3, 1,
                           "rollback receipt authenticates its exact capture", check);
        check(gather_distributed_state(*coarse, "coarse", 2) == transient_average,
              "rollback candidate reaches each remote target owner");
        coarse->rollback_step_transaction();
        fine->rollback_step_transaction();
        transfer->rollback_transaction(3);
        check(gather_distributed_state(*fine, "fine", 4) == fine_retry,
              "whole rollback restores every source owner");
        check(gather_distributed_state(*coarse, "coarse", 2) == coarse_retry,
              "whole rollback restores every target owner");
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
