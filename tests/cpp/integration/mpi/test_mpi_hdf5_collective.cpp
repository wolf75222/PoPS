#include <gtest/gtest.h>

#include <pops/parallel/world_communicator.hpp>
#include <pops/runtime/output/hdf5_collective.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(POPS_HAS_PARALLEL_HDF5)
#include <hdf5.h>
#endif

namespace {

#if defined(POPS_HAS_PARALLEL_HDF5)

class H5Owner {
 public:
  using Closer = herr_t (*)(hid_t);

  H5Owner(hid_t value, Closer closer) : value_(value), closer_(closer) {
    if (value_ < 0)
      throw std::runtime_error("native HDF5 reopen returned an invalid handle");
  }
  H5Owner(const H5Owner&) = delete;
  H5Owner& operator=(const H5Owner&) = delete;
  ~H5Owner() {
    if (value_ >= 0)
      closer_(value_);
  }

  [[nodiscard]] hid_t get() const noexcept { return value_; }

 private:
  hid_t value_ = -1;
  Closer closer_ = nullptr;
};

void require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

[[nodiscard]] std::string native_dtype(char kind, std::size_t bytes) {
  const std::uint16_t probe = 1;
  const char endian = *reinterpret_cast<const std::uint8_t*>(&probe) == 1 ? '<' : '>';
  return std::string(1, endian) + kind + std::to_string(bytes);
}

using SpatialBounds = std::array<std::size_t, pops::kNativeDimension>;
constexpr std::size_t kCellsPerRankOnAxisZero = 2;

[[nodiscard]] SpatialBounds distributed_shape(int ranks) {
  SpatialBounds result{};
  result[0] = kCellsPerRankOnAxisZero * static_cast<std::size_t>(ranks);
  for (int axis = 1; axis < pops::kNativeDimension; ++axis)
    result[static_cast<std::size_t>(axis)] = static_cast<std::size_t>(axis + 2);
  return result;
}

[[nodiscard]] SpatialBounds piece_lower(std::size_t first) {
  SpatialBounds result{};
  result[0] = first;
  return result;
}

[[nodiscard]] SpatialBounds piece_upper(const SpatialBounds& shape, std::size_t first) {
  SpatialBounds result = shape;
  result[0] = first + kCellsPerRankOnAxisZero;
  return result;
}

[[nodiscard]] std::vector<std::size_t> as_shape(const SpatialBounds& value) {
  return std::vector<std::size_t>(value.begin(), value.end());
}

[[nodiscard]] std::vector<std::size_t> piece_shape(const SpatialBounds& lower,
                                                   const SpatialBounds& upper) {
  std::vector<std::size_t> result;
  result.reserve(pops::kNativeDimension);
  for (int axis = 0; axis < pops::kNativeDimension; ++axis) {
    const auto index = static_cast<std::size_t>(axis);
    result.push_back(upper[index] - lower[index]);
  }
  return result;
}

[[nodiscard]] std::size_t cell_count(const std::vector<std::size_t>& shape) {
  std::size_t result = 1;
  for (const auto extent : shape)
    result *= extent;
  return result;
}

[[nodiscard]] std::vector<double> local_field_values(int rank, const SpatialBounds& lower,
                                                     const SpatialBounds& upper) {
  std::vector<double> result(cell_count(piece_shape(lower, upper)), 0.0);
  for (std::size_t index = 0; index < result.size(); ++index)
    result[index] = static_cast<double>(100 * rank + index) + 0.25;
  return result;
}

[[nodiscard]] pops::runtime::output::FieldView distributed_field(
    const std::string& dtype, const SpatialBounds& shape, const SpatialBounds& lower,
    const SpatialBounds& upper, const std::vector<double>& values) {
  return {"fields/0000/values",
          dtype,
          as_shape(shape),
          {{lower,
            upper,
            {dtype, piece_shape(lower, upper), values.data(), values.size() * sizeof(double)}}}};
}

[[nodiscard]] std::string shared_temporary_path(pops::WorldCommunicator& world, const char* label) {
  std::string path;
  if (world.rank() == 0) {
    const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    path = (std::filesystem::temp_directory_path() /
            (std::string("pops-") + label + "-" + std::to_string(nonce) + ".h5"))
               .string();
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
  }
  return world.broadcast_bytes(path, 0);
}

void validate_reopened_file(const std::filesystem::path& path, int ranks,
                            const std::string& manifest) {
  H5Owner file(H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);

  H5Owner attribute(H5Aopen(file.get(), "pops_output_manifest", H5P_DEFAULT), H5Aclose);
  H5Owner attribute_type(H5Aget_type(attribute.get()), H5Tclose);
  char* attribute_value = nullptr;
  require(H5Aread(attribute.get(), attribute_type.get(), &attribute_value) >= 0,
          "native HDF5 manifest reopen failed");
  const std::string reopened_manifest = attribute_value == nullptr ? "" : attribute_value;
  if (attribute_value != nullptr)
    H5free_memory(attribute_value);
  require(reopened_manifest == manifest, "native HDF5 manifest differs after reopen");

  H5Owner geometry(H5Dopen2(file.get(), "geometry/coverage", H5P_DEFAULT), H5Dclose);
  std::array<std::int32_t, 6> geometry_values{};
  require(H5Dread(geometry.get(), H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                  geometry_values.data()) >= 0,
          "native HDF5 geometry reopen failed");
  require(geometry_values == std::array<std::int32_t, 6>{1, 2, 3, 4, 5, 6},
          "native HDF5 geometry values differ after reopen");

  H5Owner field(H5Dopen2(file.get(), "fields/0000/values", H5P_DEFAULT), H5Dclose);
  H5Owner field_space(H5Dget_space(field.get()), H5Sclose);
  std::array<hsize_t, pops::kNativeDimension> dimensions{};
  require(H5Sget_simple_extent_ndims(field_space.get()) == pops::kNativeDimension &&
              H5Sget_simple_extent_dims(field_space.get(), dimensions.data(), nullptr) ==
                  pops::kNativeDimension,
          "native HDF5 field shape reopen failed");
  const SpatialBounds expected_shape = distributed_shape(ranks);
  for (int axis = 0; axis < pops::kNativeDimension; ++axis)
    require(dimensions[static_cast<std::size_t>(axis)] ==
                expected_shape[static_cast<std::size_t>(axis)],
            "native HDF5 field shape differs after reopen");
  const auto total_cells = cell_count(as_shape(expected_shape));
  std::vector<double> values(total_cells, 0.0);
  require(
      H5Dread(field.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0,
      "native HDF5 field reopen failed");
  const auto inner_cells = total_cells / expected_shape[0];
  for (std::size_t index = 0; index < total_cells; ++index) {
    const auto coordinate_zero = index / inner_cells;
    const auto owner = coordinate_zero / kCellsPerRankOnAxisZero;
    const auto local_zero = coordinate_zero % kCellsPerRankOnAxisZero;
    const auto local_index = local_zero * inner_cells + index % inner_cells;
    const double expected = static_cast<double>(100 * owner + local_index) + 0.25;
    require(values[index] == expected, "native HDF5 distributed field differs after reopen");
  }
}

[[nodiscard]] std::vector<std::byte> read_file_bytes(const std::filesystem::path& path) {
  const auto size = std::filesystem::file_size(path);
  if (size > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max()) ||
      size > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max()))
    throw std::runtime_error("native HDF5 determinism proof file is too large to compare");
  std::vector<std::byte> bytes(static_cast<std::size_t>(size));
  std::ifstream input(path, std::ios::binary);
  if (!input || (size != 0 && !input.read(reinterpret_cast<char*>(bytes.data()),
                                          static_cast<std::streamsize>(size))))
    throw std::runtime_error("native HDF5 determinism proof could not read its artifact");
  return bytes;
}

#endif

}  // namespace

TEST(MpiHdf5Collective, WritesDisjointHyperslabsAndReopensNatively) {
#if !defined(POPS_HAS_PARALLEL_HDF5)
  FAIL() << "this target must never be registered without native parallel HDF5";
#else
  auto& world = pops::WorldCommunicator::world();
  const auto communicator = world.communicator();
  const int rank = world.rank();
  const int ranks = world.size();
  ASSERT_GE(rank, 0);
  ASSERT_GE(ranks, 1);

  const std::string path_text = shared_temporary_path(world, "native-parallel-hdf5");

  constexpr std::size_t columns = 3;
  const SpatialBounds shape = distributed_shape(ranks);
  const SpatialBounds lower = piece_lower(kCellsPerRankOnAxisZero * static_cast<std::size_t>(rank));
  const SpatialBounds upper = piece_upper(shape, lower[0]);
  const std::string int32_dtype = native_dtype('i', sizeof(std::int32_t));
  const std::string float64_dtype = native_dtype('f', sizeof(double));
  const std::array<std::int32_t, 6> geometry_values{1, 2, 3, 4, 5, 6};
  const std::vector<double> local_values = local_field_values(rank, lower, upper);

  const std::vector<pops::runtime::output::NamedArrayView> arrays{{
      "geometry/coverage",
      {int32_dtype,
       {2, columns},
       geometry_values.data(),
       geometry_values.size() * sizeof(std::int32_t)},
  }};
  const std::vector<pops::runtime::output::FieldView> fields{
      distributed_field(float64_dtype, shape, lower, upper, local_values)};
  const std::string manifest = R"({"format":"native-test","version":1})";
  pops::runtime::output::write_collective_hdf5(communicator, path_text, manifest, arrays, fields);

  std::string validation_error;
  if (rank == 0) {
    try {
      validate_reopened_file(path_text, ranks, manifest);
      require(std::filesystem::remove(path_text),
              "native HDF5 proof could not remove its verified artifact");
    } catch (const std::exception& error) {
      validation_error = error.what();
    }
  }
  validation_error = world.broadcast_bytes(validation_error, 0);
  world.barrier();
  EXPECT_TRUE(validation_error.empty()) << validation_error;
#endif
}

TEST(MpiHdf5Collective, RejectsOneRankInvalidDescriptorBeforeCreatingFile) {
#if !defined(POPS_HAS_PARALLEL_HDF5)
  FAIL() << "this target must never be registered without native parallel HDF5";
#else
  auto& world = pops::WorldCommunicator::world();
  const auto communicator = world.communicator();
  const int rank = world.rank();
  const int ranks = world.size();
  if (ranks < 2)
    GTEST_SKIP() << "rank-local consensus proof requires at least two MPI ranks";

  const std::string path = shared_temporary_path(world, "native-parallel-hdf5-invalid");
  const SpatialBounds shape = distributed_shape(ranks);
  const SpatialBounds lower = piece_lower(kCellsPerRankOnAxisZero * static_cast<std::size_t>(rank));
  const SpatialBounds valid_upper = piece_upper(shape, lower[0]);
  SpatialBounds local_upper = valid_upper;
  if (rank == 1)
    local_upper[0] = shape[0] + 1;
  const std::string int32_dtype = native_dtype('i', sizeof(std::int32_t));
  const std::string float64_dtype = native_dtype('f', sizeof(double));
  const std::array<std::int32_t, 1> root_value{1};
  const std::vector<double> local_values = local_field_values(rank, lower, valid_upper);
  const std::vector<pops::runtime::output::NamedArrayView> arrays{{
      "geometry/coverage",
      {int32_dtype, {1}, root_value.data(), sizeof(std::int32_t)},
  }};
  const std::vector<pops::runtime::output::FieldView> fields{
      distributed_field(float64_dtype, shape, lower, local_upper, local_values)};

  std::string error;
  try {
    pops::runtime::output::write_collective_hdf5(
        communicator, path, R"({"format":"native-invalid-test","version":1})", arrays, fields);
  } catch (const std::exception& failure) {
    error = failure.what();
  }
  const auto errors = world.allgather_bytes(error);
  ASSERT_EQ(errors.size(), static_cast<std::size_t>(ranks));
  EXPECT_FALSE(errors.front().empty());
  for (const auto& peer : errors)
    EXPECT_EQ(peer, errors.front());
  EXPECT_NE(error.find("rank 1"), std::string::npos);
  EXPECT_NE(error.find("outside its dataset"), std::string::npos);

  std::string filesystem_error;
  if (rank == 0 && std::filesystem::exists(path))
    filesystem_error = "rank-local validation entered HDF5 and created its target";
  filesystem_error = world.broadcast_bytes(filesystem_error, 0);
  world.barrier();
  EXPECT_TRUE(filesystem_error.empty()) << filesystem_error;
#endif
}

TEST(MpiHdf5Collective, RejectsCrossRankOverlappingHyperslabsBeforeCreatingFile) {
#if !defined(POPS_HAS_PARALLEL_HDF5)
  FAIL() << "this target must never be registered without native parallel HDF5";
#else
  auto& world = pops::WorldCommunicator::world();
  const auto communicator = world.communicator();
  const int rank = world.rank();
  const int ranks = world.size();
  if (ranks < 2)
    GTEST_SKIP() << "cross-rank overlap proof requires at least two MPI ranks";

  const std::string path = shared_temporary_path(world, "native-parallel-hdf5-overlap");
  const SpatialBounds shape = distributed_shape(ranks);
  const auto disjoint_first = kCellsPerRankOnAxisZero * static_cast<std::size_t>(rank);
  const SpatialBounds lower = piece_lower(rank == 1 ? std::size_t{1} : disjoint_first);
  const SpatialBounds upper = piece_upper(shape, lower[0]);
  const std::string int32_dtype = native_dtype('i', sizeof(std::int32_t));
  const std::string float64_dtype = native_dtype('f', sizeof(double));
  const std::array<std::int32_t, 1> root_value{1};
  const std::vector<double> local_values = local_field_values(rank, lower, upper);
  const std::vector<pops::runtime::output::NamedArrayView> arrays{{
      "geometry/coverage",
      {int32_dtype, {1}, root_value.data(), sizeof(std::int32_t)},
  }};
  const std::vector<pops::runtime::output::FieldView> fields{
      distributed_field(float64_dtype, shape, lower, upper, local_values)};

  std::string error;
  try {
    pops::runtime::output::write_collective_hdf5(
        communicator, path, R"({"format":"native-overlap-test","version":1})", arrays, fields);
  } catch (const std::exception& failure) {
    error = failure.what();
  }
  const auto errors = world.allgather_bytes(error);
  ASSERT_EQ(errors.size(), static_cast<std::size_t>(ranks));
  EXPECT_FALSE(errors.front().empty());
  for (const auto& peer : errors)
    EXPECT_EQ(peer, errors.front());
  EXPECT_NE(error.find("overlap across MPI ranks 0 and 1"), std::string::npos);
  EXPECT_NE(error.find("fields/0000/values"), std::string::npos);

  std::string filesystem_error;
  if (rank == 0 && std::filesystem::exists(path))
    filesystem_error = "cross-rank overlap validation entered HDF5 and created its target";
  filesystem_error = world.broadcast_bytes(filesystem_error, 0);
  world.barrier();
  EXPECT_TRUE(filesystem_error.empty()) << filesystem_error;
#endif
}

TEST(MpiHdf5Collective, RepeatedIdenticalWritesAreByteIdenticalAcrossTime) {
#if !defined(POPS_HAS_PARALLEL_HDF5)
  FAIL() << "this target must never be registered without native parallel HDF5";
#else
  auto& world = pops::WorldCommunicator::world();
  const auto communicator = world.communicator();
  const int rank = world.rank();
  const int ranks = world.size();
  const std::string first_path = shared_temporary_path(world, "native-parallel-hdf5-exact-a");
  const std::string second_path = shared_temporary_path(world, "native-parallel-hdf5-exact-b");

  constexpr std::size_t columns = 3;
  const SpatialBounds shape = distributed_shape(ranks);
  const SpatialBounds lower = piece_lower(kCellsPerRankOnAxisZero * static_cast<std::size_t>(rank));
  const SpatialBounds upper = piece_upper(shape, lower[0]);
  const std::string int32_dtype = native_dtype('i', sizeof(std::int32_t));
  const std::string float64_dtype = native_dtype('f', sizeof(double));
  const std::array<std::int32_t, 6> geometry_values{1, 2, 3, 4, 5, 6};
  const std::vector<double> local_values = local_field_values(rank, lower, upper);

  const std::vector<pops::runtime::output::NamedArrayView> arrays{{
      "geometry/coverage",
      {int32_dtype,
       {2, columns},
       geometry_values.data(),
       geometry_values.size() * sizeof(std::int32_t)},
  }};
  const std::vector<pops::runtime::output::FieldView> fields{
      distributed_field(float64_dtype, shape, lower, upper, local_values)};
  const std::string manifest = R"({"format":"native-exact-test","version":1})";

  pops::runtime::output::write_collective_hdf5(communicator, first_path, manifest, arrays, fields);
  std::this_thread::sleep_for(std::chrono::milliseconds(1200));
  pops::runtime::output::write_collective_hdf5(communicator, second_path, manifest, arrays, fields);

  std::string validation_error;
  if (rank == 0) {
    try {
      require(read_file_bytes(first_path) == read_file_bytes(second_path),
              "identical native HDF5 writes differ at byte level across time");
      require(std::filesystem::remove(first_path),
              "native HDF5 exact-output proof could not remove its first artifact");
      require(std::filesystem::remove(second_path),
              "native HDF5 exact-output proof could not remove its second artifact");
    } catch (const std::exception& error) {
      validation_error = error.what();
    }
  }
  validation_error = world.broadcast_bytes(validation_error, 0);
  world.barrier();
  EXPECT_TRUE(validation_error.empty()) << validation_error;
#endif
}
