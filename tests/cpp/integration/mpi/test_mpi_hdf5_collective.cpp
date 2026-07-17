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
  std::array<hsize_t, 2> dimensions{};
  require(H5Sget_simple_extent_ndims(field_space.get()) == 2 &&
              H5Sget_simple_extent_dims(field_space.get(), dimensions.data(), nullptr) == 2,
          "native HDF5 field shape reopen failed");
  constexpr std::size_t rows_per_rank = 2;
  constexpr std::size_t columns = 3;
  const auto rows = rows_per_rank * static_cast<std::size_t>(ranks);
  require(dimensions == std::array<hsize_t, 2>{static_cast<hsize_t>(rows), columns},
          "native HDF5 field shape differs after reopen");
  std::vector<double> values(rows * columns, 0.0);
  require(
      H5Dread(field.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()) >= 0,
      "native HDF5 field reopen failed");
  for (std::size_t j = 0; j < rows; ++j) {
    const auto owner = j / rows_per_rank;
    const auto local_j = j % rows_per_rank;
    for (std::size_t i = 0; i < columns; ++i) {
      const double expected = static_cast<double>(100 * owner + 10 * local_j + i) + 0.25;
      require(values[j * columns + i] == expected,
              "native HDF5 distributed field differs after reopen");
    }
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
  const int rank = world.rank();
  const int ranks = world.size();
  ASSERT_GE(rank, 0);
  ASSERT_GE(ranks, 1);

  const std::string path_text = shared_temporary_path(world, "native-parallel-hdf5");

  constexpr std::size_t rows_per_rank = 2;
  constexpr std::size_t columns = 3;
  const auto jlo = rows_per_rank * static_cast<std::size_t>(rank);
  const auto jhi = jlo + rows_per_rank;
  const auto rows = rows_per_rank * static_cast<std::size_t>(ranks);
  const std::string int32_dtype = native_dtype('i', sizeof(std::int32_t));
  const std::string float64_dtype = native_dtype('f', sizeof(double));
  const std::array<std::int32_t, 6> geometry_values{1, 2, 3, 4, 5, 6};
  std::vector<double> local_values(rows_per_rank * columns, 0.0);
  for (std::size_t j = 0; j < rows_per_rank; ++j)
    for (std::size_t i = 0; i < columns; ++i)
      local_values[j * columns + i] = static_cast<double>(100 * rank + 10 * j + i) + 0.25;

  const std::vector<pops::runtime::output::NamedArrayView> arrays{{
      "geometry/coverage",
      {int32_dtype,
       {2, columns},
       geometry_values.data(),
       geometry_values.size() * sizeof(std::int32_t)},
  }};
  const std::vector<pops::runtime::output::FieldView> fields{{
      "fields/0000/values",
      float64_dtype,
      {rows, columns},
      {{jlo,
        0,
        jhi,
        columns,
        {float64_dtype,
         {rows_per_rank, columns},
         local_values.data(),
         local_values.size() * sizeof(double)}}},
  }};
  const std::string manifest = R"({"format":"native-test","version":1})";
  pops::runtime::output::write_collective_hdf5(world, path_text, manifest, arrays, fields);

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
  const int rank = world.rank();
  const int ranks = world.size();
  if (ranks < 2)
    GTEST_SKIP() << "rank-local consensus proof requires at least two MPI ranks";

  const std::string path = shared_temporary_path(world, "native-parallel-hdf5-invalid");
  constexpr std::size_t rows_per_rank = 2;
  constexpr std::size_t columns = 3;
  const auto rows = rows_per_rank * static_cast<std::size_t>(ranks);
  const auto jlo = rows_per_rank * static_cast<std::size_t>(rank);
  const auto valid_jhi = jlo + rows_per_rank;
  const auto local_jhi = rank == 1 ? rows + 1 : valid_jhi;
  const std::string int32_dtype = native_dtype('i', sizeof(std::int32_t));
  const std::string float64_dtype = native_dtype('f', sizeof(double));
  const std::array<std::int32_t, 1> root_value{1};
  const std::vector<double> local_values(rows_per_rank * columns, rank + 0.5);
  const std::vector<pops::runtime::output::NamedArrayView> arrays{{
      "geometry/coverage",
      {int32_dtype, {1}, root_value.data(), sizeof(std::int32_t)},
  }};
  const std::vector<pops::runtime::output::FieldView> fields{{
      "fields/0000/values",
      float64_dtype,
      {rows, columns},
      {{jlo,
        0,
        local_jhi,
        columns,
        {float64_dtype,
         {rows_per_rank, columns},
         local_values.data(),
         local_values.size() * sizeof(double)}}},
  }};

  std::string error;
  try {
    pops::runtime::output::write_collective_hdf5(
        world, path, R"({"format":"native-invalid-test","version":1})", arrays, fields);
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
  const int rank = world.rank();
  const int ranks = world.size();
  if (ranks < 2)
    GTEST_SKIP() << "cross-rank overlap proof requires at least two MPI ranks";

  const std::string path = shared_temporary_path(world, "native-parallel-hdf5-overlap");
  constexpr std::size_t rows_per_rank = 2;
  constexpr std::size_t columns = 3;
  const auto rows = rows_per_rank * static_cast<std::size_t>(ranks);
  const auto disjoint_jlo = rows_per_rank * static_cast<std::size_t>(rank);
  const auto jlo = rank == 1 ? std::size_t{1} : disjoint_jlo;
  const auto jhi = jlo + rows_per_rank;
  const std::string int32_dtype = native_dtype('i', sizeof(std::int32_t));
  const std::string float64_dtype = native_dtype('f', sizeof(double));
  const std::array<std::int32_t, 1> root_value{1};
  const std::vector<double> local_values(rows_per_rank * columns, rank + 0.5);
  const std::vector<pops::runtime::output::NamedArrayView> arrays{{
      "geometry/coverage",
      {int32_dtype, {1}, root_value.data(), sizeof(std::int32_t)},
  }};
  const std::vector<pops::runtime::output::FieldView> fields{{
      "fields/0000/values",
      float64_dtype,
      {rows, columns},
      {{jlo,
        0,
        jhi,
        columns,
        {float64_dtype,
         {rows_per_rank, columns},
         local_values.data(),
         local_values.size() * sizeof(double)}}},
  }};

  std::string error;
  try {
    pops::runtime::output::write_collective_hdf5(
        world, path, R"({"format":"native-overlap-test","version":1})", arrays, fields);
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
  const int rank = world.rank();
  const int ranks = world.size();
  const std::string first_path = shared_temporary_path(world, "native-parallel-hdf5-exact-a");
  const std::string second_path = shared_temporary_path(world, "native-parallel-hdf5-exact-b");

  constexpr std::size_t rows_per_rank = 2;
  constexpr std::size_t columns = 3;
  const auto rows = rows_per_rank * static_cast<std::size_t>(ranks);
  const auto jlo = rows_per_rank * static_cast<std::size_t>(rank);
  const auto jhi = jlo + rows_per_rank;
  const std::string int32_dtype = native_dtype('i', sizeof(std::int32_t));
  const std::string float64_dtype = native_dtype('f', sizeof(double));
  const std::array<std::int32_t, 6> geometry_values{1, 2, 3, 4, 5, 6};
  std::vector<double> local_values(rows_per_rank * columns, 0.0);
  for (std::size_t j = 0; j < rows_per_rank; ++j)
    for (std::size_t i = 0; i < columns; ++i)
      local_values[j * columns + i] = static_cast<double>(100 * rank + 10 * j + i) + 0.25;

  const std::vector<pops::runtime::output::NamedArrayView> arrays{{
      "geometry/coverage",
      {int32_dtype,
       {2, columns},
       geometry_values.data(),
       geometry_values.size() * sizeof(std::int32_t)},
  }};
  const std::vector<pops::runtime::output::FieldView> fields{{
      "fields/0000/values",
      float64_dtype,
      {rows, columns},
      {{jlo,
        0,
        jhi,
        columns,
        {float64_dtype,
         {rows_per_rank, columns},
         local_values.data(),
         local_values.size() * sizeof(double)}}},
  }};
  const std::string manifest = R"({"format":"native-exact-test","version":1})";

  pops::runtime::output::write_collective_hdf5(world, first_path, manifest, arrays, fields);
  std::this_thread::sleep_for(std::chrono::milliseconds(1200));
  pops::runtime::output::write_collective_hdf5(world, second_path, manifest, arrays, fields);

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
