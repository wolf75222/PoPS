#include <pops/runtime/output/hdf5_collective.hpp>
#include <pops/parallel/world_communicator.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>

#if defined(POPS_HAS_PARALLEL_HDF5)
#include <hdf5.h>
#include <mpi.h>
#endif

namespace pops::runtime::output {

#if defined(POPS_HAS_PARALLEL_HDF5)
namespace {

// HDF5's global error stack and collective file machinery are not entered concurrently through
// this adapter.  This lock protects only this native HDF5 implementation; MPI concurrency is
// governed process-wide by the MPI_THREAD_MULTIPLE contract in pops/parallel/comm.hpp.
[[nodiscard]] std::mutex& parallel_hdf5_mutex() {
  static std::mutex value;
  return value;
}

[[nodiscard]] std::size_t checked_multiply(std::size_t left, std::size_t right, const char* where) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::overflow_error(std::string(where) + " overflows size_t");
  return left * right;
}

[[nodiscard]] std::size_t checked_add(std::size_t left, std::size_t right, const char* where) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::overflow_error(std::string(where) + " overflows size_t");
  return left + right;
}

class H5Handle {
 public:
  using Closer = herr_t (*)(hid_t);

  H5Handle() = default;
  H5Handle(hid_t value, Closer closer) : value_(value), closer_(closer) {}
  H5Handle(const H5Handle&) = delete;
  H5Handle& operator=(const H5Handle&) = delete;
  H5Handle(H5Handle&& other) noexcept
      : value_(std::exchange(other.value_, -1)), closer_(other.closer_) {}
  H5Handle& operator=(H5Handle&& other) noexcept {
    if (this != &other) {
      reset();
      value_ = std::exchange(other.value_, -1);
      closer_ = other.closer_;
    }
    return *this;
  }
  ~H5Handle() { reset(); }

  [[nodiscard]] hid_t get() const noexcept { return value_; }
  [[nodiscard]] explicit operator bool() const noexcept { return value_ >= 0; }
  [[nodiscard]] hid_t release() noexcept { return std::exchange(value_, -1); }
  void reset() noexcept {
    if (value_ >= 0 && closer_ != nullptr)
      closer_(value_);
    value_ = -1;
  }

 private:
  hid_t value_ = -1;
  Closer closer_ = nullptr;
};

[[nodiscard]] std::size_t checked_product(const std::vector<std::size_t>& shape,
                                          const char* where) {
  if (shape.empty())
    throw std::invalid_argument(std::string(where) + " must have at least one dimension");
  std::size_t result = 1;
  for (const auto extent : shape) {
    if (extent == 0)
      throw std::invalid_argument(std::string(where) + " dimensions must be positive");
    result = checked_multiply(result, extent, where);
  }
  return result;
}

[[nodiscard]] std::string mpi_error_text(int code) {
  char buffer[MPI_MAX_ERROR_STRING] = {};
  int length = 0;
  MPI_Error_string(code, buffer, &length);
  return std::string(buffer, static_cast<std::size_t>(std::max(length, 0)));
}

void require_mpi(int code, const char* operation) {
  if (code != MPI_SUCCESS)
    throw std::runtime_error(std::string(operation) + " failed: " + mpi_error_text(code));
}

// Error agreement itself must not allocate.  Otherwise one rank can fail while constructing the
// error envelope and skip the MPI collective used to tell its peers to stop.  The first failing rank
// is selected deterministically and broadcasts one bounded diagnostic to every peer.
inline constexpr std::size_t kCollectiveErrorCapacity = 2048;

struct LocalFailure {
  int failed = 0;
  std::array<char, kCollectiveErrorCapacity> detail{};
};

struct AgreedFailure {
  int rank = -1;
  std::array<char, kCollectiveErrorCapacity> detail{};

  [[nodiscard]] bool failed() const noexcept { return rank >= 0; }
};

void set_failure(LocalFailure& failure, std::string_view text) noexcept {
  failure.failed = 1;
  const auto count = std::min(text.size(), failure.detail.size() - 1);
  std::copy_n(text.data(), count, failure.detail.data());
  failure.detail[count] = '\0';
}

template <class Operation>
[[nodiscard]] LocalFailure capture_local_failure(Operation&& operation) noexcept {
  LocalFailure failure;
  try {
    operation();
  } catch (const std::exception& error) {
    const char* text = error.what();
    set_failure(failure, text == nullptr ? std::string_view{"native exception without detail"}
                                         : std::string_view{text});
  } catch (...) {
    set_failure(failure, "unknown native exception");
  }
  return failure;
}

[[nodiscard]] AgreedFailure agree_failure(int rank, const LocalFailure& local) {
  int first = local.failed != 0 ? rank : std::numeric_limits<int>::max();
  require_mpi(MPI_Allreduce(MPI_IN_PLACE, &first, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD),
              "MPI_Allreduce(first failing rank)");
  if (first == std::numeric_limits<int>::max())
    return {};
  AgreedFailure result;
  result.rank = first;
  if (rank == first)
    result.detail = local.detail;
  require_mpi(MPI_Bcast(result.detail.data(), static_cast<int>(result.detail.size()), MPI_CHAR,
                        first, MPI_COMM_WORLD),
              "MPI_Bcast(collective failure)");
  return result;
}

template <class Operation>
[[nodiscard]] AgreedFailure collective_phase(int rank, Operation&& operation) {
  return agree_failure(rank, capture_local_failure(std::forward<Operation>(operation)));
}

[[noreturn]] void throw_collective_failure(std::string_view phase, std::string_view subject,
                                           const AgreedFailure& failure) {
  std::string message = "collective HDF5 ";
  message.append(phase);
  if (!subject.empty()) {
    message.push_back(' ');
    message.append(subject);
  }
  message += " failed on rank " + std::to_string(failure.rank) + ": ";
  message.append(failure.detail.data());
  throw std::runtime_error(std::move(message));
}

void require_collective_success(std::string_view phase, std::string_view subject,
                                const AgreedFailure& failure) {
  if (failure.failed())
    throw_collective_failure(phase, subject, failure);
}

[[nodiscard]] AgreedFailure require_identical_text(int rank, std::string_view local) {
  int overflow = local.size() > static_cast<std::size_t>(
                                    std::numeric_limits<unsigned long long>::max());
  require_mpi(MPI_Allreduce(MPI_IN_PLACE, &overflow, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD),
              "MPI_Allreduce(schema length overflow)");
  if (overflow != 0) {
    LocalFailure failure;
    set_failure(failure, "schema length exceeds the native MPI length domain");
    return agree_failure(rank, failure);
  }

  unsigned long long length = rank == 0 ? static_cast<unsigned long long>(local.size()) : 0ULL;
  require_mpi(MPI_Bcast(&length, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD),
              "MPI_Bcast(schema length)");

  std::string reference;
  const auto allocation = capture_local_failure([&] {
    if (rank != 0) {
      if (length > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max()))
        throw std::length_error("schema length cannot be represented by std::size_t");
      reference.resize(static_cast<std::size_t>(length));
    }
  });
  if (auto failure = agree_failure(rank, allocation); failure.failed())
    return failure;

  unsigned long long offset = 0;
  while (offset < length) {
    const int count = static_cast<int>(std::min<unsigned long long>(
        length - offset, static_cast<unsigned long long>(std::numeric_limits<int>::max())));
    char* buffer = rank == 0 ? const_cast<char*>(local.data()) + static_cast<std::size_t>(offset)
                             : reference.data() + static_cast<std::size_t>(offset);
    require_mpi(MPI_Bcast(buffer, count, MPI_CHAR, 0, MPI_COMM_WORLD),
                "MPI_Bcast(schema bytes)");
    offset += static_cast<unsigned long long>(count);
  }

  LocalFailure mismatch;
  const std::string_view expected = rank == 0 ? local : std::string_view{reference};
  if (local != expected)
    set_failure(mismatch, "path, manifest, or dataset schema differs across ranks");
  return agree_failure(rank, mismatch);
}

using PieceDescriptor = std::array<unsigned long long, 5>;

[[nodiscard]] std::vector<PieceDescriptor> piece_descriptors(
    const std::vector<FieldView>& fields) {
  std::size_t count = 0;
  for (const auto& field : fields)
    count = checked_add(count, field.pieces.size(), "native HDF5 piece descriptor count");
  std::vector<PieceDescriptor> result;
  result.reserve(count);
  for (std::size_t field_index = 0; field_index < fields.size(); ++field_index) {
    const auto& field = fields[field_index];
    if constexpr (sizeof(std::size_t) > sizeof(unsigned long long)) {
      const auto maximum = static_cast<std::size_t>(std::numeric_limits<unsigned long long>::max());
      if (field_index > maximum)
        throw std::overflow_error("native HDF5 field index exceeds the MPI descriptor domain");
      for (const auto& piece : field.pieces) {
        if (piece.jlo > maximum || piece.ilo > maximum || piece.jhi > maximum ||
            piece.ihi > maximum)
          throw std::overflow_error("native HDF5 piece bounds exceed the MPI descriptor domain");
      }
    }
    for (const auto& piece : field.pieces) {
      result.push_back({static_cast<unsigned long long>(field_index),
                        static_cast<unsigned long long>(piece.jlo),
                        static_cast<unsigned long long>(piece.ilo),
                        static_cast<unsigned long long>(piece.jhi),
                        static_cast<unsigned long long>(piece.ihi)});
    }
  }
  return result;
}

[[nodiscard]] bool pieces_overlap(const PieceDescriptor& left,
                                  const PieceDescriptor& right) noexcept {
  return left[0] == right[0] && left[1] < right[3] && right[1] < left[3] &&
         left[2] < right[4] && right[2] < left[4];
}

[[nodiscard]] AgreedFailure require_disjoint_rank_pieces(
    int rank, int ranks, const std::vector<PieceDescriptor>& local,
    const std::vector<FieldView>& fields) {
  static_assert(sizeof(PieceDescriptor) == 5 * sizeof(unsigned long long));
  int length_overflow = 0;
  if constexpr (sizeof(std::size_t) > sizeof(unsigned long long)) {
    length_overflow =
        local.size() > static_cast<std::size_t>(std::numeric_limits<unsigned long long>::max());
  }
  require_mpi(MPI_Allreduce(MPI_IN_PLACE, &length_overflow, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD),
              "MPI_Allreduce(piece descriptor length overflow)");
  if (length_overflow != 0) {
    LocalFailure failure;
    set_failure(failure, "piece descriptor count exceeds the native MPI length domain");
    return agree_failure(rank, failure);
  }

  MPI_Datatype descriptor_type = MPI_DATATYPE_NULL;
  const auto type_failure = collective_phase(rank, [&] {
    require_mpi(MPI_Type_contiguous(5, MPI_UNSIGNED_LONG_LONG, &descriptor_type),
                "MPI_Type_contiguous(piece descriptor)");
    require_mpi(MPI_Type_commit(&descriptor_type), "MPI_Type_commit(piece descriptor)");
  });
  auto finish = [&](const AgreedFailure& primary) {
    const auto free_failure = collective_phase(rank, [&] {
      if (descriptor_type != MPI_DATATYPE_NULL)
        require_mpi(MPI_Type_free(&descriptor_type), "MPI_Type_free(piece descriptor)");
    });
    return primary.failed() ? primary : free_failure;
  };
  if (type_failure.failed())
    return finish(type_failure);

  for (int owner = 0; owner < ranks; ++owner) {
    unsigned long long count =
        rank == owner ? static_cast<unsigned long long>(local.size()) : 0ULL;
    require_mpi(MPI_Bcast(&count, 1, MPI_UNSIGNED_LONG_LONG, owner, MPI_COMM_WORLD),
                "MPI_Bcast(piece descriptor count)");

    std::vector<PieceDescriptor> remote;
    const auto allocation = capture_local_failure([&] {
      if (count > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max()))
        throw std::length_error("piece descriptor count cannot be represented by std::size_t");
      if (rank != owner)
        remote.resize(static_cast<std::size_t>(count));
    });
    if (auto failure = agree_failure(rank, allocation); failure.failed())
      return finish(failure);

    auto* buffer = rank == owner ? const_cast<PieceDescriptor*>(local.data()) : remote.data();
    unsigned long long offset = 0;
    while (offset < count) {
      const int chunk = static_cast<int>(std::min<unsigned long long>(
          count - offset, static_cast<unsigned long long>(std::numeric_limits<int>::max())));
      require_mpi(MPI_Bcast(buffer + static_cast<std::size_t>(offset), chunk, descriptor_type, owner,
                            MPI_COMM_WORLD),
                  "MPI_Bcast(piece descriptors)");
      offset += static_cast<unsigned long long>(chunk);
    }

    const auto overlap = capture_local_failure([&] {
      if (rank <= owner)
        return;
      for (const auto& mine : local) {
        for (const auto& theirs : remote) {
          if (!pieces_overlap(mine, theirs))
            continue;
          const auto field_index = static_cast<std::size_t>(mine[0]);
          const std::string_view dataset =
              field_index < fields.size() ? std::string_view{fields[field_index].dataset}
                                          : std::string_view{"<invalid-field-index>"};
          throw std::invalid_argument(
              "field pieces overlap across MPI ranks " + std::to_string(owner) + " and " +
              std::to_string(rank) + " for dataset " + std::string(dataset));
        }
      }
    });
    if (auto failure = agree_failure(rank, overlap); failure.failed())
      return finish(failure);
  }
  return finish({});
}

[[nodiscard]] std::string schema_text(const std::string& path, const std::string& manifest,
                                      const std::vector<NamedArrayView>& arrays,
                                      const std::vector<FieldView>& fields) {
  auto append_text = [](std::string& out, std::string_view value) {
    out += std::to_string(value.size());
    out.push_back(':');
    out.append(value);
    out.push_back(';');
  };
  auto append_shape = [&append_text](std::string& out, const std::vector<std::size_t>& shape) {
    std::string encoded;
    for (const auto extent : shape) {
      encoded += std::to_string(extent);
      encoded.push_back(',');
    }
    append_text(out, encoded);
  };
  std::string result;
  append_text(result, path);
  append_text(result, manifest);
  append_text(result, std::to_string(arrays.size()));
  for (const auto& array : arrays) {
    append_text(result, array.dataset);
    append_text(result, array.values.dtype);
    append_shape(result, array.values.shape);
  }
  append_text(result, std::to_string(fields.size()));
  for (const auto& field : fields) {
    append_text(result, field.dataset);
    append_text(result, field.dtype);
    append_shape(result, field.shape);
  }
  return result;
}

struct ParsedDtype {
  char endian = '=';
  char kind = '\0';
  std::size_t bytes = 0;
};

[[nodiscard]] ParsedDtype parse_dtype(const std::string& value) {
  if (value.size() < 2)
    throw std::invalid_argument("native HDF5 requires a canonical NumPy dtype.str");
  std::size_t cursor = 0;
  char endian = '=';
  if (value[cursor] == '<' || value[cursor] == '>' || value[cursor] == '=' || value[cursor] == '|')
    endian = value[cursor++];
  if (cursor >= value.size())
    throw std::invalid_argument("native HDF5 dtype is missing its scalar kind");
  const char kind = value[cursor++];
  if (cursor >= value.size() || !std::isdigit(static_cast<unsigned char>(value[cursor])))
    throw std::invalid_argument("native HDF5 dtype is missing its scalar byte width");
  std::size_t bytes = 0;
  for (; cursor < value.size(); ++cursor) {
    const char digit = value[cursor];
    if (!std::isdigit(static_cast<unsigned char>(digit)))
      throw std::invalid_argument("native HDF5 dtype has a noncanonical suffix");
    bytes = checked_multiply(bytes, 10, "native HDF5 dtype byte width");
    bytes =
        checked_add(bytes, static_cast<std::size_t>(digit - '0'), "native HDF5 dtype byte width");
  }
  if (bytes == 0)
    throw std::invalid_argument("native HDF5 dtype byte width must be positive");
  return {endian, kind, bytes};
}

[[nodiscard]] hid_t integer_type(char kind, std::size_t bytes, bool little) {
  if (kind != 'i' && kind != 'u')
    return -1;
  if (bytes == 1)
    return H5Tcopy(kind == 'i' ? H5T_STD_I8LE : H5T_STD_U8LE);
  if (bytes == 2)
    return H5Tcopy(kind == 'i' ? (little ? H5T_STD_I16LE : H5T_STD_I16BE)
                               : (little ? H5T_STD_U16LE : H5T_STD_U16BE));
  if (bytes == 4)
    return H5Tcopy(kind == 'i' ? (little ? H5T_STD_I32LE : H5T_STD_I32BE)
                               : (little ? H5T_STD_U32LE : H5T_STD_U32BE));
  if (bytes == 8)
    return H5Tcopy(kind == 'i' ? (little ? H5T_STD_I64LE : H5T_STD_I64BE)
                               : (little ? H5T_STD_U64LE : H5T_STD_U64BE));
  return -1;
}

[[nodiscard]] hid_t float_type(std::size_t bytes, bool little) {
  if (bytes == 4)
    return H5Tcopy(little ? H5T_IEEE_F32LE : H5T_IEEE_F32BE);
  if (bytes == 8)
    return H5Tcopy(little ? H5T_IEEE_F64LE : H5T_IEEE_F64BE);
  return -1;
}

[[nodiscard]] H5Handle hdf5_type(const std::string& value) {
  const auto parsed = parse_dtype(value);
  const std::uint16_t endian_probe = 1;
  const bool native_little = *reinterpret_cast<const std::uint8_t*>(&endian_probe) == 1;
  const bool little =
      parsed.endian == '<' || parsed.endian == '|' || (parsed.endian == '=' && native_little);
  if (parsed.kind == 'b' && parsed.bytes == 1) {
    H5Handle base(H5Tcopy(H5T_STD_U8LE), H5Tclose);
    if (!base)
      throw std::runtime_error("H5Tcopy(bool base) failed");
    H5Handle enumeration(H5Tenum_create(base.get()), H5Tclose);
    if (!enumeration)
      throw std::runtime_error("H5Tenum_create(bool) failed");
    const std::uint8_t false_value = 0;
    const std::uint8_t true_value = 1;
    if (H5Tenum_insert(enumeration.get(), "FALSE", &false_value) < 0 ||
        H5Tenum_insert(enumeration.get(), "TRUE", &true_value) < 0)
      throw std::runtime_error("H5Tenum_insert(bool) failed");
    return enumeration;
  }
  hid_t scalar = -1;
  if (parsed.kind == 'i' || parsed.kind == 'u')
    scalar = integer_type(parsed.kind, parsed.bytes, little);
  else if (parsed.kind == 'f')
    scalar = float_type(parsed.bytes, little);
  if (scalar >= 0)
    return H5Handle(scalar, H5Tclose);
  if (parsed.kind == 'c' && (parsed.bytes == 8 || parsed.bytes == 16)) {
    const auto component_bytes = parsed.bytes / 2;
    H5Handle component(float_type(component_bytes, little), H5Tclose);
    H5Handle complex(H5Tcreate(H5T_COMPOUND, parsed.bytes), H5Tclose);
    if (!component || !complex || H5Tinsert(complex.get(), "r", 0, component.get()) < 0 ||
        H5Tinsert(complex.get(), "i", component_bytes, component.get()) < 0)
      throw std::runtime_error("HDF5 complex dtype creation failed");
    return complex;
  }
  throw std::invalid_argument(
      "native collective HDF5 supports bool, 8/16/32/64-bit integers, float32/64, "
      "and complex64/128; got " +
      value);
}

void validate_array(const ArrayView& array, const char* where) {
  const auto dtype = parse_dtype(array.dtype);
  const auto elements = checked_product(array.shape, where);
  if (elements > std::numeric_limits<std::size_t>::max() / dtype.bytes ||
      elements * dtype.bytes != array.bytes || array.data == nullptr)
    throw std::invalid_argument(std::string(where) + " buffer size differs from shape/dtype");
  (void)hdf5_type(array.dtype);
}

void validate_inputs(const std::string& path, const std::string& manifest,
                     const std::vector<NamedArrayView>& arrays,
                     const std::vector<FieldView>& fields) {
  if (path.empty() || manifest.empty())
    throw std::invalid_argument("native collective HDF5 requires path and manifest bytes");
  std::vector<std::string> names;
  names.reserve(arrays.size() + fields.size());
  for (const auto& array : arrays) {
    if (array.dataset.empty())
      throw std::invalid_argument("native HDF5 root array has an empty dataset path");
    validate_array(array.values, "native HDF5 root array");
    names.push_back(array.dataset);
  }
  for (const auto& field : fields) {
    if (field.dataset.empty() || (field.shape.size() != 2 && field.shape.size() != 3))
      throw std::invalid_argument("native HDF5 field schema is malformed");
    const auto parsed = parse_dtype(field.dtype);
    (void)hdf5_type(field.dtype);
    (void)checked_product(field.shape, "native HDF5 field shape");
    names.push_back(field.dataset);
    const auto components = field.shape.size() == 3 ? field.shape[0] : 1;
    const auto ny = field.shape[field.shape.size() - 2];
    const auto nx = field.shape[field.shape.size() - 1];
    struct PieceBox {
      std::size_t jlo = 0;
      std::size_t ilo = 0;
      std::size_t jhi = 0;
      std::size_t ihi = 0;
    };
    std::vector<PieceBox> boxes;
    boxes.reserve(field.pieces.size());
    for (const auto& piece : field.pieces) {
      if (piece.jlo >= piece.jhi || piece.ilo >= piece.ihi || piece.jhi > ny || piece.ihi > nx)
        throw std::invalid_argument("native HDF5 field piece lies outside its dataset");
      validate_array(piece.values, "native HDF5 field piece");
      const std::vector<std::size_t> expected =
          field.shape.size() == 3
              ? std::vector<std::size_t>{components, piece.jhi - piece.jlo, piece.ihi - piece.ilo}
              : std::vector<std::size_t>{piece.jhi - piece.jlo, piece.ihi - piece.ilo};
      if (piece.values.dtype != field.dtype || piece.values.shape != expected ||
          parse_dtype(piece.values.dtype).bytes != parsed.bytes)
        throw std::invalid_argument("native HDF5 field piece differs from its field schema");
      boxes.push_back({piece.jlo, piece.ilo, piece.jhi, piece.ihi});
    }
    std::sort(boxes.begin(), boxes.end(), [](const auto& left, const auto& right) {
      return std::tie(left.jlo, left.ilo, left.jhi, left.ihi) <
             std::tie(right.jlo, right.ilo, right.jhi, right.ihi);
    });
    for (std::size_t left = 0; left < boxes.size(); ++left) {
      for (std::size_t right = left + 1; right < boxes.size(); ++right) {
        if (boxes[right].jlo >= boxes[left].jhi)
          break;
        if (boxes[left].ilo < boxes[right].ihi && boxes[right].ilo < boxes[left].ihi)
          throw std::invalid_argument("native HDF5 rank-local field pieces overlap");
      }
    }
  }
  std::sort(names.begin(), names.end());
  if (std::adjacent_find(names.begin(), names.end()) != names.end())
    throw std::invalid_argument("native HDF5 dataset paths must be unique");
}

[[nodiscard]] std::vector<hsize_t> hdf5_shape(const std::vector<std::size_t>& shape) {
  std::vector<hsize_t> result;
  result.reserve(shape.size());
  for (const auto extent : shape) {
    if (extent > static_cast<std::size_t>(std::numeric_limits<hsize_t>::max()))
      throw std::overflow_error("native HDF5 extent exceeds hsize_t");
    result.push_back(static_cast<hsize_t>(extent));
  }
  return result;
}

[[nodiscard]] std::vector<std::string> group_paths(
    const std::vector<std::string>& datasets) {
  std::vector<std::string> groups;
  for (const auto& dataset : datasets) {
    std::size_t cursor = 0;
    while ((cursor = dataset.find('/', cursor)) != std::string::npos) {
      if (cursor == 0)
        throw std::invalid_argument("native HDF5 dataset paths must be relative");
      groups.push_back(dataset.substr(0, cursor));
      ++cursor;
    }
  }
  std::sort(groups.begin(), groups.end());
  groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
  return groups;
}

struct DatasetCreatePlan {
  H5Handle space;
  H5Handle type;
  H5Handle creation;
  std::vector<std::byte> zero;
};

[[nodiscard]] DatasetCreatePlan prepare_dataset_creation(
    const std::vector<std::size_t>& shape, const std::string& dtype) {
  const auto dimensions = hdf5_shape(shape);
  DatasetCreatePlan plan;
  plan.space = H5Handle(
      H5Screate_simple(static_cast<int>(dimensions.size()), dimensions.data(), nullptr), H5Sclose);
  plan.type = hdf5_type(dtype);
  plan.creation = H5Handle(H5Pcreate(H5P_DATASET_CREATE), H5Pclose);
  plan.zero.resize(parse_dtype(dtype).bytes);
  if (!plan.space || !plan.type || !plan.creation ||
      H5Pset_obj_track_times(plan.creation.get(), false) < 0 ||
      H5Pset_fill_value(plan.creation.get(), plan.type.get(), plan.zero.data()) < 0 ||
      H5Pset_fill_time(plan.creation.get(), H5D_FILL_TIME_ALLOC) < 0)
    throw std::runtime_error("HDF5 dataset creation-property preparation failed");
  return plan;
}

[[nodiscard]] H5Handle create_dataset(hid_t file, const std::string& name,
                                      const DatasetCreatePlan& plan) {
  H5Handle dataset(H5Dcreate2(file, name.c_str(), plan.type.get(), plan.space.get(), H5P_DEFAULT,
                              plan.creation.get(), H5P_DEFAULT),
                   H5Dclose);
  if (!dataset)
    throw std::runtime_error("H5Dcreate2 failed for " + name);
  return dataset;
}

struct RootArrayWrite {
  H5Handle memory_space;
  H5Handle file_space;
  const void* root_buffer = nullptr;
  std::byte empty = {};

  [[nodiscard]] const void* data() const noexcept {
    return root_buffer == nullptr ? static_cast<const void*>(&empty) : root_buffer;
  }
};

[[nodiscard]] RootArrayWrite prepare_root_array(hid_t dataset, const ArrayView& array, int rank) {
  RootArrayWrite plan;
  plan.file_space = H5Handle(H5Dget_space(dataset), H5Sclose);
  if (rank == 0) {
    const auto dimensions = hdf5_shape(array.shape);
    plan.memory_space =
        H5Handle(H5Screate_simple(static_cast<int>(dimensions.size()), dimensions.data(), nullptr),
                 H5Sclose);
    plan.root_buffer = array.data;
  } else {
    const hsize_t one = 1;
    plan.memory_space = H5Handle(H5Screate_simple(1, &one, nullptr), H5Sclose);
    if (!plan.file_space || !plan.memory_space || H5Sselect_none(plan.file_space.get()) < 0 ||
        H5Sselect_none(plan.memory_space.get()) < 0)
      throw std::runtime_error("HDF5 root-array empty selection failed");
  }
  if (!plan.file_space || !plan.memory_space)
    throw std::runtime_error("HDF5 root-array write-plan preparation failed");
  return plan;
}

struct PackedField {
  H5Handle memory_space;
  H5Handle file_space;
  std::vector<std::byte> bytes;
  std::byte empty = {};

  [[nodiscard]] const void* data() const noexcept {
    return bytes.empty() ? static_cast<const void*>(&empty) : bytes.data();
  }
};

[[nodiscard]] PackedField pack_field(hid_t dataset, const FieldView& field) {
  PackedField packed;
  packed.file_space = H5Handle(H5Dget_space(dataset), H5Sclose);
  if (!packed.file_space || H5Sselect_none(packed.file_space.get()) < 0)
    throw std::runtime_error("HDF5 field selection initialization failed");
  const auto components = field.shape.size() == 3 ? field.shape[0] : 1;
  const auto scalar_bytes = parse_dtype(field.dtype).bytes;
  struct RowSegment {
    std::size_t global_row = 0;
    std::size_t ilo = 0;
    std::size_t ihi = 0;
    const FieldPieceView* piece = nullptr;
    std::size_t local_row = 0;
  };
  std::vector<RowSegment> segments;
  std::size_t selected_cells = 0;
  for (const auto& piece : field.pieces) {
    const auto height = piece.jhi - piece.jlo;
    const auto width = piece.ihi - piece.ilo;
    std::vector<hsize_t> start;
    std::vector<hsize_t> count;
    if (field.shape.size() == 3) {
      start = {0, static_cast<hsize_t>(piece.jlo), static_cast<hsize_t>(piece.ilo)};
      count = {static_cast<hsize_t>(components), static_cast<hsize_t>(height),
               static_cast<hsize_t>(width)};
    } else {
      start = {static_cast<hsize_t>(piece.jlo), static_cast<hsize_t>(piece.ilo)};
      count = {static_cast<hsize_t>(height), static_cast<hsize_t>(width)};
    }
    if (H5Sselect_hyperslab(packed.file_space.get(), H5S_SELECT_OR, start.data(), nullptr,
                            count.data(), nullptr) < 0)
      throw std::runtime_error("HDF5 field hyperslab selection failed");
    for (std::size_t j = 0; j < height; ++j)
      segments.push_back({piece.jlo + j, piece.ilo, piece.ihi, &piece, j});
    selected_cells = checked_add(selected_cells,
                                 checked_multiply(height, width, "native HDF5 selected cell count"),
                                 "native HDF5 selected cell count");
  }
  std::sort(segments.begin(), segments.end(), [](const RowSegment& left, const RowSegment& right) {
    return std::tie(left.global_row, left.ilo, left.ihi) <
           std::tie(right.global_row, right.ilo, right.ihi);
  });
  if (std::adjacent_find(segments.begin(), segments.end(),
                         [](const RowSegment& left, const RowSegment& right) {
                           return left.global_row == right.global_row && left.ihi > right.ilo;
                         }) != segments.end())
    throw std::invalid_argument("native HDF5 rank-local field pieces overlap");
  if (segments.empty()) {
    const hsize_t one = 1;
    packed.memory_space = H5Handle(H5Screate_simple(1, &one, nullptr), H5Sclose);
    if (!packed.memory_space || H5Sselect_none(packed.memory_space.get()) < 0)
      throw std::runtime_error("HDF5 empty field memory selection failed");
    return packed;
  }
  const auto elements =
      checked_multiply(components, selected_cells, "native HDF5 packed element count");
  packed.bytes.resize(checked_multiply(elements, scalar_bytes, "native HDF5 packed byte count"));
  std::size_t destination = 0;
  for (std::size_t component = 0; component < components; ++component) {
    for (const auto& segment : segments) {
      const auto height = segment.piece->jhi - segment.piece->jlo;
      const auto width = segment.piece->ihi - segment.piece->ilo;
      const auto source_row =
          checked_add(checked_multiply(component, height, "native HDF5 piece row offset"),
                      segment.local_row, "native HDF5 piece row offset");
      const auto source_index =
          checked_multiply(source_row, width, "native HDF5 piece scalar offset");
      const auto row_bytes =
          checked_multiply(width, scalar_bytes, "native HDF5 packed row byte count");
      std::memcpy(
          packed.bytes.data() + destination,
          static_cast<const std::byte*>(segment.piece->values.data) + source_index * scalar_bytes,
          row_bytes);
      destination += row_bytes;
    }
  }
  if (destination != packed.bytes.size())
    throw std::logic_error("native HDF5 packed field byte count is inconsistent");
  const hsize_t count = static_cast<hsize_t>(elements);
  packed.memory_space = H5Handle(H5Screate_simple(1, &count, nullptr), H5Sclose);
  if (!packed.memory_space)
    throw std::runtime_error("HDF5 field memory-space creation failed");
  return packed;
}

struct ManifestAttributePlan {
  H5Handle type;
  H5Handle space;
};

[[nodiscard]] ManifestAttributePlan prepare_manifest_attribute() {
  ManifestAttributePlan plan;
  plan.type = H5Handle(H5Tcopy(H5T_C_S1), H5Tclose);
  plan.space = H5Handle(H5Screate(H5S_SCALAR), H5Sclose);
  if (!plan.type || !plan.space || H5Tset_size(plan.type.get(), H5T_VARIABLE) < 0 ||
      H5Tset_cset(plan.type.get(), H5T_CSET_UTF8) < 0 ||
      H5Tset_strpad(plan.type.get(), H5T_STR_NULLTERM) < 0)
    throw std::runtime_error("HDF5 manifest string type creation failed");
  return plan;
}

}  // namespace
#endif

ParallelHdf5Capability parallel_hdf5_capability() {
#if defined(POPS_HAS_PARALLEL_HDF5)
  std::lock_guard<std::mutex> guard{parallel_hdf5_mutex()};
  int initialized = 0;
  require_mpi(MPI_Initialized(&initialized), "MPI_Initialized");
  const std::string version = std::to_string(H5_VERS_MAJOR) + "." + std::to_string(H5_VERS_MINOR) +
                              "." + std::to_string(H5_VERS_RELEASE);
  return {true, version, initialized ? "" : "MPI is compiled but not initialized"};
#else
  return {false, "", "module was not built with MPI and a parallel HDF5 C library"};
#endif
}

void collective_hdf5_input_consensus(const WorldCommunicator& world,
                                     const std::string& local_error) {
#if !defined(POPS_HAS_PARALLEL_HDF5)
  (void)world;
  (void)local_error;
  throw std::runtime_error(
      "collective HDF5 is unavailable: module lacks native parallel HDF5 support");
#else
  world.require_active_mpi_world();
  if (&world != &WorldCommunicator::world())
    throw std::invalid_argument(
        "collective HDF5 requires the exact native MPI_COMM_WORLD resource");
  std::lock_guard<std::mutex> guard{parallel_hdf5_mutex()};
  detail::require_active_mpi_world_unlocked();
  int rank = 0;
  require_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
  LocalFailure local;
  if (!local_error.empty())
    set_failure(local, local_error);
  require_collective_success("binding input validation", "", agree_failure(rank, local));
#endif
}

void write_collective_hdf5(const WorldCommunicator& world, const std::string& path,
                           const std::string& manifest_json,
                           const std::vector<NamedArrayView>& root_arrays,
                           const std::vector<FieldView>& fields) {
#if !defined(POPS_HAS_PARALLEL_HDF5)
  (void)world;
  (void)path;
  (void)manifest_json;
  (void)root_arrays;
  (void)fields;
  throw std::runtime_error(
      "collective HDF5 is unavailable: rebuild with POPS_USE_MPI=ON, POPS_USE_HDF5=ON, "
      "and a parallel HDF5 C library");
#else
  world.require_active_mpi_world();
  if (&world != &WorldCommunicator::world())
    throw std::invalid_argument(
        "collective HDF5 requires the exact native MPI_COMM_WORLD resource");
  std::lock_guard<std::mutex> guard{parallel_hdf5_mutex()};
  detail::require_active_mpi_world_unlocked();
  int rank = 0;
  int ranks = 1;
  require_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
  require_mpi(MPI_Comm_size(MPI_COMM_WORLD, &ranks), "MPI_Comm_size");

  require_collective_success("input validation", "", collective_phase(rank, [&] {
    validate_inputs(path, manifest_json, root_arrays, fields);
  }));

  std::string schema;
  require_collective_success("schema preparation", "", collective_phase(rank, [&] {
    schema = schema_text(path, manifest_json, root_arrays, fields);
  }));
  require_collective_success("schema consensus", "", require_identical_text(rank, schema));

  std::vector<PieceDescriptor> descriptors;
  require_collective_success("piece descriptor preparation", "", collective_phase(rank, [&] {
    descriptors = piece_descriptors(fields);
  }));
  require_collective_success("piece descriptor consensus", "",
                             require_disjoint_rank_pieces(rank, ranks, descriptors, fields));

  std::vector<std::string> dataset_names;
  std::vector<std::string> groups;
  std::vector<DatasetCreatePlan> root_creation_plans;
  std::vector<DatasetCreatePlan> field_creation_plans;
  ManifestAttributePlan manifest_plan;
  H5Handle group_creation;
  H5Handle file_creation;
  H5Handle access;
  H5Handle transfer;
  require_collective_success("local HDF5 preparation", "", collective_phase(rank, [&] {
    dataset_names.reserve(root_arrays.size() + fields.size());
    for (const auto& array : root_arrays)
      dataset_names.push_back(array.dataset);
    for (const auto& field : fields)
      dataset_names.push_back(field.dataset);
    groups = group_paths(dataset_names);

    root_creation_plans.reserve(root_arrays.size());
    for (const auto& array : root_arrays)
      root_creation_plans.push_back(
          prepare_dataset_creation(array.values.shape, array.values.dtype));
    field_creation_plans.reserve(fields.size());
    for (const auto& field : fields)
      field_creation_plans.push_back(prepare_dataset_creation(field.shape, field.dtype));
    manifest_plan = prepare_manifest_attribute();
    group_creation = H5Handle(H5Pcreate(H5P_GROUP_CREATE), H5Pclose);
    if (!group_creation || H5Pset_obj_track_times(group_creation.get(), false) < 0)
      throw std::runtime_error("HDF5 deterministic group creation-property preparation failed");
    file_creation = H5Handle(H5Pcreate(H5P_FILE_CREATE), H5Pclose);
    if (!file_creation || H5Pset_obj_track_times(file_creation.get(), false) < 0)
      throw std::runtime_error("HDF5 deterministic file creation-property preparation failed");

    access = H5Handle(H5Pcreate(H5P_FILE_ACCESS), H5Pclose);
    if (!access || H5Pset_fapl_mpio(access.get(), MPI_COMM_WORLD, MPI_INFO_NULL) < 0)
      throw std::runtime_error("H5Pset_fapl_mpio(MPI_COMM_WORLD) failed");
#if H5_VERSION_GE(1, 10, 0)
    if (H5Pset_all_coll_metadata_ops(access.get(), 1) < 0 ||
        H5Pset_coll_metadata_write(access.get(), 1) < 0)
      throw std::runtime_error("parallel HDF5 collective metadata configuration failed");
#endif
    transfer = H5Handle(H5Pcreate(H5P_DATASET_XFER), H5Pclose);
    if (!transfer || H5Pset_dxpl_mpio(transfer.get(), H5FD_MPIO_COLLECTIVE) < 0)
      throw std::runtime_error("H5Pset_dxpl_mpio(H5FD_MPIO_COLLECTIVE) failed");
  }));

  H5Handle file;
  require_collective_success("file creation", path, collective_phase(rank, [&] {
    file = H5Handle(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, file_creation.get(), access.get()),
                    H5Fclose);
    if (!file)
      throw std::runtime_error("H5Fcreate returned an invalid handle");
  }));

  AgreedFailure transaction_failure;
  auto remember_failure = [&](const AgreedFailure& failure) noexcept {
    if (!transaction_failure.failed() && failure.failed())
      transaction_failure = failure;
  };

  for (const auto& group : groups) {
    const auto failure = collective_phase(rank, [&] {
      const hid_t handle =
          H5Gcreate2(file.get(), group.c_str(), H5P_DEFAULT, group_creation.get(), H5P_DEFAULT);
      if (handle < 0)
        throw std::runtime_error("H5Gcreate2 returned an invalid handle");
      if (H5Gclose(handle) < 0)
        throw std::runtime_error("H5Gclose failed");
    });
    remember_failure(failure);
    if (transaction_failure.failed())
      break;
  }

  if (!transaction_failure.failed()) {
    H5Handle attribute;
    const auto create_failure = collective_phase(rank, [&] {
      attribute = H5Handle(H5Acreate2(file.get(), "pops_output_manifest", manifest_plan.type.get(),
                                      manifest_plan.space.get(), H5P_DEFAULT, H5P_DEFAULT),
                           H5Aclose);
      if (!attribute)
        throw std::runtime_error("H5Acreate2 returned an invalid handle");
    });
    remember_failure(create_failure);
    if (!transaction_failure.failed()) {
      remember_failure(collective_phase(rank, [&] {
        const char* value = manifest_json.c_str();
        if (H5Awrite(attribute.get(), manifest_plan.type.get(), &value) < 0)
          throw std::runtime_error("H5Awrite failed");
      }));
    }
    if (!create_failure.failed()) {
      const auto close_failure = collective_phase(rank, [&] {
        const hid_t handle = attribute.release();
        if (H5Aclose(handle) < 0)
          throw std::runtime_error("H5Aclose failed");
      });
      remember_failure(close_failure);
    }
  }

  if (!transaction_failure.failed()) {
    for (std::size_t index = 0; index < root_arrays.size(); ++index) {
      const auto& array = root_arrays[index];
      H5Handle dataset;
      RootArrayWrite write_plan;
      const auto create_failure = collective_phase(rank, [&] {
        dataset = create_dataset(file.get(), array.dataset, root_creation_plans[index]);
      });
      remember_failure(create_failure);
      if (!transaction_failure.failed()) {
        remember_failure(collective_phase(rank, [&] {
          write_plan = prepare_root_array(dataset.get(), array.values, rank);
        }));
      }
      if (!transaction_failure.failed()) {
        remember_failure(collective_phase(rank, [&] {
          if (H5Dwrite(dataset.get(), root_creation_plans[index].type.get(),
                       write_plan.memory_space.get(), write_plan.file_space.get(), transfer.get(),
                       write_plan.data()) < 0)
            throw std::runtime_error("H5Dwrite(root array) failed");
        }));
      }
      if (!create_failure.failed()) {
        const auto close_failure = collective_phase(rank, [&] {
          const hid_t handle = dataset.release();
          if (H5Dclose(handle) < 0)
            throw std::runtime_error("H5Dclose(root array) failed");
        });
        remember_failure(close_failure);
      }
      if (transaction_failure.failed())
        break;
    }
  }

  if (!transaction_failure.failed()) {
    for (std::size_t index = 0; index < fields.size(); ++index) {
      const auto& field = fields[index];
      H5Handle dataset;
      PackedField packed;
      const auto create_failure = collective_phase(rank, [&] {
        dataset = create_dataset(file.get(), field.dataset, field_creation_plans[index]);
      });
      remember_failure(create_failure);
      if (!transaction_failure.failed()) {
        remember_failure(
            collective_phase(rank, [&] { packed = pack_field(dataset.get(), field); }));
      }
      if (!transaction_failure.failed()) {
        remember_failure(collective_phase(rank, [&] {
          if (H5Dwrite(dataset.get(), field_creation_plans[index].type.get(),
                       packed.memory_space.get(), packed.file_space.get(), transfer.get(),
                       packed.data()) < 0)
            throw std::runtime_error("H5Dwrite(field) failed");
        }));
      }
      if (!create_failure.failed()) {
        const auto close_failure = collective_phase(rank, [&] {
          const hid_t handle = dataset.release();
          if (H5Dclose(handle) < 0)
            throw std::runtime_error("H5Dclose(field) failed");
        });
        remember_failure(close_failure);
      }
      if (transaction_failure.failed())
        break;
    }
  }

  if (!transaction_failure.failed()) {
    remember_failure(collective_phase(rank, [&] {
      if (H5Fflush(file.get(), H5F_SCOPE_GLOBAL) < 0)
        throw std::runtime_error("H5Fflush failed");
    }));
  }

  const auto close_failure = collective_phase(rank, [&] {
      const hid_t handle = file.release();
      if (H5Fclose(handle) < 0)
        throw std::runtime_error("H5Fclose failed");
  });
  remember_failure(close_failure);
  require_collective_success("transaction", "", transaction_failure);
#endif
}

}  // namespace pops::runtime::output
