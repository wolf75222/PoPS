#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace pops {
class WorldCommunicator;
}

namespace pops::runtime::output {

/// Non-owning, contiguous NumPy-compatible array view used by the native HDF5 adapter.
///
/// ``dtype`` is the exact NumPy ``dtype.str`` spelling.  The binding owns the Python arrays for
/// the complete duration of ``write_collective_hdf5``; this layer never retains the pointers.
struct ArrayView {
  std::string dtype;
  std::vector<std::size_t> shape;
  const void* data = nullptr;
  std::size_t bytes = 0;
};

struct FieldPieceView {
  std::size_t jlo = 0;
  std::size_t ilo = 0;
  std::size_t jhi = 0;
  std::size_t ihi = 0;
  ArrayView values;
};

struct FieldView {
  std::string dataset;
  std::string dtype;
  std::vector<std::size_t> shape;
  std::vector<FieldPieceView> pieces;
};

struct NamedArrayView {
  std::string dataset;
  ArrayView values;
};

struct ParallelHdf5Capability {
  bool available = false;
  std::string hdf5_version;
  std::string reason;
};

/// Compile-time and runtime capability of the direct C HDF5 + MPI_COMM_WORLD route.
[[nodiscard]] ParallelHdf5Capability parallel_hdf5_capability();

/// Turn a rank-local binding/descriptor validation failure into one all-rank exception before any
/// rank is allowed to enter HDF5.  An empty string means that local validation succeeded.
void collective_hdf5_input_consensus(const WorldCommunicator& world,
                                     const std::string& local_error);

/// Write one exact scientific-output artifact collectively on native ``MPI_COMM_WORLD``.
///
/// Every rank must call this function with the same ``path``, ``manifest_json``, root-array schema,
/// and field schema.  Field pieces remain rank-local, but their bounds are exchanged natively and
/// cross-rank overlaps are rejected before entering HDF5.  The implementation performs all metadata
/// and dataset transfers collectively, disables object timestamps so identical inputs on one
/// resolved platform produce byte-identical artifacts, and reports one consensus error on every
/// rank.  It never initializes MPI and never accepts a Python or foreign communicator.
void write_collective_hdf5(const WorldCommunicator& world, const std::string& path,
                           const std::string& manifest_json,
                           const std::vector<NamedArrayView>& root_arrays,
                           const std::vector<FieldView>& fields);

}  // namespace pops::runtime::output
