#pragma once

#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/collective_exception.hpp>
#include <pops/runtime/program/field_read_capture.hpp>
#include <pops/runtime/program/field_read_closure.hpp>

#include <array>
#include <bit>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::runtime::program {

namespace field_read_detail {
class Budget {
 public:
  explicit Budget(std::uint64_t capacity) : capacity_(capacity) {}
  void charge(std::uint64_t count, std::uint64_t width = 1) {
    if (width && count > (capacity_ - used_) / width)
      throw std::length_error("Field read transport exceeds its workspace byte capacity");
    used_ += count * width;
  }
  std::uint64_t used() const noexcept { return used_; }
  std::uint64_t remaining() const noexcept { return capacity_ - used_; }

 private:
  std::uint64_t capacity_, used_ = 0;
};

template <class Function>
void phase(const ExecutionLane& lane, Function&& function, const char* message) {
  std::exception_ptr error;
  try {
    function();
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane, message);
}

// A source length is communicated exactly using four bounded limbs. No payload
// allocation occurs until every rank has checked the same next allocation.
inline std::vector<std::vector<std::uint64_t>> gather(const ExecutionLane& lane,
                                                      std::span<const std::uint64_t> local,
                                                      Budget& budget) {
  std::vector<std::vector<std::uint64_t>> shards;
  phase(
      lane,
      [&] {
        if (lane.size() < 1)
          throw std::invalid_argument("Field read gather has no live execution ranks");
        budget.charge(static_cast<std::uint64_t>(lane.size()), sizeof(shards.front()));
        shards.resize(static_cast<std::size_t>(lane.size()));
      },
      "Field read gather table preflight failed");
  for (int root = 0; root < lane.size(); ++root) {
    std::uint64_t count = 0;
    for (unsigned shift = 0; shift < 64; shift += 21) {
      const auto part =
          lane.rank() == root ? (std::uint64_t(local.size()) >> shift) & 0x1fffffULL : 0;
      count |= std::uint64_t(all_reduce_max(static_cast<long>(part), lane.communicator())) << shift;
    }
    phase(
        lane,
        [&] {
          budget.charge(count, sizeof(std::uint64_t));
          auto& shard = shards[static_cast<std::size_t>(root)];
          if (count > shard.max_size())
            throw std::length_error("Field read gather length is not representable");
          shard.resize(static_cast<std::size_t>(count));
          if (lane.rank() == root && count)
            std::memcpy(shard.data(), local.data(), shard.size() * sizeof(std::uint64_t));
        },
        "Field read gather payload preflight failed");
    auto& shard = shards[static_cast<std::size_t>(root)];
    broadcast_bytes_inplace(reinterpret_cast<char*>(shard.data()),
                            shard.size() * sizeof(std::uint64_t), root, lane.communicator());
  }
  return shards;
}
}  // namespace field_read_detail

// Pure storage/transport for the actual Kokkos read journal. Authentication of
// the source, schema, producer support and provider identities belongs to the
// invoking ProgramContext; no text passed here issues a live Field capability.
template <int Dim>
struct CapturedFieldReads {
  using source_storage_type = typename MultiFab<Dim>::fab_type::storage_type::const_type;
  MultiFab<Dim> journal;
  const MultiFab<Dim>* source_identity = nullptr;
  std::shared_ptr<const void> source_incarnation;
  std::vector<source_storage_type> source_allocations;
  std::uint64_t retained_bytes = 0;

  explicit CapturedFieldReads(MultiFab<Dim>&& value) : journal(std::move(value)) {}

  static CapturedFieldReads prepare(const MultiFab<Dim>& source, std::uint64_t bindings,
                                    const ExecutionLane& lane, std::uint64_t capacity) {
    std::optional<CapturedFieldReads> result;
    field_read_detail::Budget budget(capacity);
    int components = 0;
    field_read_detail::phase(
        lane,
        [&] {
          components = field_read_capture_components(bindings);
          if (source.rank_space().size() != static_cast<std::size_t>(lane.size()) ||
              source.rank_space().coordinate(static_cast<std::size_t>(lane.rank())) !=
                  source.local_rank())
            throw std::invalid_argument(
                "Field read journal source belongs to another execution lane");
          budget.charge(1, sizeof(CapturedFieldReads));
          budget.charge(MultiFab<Dim>::storage_identity_payload_bytes());
          budget.charge(source.layout().size(),
                        4 * (sizeof(Box<Dim>) + sizeof(Index<Dim>) + sizeof(std::size_t)));
          budget.charge(source.local_size(),
                        sizeof(typename MultiFab<Dim>::fab_type) + sizeof(std::size_t));
          for (std::size_t local = 0; local < source.local_size(); ++local)
            budget.charge(
                field_read_capture_bytes(source.box(local).numPts(), bindings, budget.remaining()));
          budget.charge(source.local_size(), sizeof(source_storage_type));
          // Construct the Field only inside the converged allocation phase. In
          // particular, its incarnation token may allocate even on empty ranks.
          auto& prepared =
              result.emplace(MultiFab<Dim>(source.layout(), source.distribution(),
                                           source.local_rank(), components, Extent<Dim>{}));
          prepared.source_allocations.resize(source.local_size());
          for (std::size_t local = 0; local < source.local_size(); ++local)
            prepared.source_allocations[local] = source.fab(local).storage();
          prepared.source_identity = &source;
          prepared.source_incarnation = source.storage_identity();
          // A valid generated kernel must initialize every valid row, including
          // inactive/masked cells. An omitted evaluation cannot look like zero reads.
          prepared.journal.set_val(Real(-1));
          prepared.retained_bytes = budget.used();
        },
        "Field read journal allocation failed");
    return std::move(*result);
  }

  std::string close(const MultiFab<Dim>& source, std::string_view schema, std::string_view support,
                    std::span<const std::string_view> authorities, const ExecutionLane& lane,
                    std::uint64_t capacity) const {
    static_assert(sizeof(Real) == sizeof(std::uint64_t));
    field_read_detail::Budget budget(capacity);
    std::vector<std::uint64_t> local;
    std::size_t width = 0, cursor = 0;
    field_read_detail::phase(
        lane,
        [&] {
          budget.charge(retained_bytes);
          if (&source != source_identity || source.storage_identity() != source_incarnation ||
              source_allocations.size() != source.local_size() ||
              source.layout() != journal.layout() ||
              source.distribution() != journal.distribution() ||
              source.local_rank() != journal.local_rank() ||
              source.local_size() != journal.local_size() ||
              source.rank_space().size() != static_cast<std::size_t>(lane.size()) ||
              source.rank_space().coordinate(static_cast<std::size_t>(lane.rank())) !=
                  source.local_rank() ||
              journal.ncomp() != field_read_capture_components(authorities.size()))
            throw std::invalid_argument("Field read journal differs from its exact source support");
          for (std::size_t local = 0; local < source.local_size(); ++local)
            if (source_allocations[local].data() != source.fab(local).view().data)
              throw std::invalid_argument("Field read journal source allocation was replaced");
          width = Dim + static_cast<std::size_t>(journal.ncomp());
          std::size_t count = 0;
          for (std::size_t patch = 0; patch < journal.local_size(); ++patch) {
            const auto cells = static_cast<std::uint64_t>(journal.box(patch).numPts());
            if (cells > (local.max_size() - count) / width)
              throw std::length_error("Field read journal rows exceed native storage");
            count += static_cast<std::size_t>(cells) * width;
            // Host mirrors coexist with the device journal and encoded source rows.
            budget.charge(cells, std::uint64_t(journal.ncomp()) * sizeof(Real));
          }
          budget.charge(count, sizeof(std::uint64_t));
          local.resize(count);
          for (std::size_t patch = 0; patch < journal.local_size(); ++patch) {
            const auto& fab = journal.fab(patch);
            auto host = fab.create_host_mirror();
            fab.copy_to_host(host);
            const auto view = fab.view();
            const auto box = journal.box(patch);
            const auto extent = box.extent();
            for (std::int64_t ordinal = 0; ordinal < box.numPts(); ++ordinal) {
              std::int64_t quotient = ordinal, offset = 0;
              for (int axis = 0; axis < Dim; ++axis) {
                const auto coordinate = box.lo[axis] + quotient % extent[axis];
                quotient /= extent[axis];
                local[cursor++] = std::bit_cast<std::uint64_t>(std::int64_t(coordinate));
                offset += (coordinate - view.origin[axis]) * view.strides[axis];
              }
              for (int component = 0; component < journal.ncomp(); ++component) {
                const Real value =
                    host(static_cast<std::size_t>(offset + component * view.component_stride));
                if (component == 0 || component % 2 == 1) {
                  if (value != Real(0) && value != Real(1))
                    throw std::invalid_argument("Field read journal has a failed or invalid flag");
                  local[cursor++] = value == Real(0) ? 0 : 1;
                } else {
                  local[cursor++] = std::bit_cast<std::uint64_t>(value);
                }
              }
            }
          }
        },
        "Field read journal extraction failed");

    auto shards = field_read_detail::gather(lane, local, budget);
    std::vector<std::span<const std::uint64_t>> views;
    std::string result;
    field_read_detail::phase(
        lane,
        [&] {
          // Replication is accepted only after exact comparison of all replicas.
          // Partitioned layouts keep every shard, including empty ranks.
          const bool replicated = source.distribution().replicated();
          if (replicated)
            for (std::size_t rank = 1; rank < shards.size(); ++rank)
              if (shards[rank] != shards.front())
                throw std::invalid_argument("replicated Field read journals differ across ranks");
          budget.charge(replicated ? 1 : shards.size(), sizeof(views.front()));
          views.reserve(replicated ? 1 : shards.size());
          for (std::size_t rank = 0; rank < (replicated ? 1 : shards.size()); ++rank)
            views.emplace_back(shards[rank]);
          std::uint64_t cells = 0, expected = 0;
          const auto checked_add = [](std::uint64_t& total, std::uint64_t amount) {
            if (amount > std::numeric_limits<std::uint64_t>::max() - total)
              throw std::overflow_error("Field read cell cardinality overflows uint64");
            total += amount;
          };
          for (const auto& box : source.layout().boxes())
            checked_add(expected, static_cast<std::uint64_t>(box.numPts()));
          for (const auto view : views)
            checked_add(cells, view.size() / width);
          if (cells != expected)
            throw std::invalid_argument(
                "Field read journal does not cover the exact source layout");
          // The codec additionally charges sources, indices, strings and output;
          // double-counting retained rows is conservative and remains user-visible.
          result = FieldReadClosure<Dim>::encode({schema, support, authorities, views},
                                                 budget.remaining());
        },
        "Field read closure preparation failed");
    const std::array<ExactOrderedBytePair, 1> identity{
        ExactOrderedBytePair{"field-read-closure", result}};
    if (!all_ranks_agree_exact_ordered_byte_pairs(identity, lane))
      throw std::invalid_argument("Field read closures differ across execution ranks");
    return result;
  }
};
}  // namespace pops::runtime::program
