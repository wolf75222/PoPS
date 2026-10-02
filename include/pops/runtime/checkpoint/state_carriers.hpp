#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace pops::runtime::checkpoint {

/// POPSCAR1 stores patch-owned full storage, independent of destination ownership.
/// Words are little-endian uint64; signed coordinates, owner and shard use int64 bits.
/// shard=-1 denotes the merged archive. Replicated patches are consensus-checked across every source rank.
template <int Dim> struct StateCarrierPatch {
  std::uint64_t block = 0, level = 0, patch = 0, components = 0;
  std::int64_t owner = 0;
  std::array<std::int64_t, Dim> lo{}, hi{}, grown_lo{}, grown_hi{};
  std::vector<std::uint64_t> bits;
  auto key() const { return std::tuple(block, level, patch); }
  bool operator==(const StateCarrierPatch&) const = default;
};
template <int Dim> struct StateCarrierArchive {
  std::uint64_t real_bits = 0, ranks = 0, levels = 0;
  std::int64_t shard = -1;
  std::vector<std::string> blocks;
  std::vector<StateCarrierPatch<Dim>> patches;
};

/// Configured storage envelope for the existing POPSCAR1 wire, capacity contract@1.
template <int Dim> struct StateCarrierStorageCapacity {
  std::string name;
  std::uint64_t components = 0;
  std::array<std::uint64_t, Dim> ghosts{};
};

template <int Dim>
std::uint64_t state_carriers_byte_capacity(std::span<const std::uint64_t> level_cells,
    std::span<const StateCarrierStorageCapacity<Dim>> blocks) {
  static_assert(Dim >= 1 && Dim <= 3);
  const auto add = [](std::uint64_t a, std::uint64_t b) {
    if (a > std::numeric_limits<std::uint64_t>::max() - b)
      throw std::length_error("state carrier capacity addition overflows");
    return a + b;
  };
  const auto mul = [](std::uint64_t a, std::uint64_t b) {
    if (b && a > std::numeric_limits<std::uint64_t>::max() / b)
      throw std::length_error("state carrier capacity multiplication overflows");
    return a * b;
  };
  if (level_cells.empty() || blocks.empty())
    throw std::invalid_argument("state carrier capacity requires configured levels and blocks");
  std::uint64_t cells = 0;
  for (auto count : level_cells) {
    if (!count) throw std::invalid_argument("state carrier capacity level is empty");
    cells = add(cells, count);
  }
  // Magic + six authority words + patch count; each name is size-prefixed.
  std::uint64_t bytes = 8 + 7 * 8;
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    const auto& block = blocks[i];
    if (block.name.empty() || !block.components)
      throw std::invalid_argument("state carrier capacity block authority is empty");
    for (std::size_t j = 0; j < i; ++j)
      if (block.name == blocks[j].name)
        throw std::invalid_argument("state carrier capacity block authority is duplicated");
    bytes = add(bytes, add(8, block.name.size()));
    std::uint64_t grown_per_valid = block.components;
    for (auto ghost : block.ghosts)
      grown_per_valid = mul(grown_per_valid, add(1, mul(2, ghost)));
    // Each disjoint patch contains >=1 valid cell. For any side L>=1,
    // L+2g <= L*(1+2g); this includes maximal one-cell fragmentation.
    const auto patch_bytes = add((6 + 4 * Dim) * 8, mul(grown_per_valid, 8));
    bytes = add(bytes, mul(cells, patch_bytes));
  }
  return bytes;
}

namespace state_carrier_detail {
inline std::uint64_t product(std::uint64_t a, std::uint64_t b) {
  if (b && a > std::numeric_limits<std::uint64_t>::max() / b)
    throw std::invalid_argument("state carrier shape overflows");
  return a * b;
}
template <int Dim> void validate(const StateCarrierArchive<Dim>& a) {
  if ((a.real_bits != 32 && a.real_bits != 64) || !a.ranks || !a.levels || a.blocks.empty() ||
      a.ranks > std::uint64_t(std::numeric_limits<std::int64_t>::max()) || a.shard < -1 ||
      (a.shard >= 0 && std::uint64_t(a.shard) >= a.ranks))
    throw std::invalid_argument("state carrier archive has invalid authority");
  for (std::size_t i = 0; i < a.blocks.size(); ++i)
    if (a.blocks[i].empty() ||
        std::find(a.blocks.begin(), a.blocks.begin() + i, a.blocks[i]) != a.blocks.begin() + i)
      throw std::invalid_argument("state carrier block identities are empty or duplicated");
  for (std::size_t i = 0; i < a.patches.size(); ++i) {
    const auto& p = a.patches[i];
    if (p.block >= a.blocks.size() || p.level >= a.levels || !p.components || p.owner < -1 ||
        (p.owner >= 0 && std::uint64_t(p.owner) >= a.ranks) ||
        (a.shard >= 0 && p.owner >= 0 && p.owner != a.shard) ||
        (i && !(a.patches[i - 1].key() < p.key())))
      throw std::invalid_argument("state carrier patch has duplicate/order/owner authority error");
    auto count = p.components;
    for (int d = 0; d < Dim; ++d) {
      if (p.lo[d] > p.hi[d] || p.grown_lo[d] > p.lo[d] || p.grown_hi[d] < p.hi[d])
        throw std::invalid_argument("state carrier valid/grown shape is invalid");
      auto extent = std::uint64_t(p.grown_hi[d]) - std::uint64_t(p.grown_lo[d]);
      if (extent == std::numeric_limits<std::uint64_t>::max())
        throw std::invalid_argument("state carrier extent overflows");
      count = product(count, extent + 1);
    }
    if (count != p.bits.size())
      throw std::invalid_argument("state carrier payload differs from shape");
    if (a.real_bits == 32 && std::any_of(p.bits.begin(), p.bits.end(),
                                      [](auto v) { return v > UINT32_MAX; }))
      throw std::invalid_argument("state carrier bits exceed native width");
  }
}
}  // namespace state_carrier_detail

template <int Dim>
std::vector<std::uint8_t> encode_state_carriers(const StateCarrierArchive<Dim>& a) {
  static_assert(Dim >= 1 && Dim <= 3);
  state_carrier_detail::validate(a);
  std::vector<std::uint8_t> out{'P', 'O', 'P', 'S', 'C', 'A', 'R', '1'};
  auto word = [&](std::uint64_t v) {
    for (int b = 0; b < 8; ++b) out.push_back(std::uint8_t(v >> (8 * b)));
  };
  word(Dim); word(a.real_bits); word(a.ranks);
  word(std::bit_cast<std::uint64_t>(a.shard)); word(a.levels); word(a.blocks.size());
  for (const auto& name : a.blocks) {
    word(name.size()); out.insert(out.end(), name.begin(), name.end());
  }
  word(a.patches.size());
  for (const auto& p : a.patches) {
    word(p.block); word(p.level); word(p.patch); word(p.components);
    word(std::bit_cast<std::uint64_t>(p.owner));
    for (int d = 0; d < Dim; ++d) {
      word(std::bit_cast<std::uint64_t>(p.lo[d]));
      word(std::bit_cast<std::uint64_t>(p.hi[d]));
      word(std::bit_cast<std::uint64_t>(p.grown_lo[d]));
      word(std::bit_cast<std::uint64_t>(p.grown_hi[d]));
    }
    word(p.bits.size()); for (auto v : p.bits) word(v);
  }
  return out;
}

template <int Dim>
StateCarrierArchive<Dim> decode_state_carriers(std::span<const std::uint8_t> bytes) {
  constexpr std::array<std::uint8_t, 8> magic{'P', 'O', 'P', 'S', 'C', 'A', 'R', '1'};
  if (bytes.size() < 64 || !std::equal(magic.begin(), magic.end(), bytes.begin()))
    throw std::invalid_argument("state carrier archive has invalid POPSCAR1 header");
  std::size_t at = 8;
  auto word = [&]() {
    if (bytes.size() - at < 8) throw std::invalid_argument("state carrier archive is truncated");
    std::uint64_t v = 0;
    for (int b = 0; b < 8; ++b) v |= std::uint64_t(bytes[at++]) << (8 * b);
    return v;
  };
  if (word() != Dim) throw std::invalid_argument("state carrier dimension changed");
  StateCarrierArchive<Dim> a;
  a.real_bits = word(); a.ranks = word(); a.shard = std::bit_cast<std::int64_t>(word());
  a.levels = word(); auto n = word();
  if (n > (bytes.size() - at) / 8)
    throw std::invalid_argument("state carrier names exceed bytes");
  for (std::uint64_t i = 0; i < n; ++i) {
    auto length = word();
    if (length > bytes.size() - at) throw std::invalid_argument("state carrier name exceeds bytes");
    a.blocks.emplace_back(reinterpret_cast<const char*>(bytes.data() + at), std::size_t(length));
    at += std::size_t(length);
  }
  n = word();
  if (n > (bytes.size() - at) / ((6 + 4 * Dim) * 8))
    throw std::invalid_argument("state carrier rows exceed bytes");
  for (std::uint64_t i = 0; i < n; ++i) {
    StateCarrierPatch<Dim> p;
    p.block = word(); p.level = word(); p.patch = word(); p.components = word();
    p.owner = std::bit_cast<std::int64_t>(word());
    for (int d = 0; d < Dim; ++d) {
      p.lo[d] = std::bit_cast<std::int64_t>(word()); p.hi[d] = std::bit_cast<std::int64_t>(word());
      p.grown_lo[d] = std::bit_cast<std::int64_t>(word());
      p.grown_hi[d] = std::bit_cast<std::int64_t>(word());
    }
    auto count = word();
    if (count > (bytes.size() - at) / 8)
      throw std::invalid_argument("state carrier payload exceeds bytes");
    p.bits.reserve(std::size_t(count));
    for (std::uint64_t j = 0; j < count; ++j) p.bits.push_back(word());
    a.patches.push_back(std::move(p));
  }
  if (at != bytes.size()) throw std::invalid_argument("state carrier archive has trailing bytes");
  state_carrier_detail::validate(a);
  return a;
}

template <int Dim>
StateCarrierArchive<Dim> merge_state_carrier_shards(const std::vector<std::string>& shards) {
  if (shards.empty()) throw std::invalid_argument("state carrier archive has no source shards");
  StateCarrierArchive<Dim> result;
  std::map<std::tuple<std::uint64_t, std::uint64_t, std::uint64_t>, std::size_t> replicas;
  for (std::size_t rank = 0; rank < shards.size(); ++rank) {
    const auto& s = shards[rank];
    auto a = decode_state_carriers<Dim>(
        {reinterpret_cast<const std::uint8_t*>(s.data()), s.size()});
    if (a.shard != std::int64_t(rank) || a.ranks != shards.size())
      throw std::invalid_argument("state carrier source shard authority changed");
    for (const auto& row : a.patches) if (row.owner == -1) ++replicas[row.key()];
    if (rank == 0) { result = a; result.shard = -1; }
    else {
      if (a.blocks != result.blocks || a.levels != result.levels || a.real_bits != result.real_bits)
        throw std::invalid_argument("state carrier source envelopes differ");
      result.patches.insert(result.patches.end(), std::make_move_iterator(a.patches.begin()),
                            std::make_move_iterator(a.patches.end()));
    }
  }
  std::sort(result.patches.begin(), result.patches.end(),
            [](const auto& a, const auto& b) { return a.key() < b.key(); });
  std::vector<StateCarrierPatch<Dim>> unique;
  for (auto& row : result.patches) {
    if (!unique.empty() && unique.back().key() == row.key()) {
      if (row.owner != -1 || unique.back() != row)
        throw std::invalid_argument("state carrier duplicate or divergent replicated source patch");
    } else unique.push_back(std::move(row));
  }
  for (const auto& [key, count] : replicas)
    if (count != shards.size())
      throw std::invalid_argument("state carrier replicated source patch is missing a rank");
  result.patches = std::move(unique);
  state_carrier_detail::validate(result);
  return result;
}

/// Optional host inspection counters. They measure actual BVH traversal and allocated index
/// storage; they do not change validation, payload format or impose a patch-count limit.
struct StateCarrierSpatialValidationStats {
  std::uint64_t node_tests = 0, leaf_pairs = 0;
  std::size_t peak_index_bytes = 0;
};

namespace state_carrier_detail {
/// Balanced BVH over closed int64 cell boxes. Midpoints never add signed endpoints, and spans
/// use exact unsigned subtraction. The index only supplies candidates; closed-box intersection
/// remains the authority. Median splitting guarantees bounded recursion for arbitrary geometry.
template <int Dim> class SpatialIndex {
  struct Node {
    std::array<std::int64_t, Dim> lo{}, hi{};
    std::size_t left = 0, right = 0, patch = 0;
    bool leaf = false;
  };
  std::span<const StateCarrierPatch<Dim>> rows_;
  std::vector<std::size_t> order_;
  std::vector<Node> nodes_;
  std::size_t build(std::size_t begin, std::size_t end) {
    const auto slot = nodes_.size();
    Node node;
    node.lo = rows_[order_[begin]].lo; node.hi = rows_[order_[begin]].hi;
    for (std::size_t at = begin + 1; at < end; ++at)
      for (int d = 0; d < Dim; ++d) {
        node.lo[d] = std::min(node.lo[d], rows_[order_[at]].lo[d]);
        node.hi[d] = std::max(node.hi[d], rows_[order_[at]].hi[d]);
      }
    nodes_.push_back(node);
    if (end - begin == 1) {
      nodes_[slot].leaf = true; nodes_[slot].patch = order_[begin];
      return slot;
    }
    int axis = 0;
    for (int d = 1; d < Dim; ++d)
      if (std::uint64_t(node.hi[d]) - std::uint64_t(node.lo[d]) >
          std::uint64_t(node.hi[axis]) - std::uint64_t(node.lo[axis])) axis = d;
    const auto middle = begin + (end - begin) / 2;
    std::nth_element(order_.begin() + begin, order_.begin() + middle, order_.begin() + end,
      [&](auto a, auto b) {
        const auto ca = std::midpoint(rows_[a].lo[axis], rows_[a].hi[axis]);
        const auto cb = std::midpoint(rows_[b].lo[axis], rows_[b].hi[axis]);
        return ca < cb || (ca == cb && a < b);
      });
    // Recursion can grow nodes_, so retain an index rather than a Node reference.
    const auto left = build(begin, middle), right = build(middle, end);
    nodes_[slot].left = left; nodes_[slot].right = right;
    return slot;
  }
  bool query(std::size_t query_patch, std::size_t slot,
             StateCarrierSpatialValidationStats* stats) const {
    const auto& node = nodes_[slot];
    if (node.leaf && node.patch <= query_patch) return false;
    if (stats) ++stats->node_tests;
    const auto& row = rows_[query_patch];
    for (int d = 0; d < Dim; ++d)
      if (row.lo[d] > node.hi[d] || node.lo[d] > row.hi[d]) return false;
    if (node.leaf) {
      if (stats) ++stats->leaf_pairs;
      return true;
    }
    return query(query_patch, node.left, stats) || query(query_patch, node.right, stats);
  }
 public:
  explicit SpatialIndex(std::span<const StateCarrierPatch<Dim>> rows) : rows_(rows) {
    if (rows.empty()) throw std::invalid_argument("state carrier spatial index is empty");
    if (rows.size() > std::numeric_limits<std::size_t>::max() / 2)
      throw std::length_error("state carrier spatial index node count overflows");
    order_.resize(rows.size()); std::iota(order_.begin(), order_.end(), std::size_t{0});
    nodes_.reserve(rows.size() * 2 - 1); build(0, rows.size());
  }
  void require_disjoint(StateCarrierSpatialValidationStats* stats = nullptr) const {
    if (stats) stats->peak_index_bytes = std::max(stats->peak_index_bytes,
        order_.capacity() * sizeof(std::size_t) + nodes_.capacity() * sizeof(Node));
    for (std::size_t patch = 0; patch < rows_.size(); ++patch)
      if (query(patch, 0, stats))
        throw std::invalid_argument("state carrier source valid patches overlap");
  }
};
}  // namespace state_carrier_detail

/// A global archive covers every block on every source level, with contiguous global patches.
/// Block carriers share physical patch geometry and source ownership; component/ghost widths
/// remain individual storage properties. This check never assumes the destination topology.
template <int Dim> void validate_complete_state_carriers(
    const StateCarrierArchive<Dim>& a, StateCarrierSpatialValidationStats* stats = nullptr) {
  state_carrier_detail::validate(a);
  if (a.shard != -1) throw std::invalid_argument("state carrier restore requires a merged archive");
  if (state_carrier_detail::product(a.blocks.size(), a.levels) > a.patches.size())
    throw std::invalid_argument("state carrier source block/level count exceeds actual rows");
  // Cache block-zero source ranges once. All subsequent blocks compare linearly to these exact
  // rows, rather than scanning the archive or searching all source levels for every patch.
  std::vector<std::pair<std::size_t, std::size_t>> first_ranges;
  first_ranges.reserve(static_cast<std::size_t>(a.levels));
  std::size_t at = 0;
  for (std::size_t block = 0; block < a.blocks.size(); ++block)
    for (std::uint64_t level = 0; level < a.levels; ++level) {
      std::uint64_t patch = 0;
      const auto begin = at;
      while (at < a.patches.size() && a.patches[at].block == block && a.patches[at].level == level) {
        const auto& p = a.patches[at++];
        if (p.patch != patch++) throw std::invalid_argument("state carrier source coverage has a gap");
        if (at > begin + 1 && p.components != a.patches[begin].components)
          throw std::invalid_argument("state carrier source component count changes within level");
        if (block) {
          const auto [first, count] = first_ranges[static_cast<std::size_t>(level)];
          if (p.patch >= count) throw std::invalid_argument("state carrier source patch coverage differs");
          const auto& row = a.patches[first + static_cast<std::size_t>(p.patch)];
          if (row.lo != p.lo || row.hi != p.hi || row.owner != p.owner)
            throw std::invalid_argument("state carrier source block geometries disagree");
        }
      }
      if (at == begin) throw std::invalid_argument("state carrier source block/level is missing");
      if (block) {
        if (first_ranges[static_cast<std::size_t>(level)].second != patch)
          throw std::invalid_argument("state carrier source patch coverage differs");
      } else {
        first_ranges.emplace_back(begin, at - begin);
        state_carrier_detail::SpatialIndex<Dim>(
            std::span<const StateCarrierPatch<Dim>>(a.patches).subspan(begin, at - begin))
            .require_disjoint(stats);
      }
    }
  if (at != a.patches.size()) throw std::invalid_argument("state carrier source coverage is incomplete");
}
}  // namespace pops::runtime::checkpoint
