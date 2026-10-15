/// @file
/// @brief Native, bit-preserving rematerialization of finite continuation Field
/// images.

#pragma once

#include <pops/runtime/program/continuation_image.hpp>

#include <bit>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <tuple>
#include <vector>

namespace pops::runtime::program {

/// Explicit workspace for archive conversion. Charges are logical native
/// storage sizes, including native fields, structural nodes, references and
/// temporary host copies. No allocation modifies accepted storage; allocation
/// failures abort the outer transaction.
class ContinuationMigrationBudget {
 public:
  explicit ContinuationMigrationBudget(std::uint64_t bytes) : capacity_(bytes) {}
  void charge(std::uint64_t count, std::uint64_t width = 1) {
    if (width && count > (capacity_ - used_) / width)
      throw std::length_error("continuation migration workspace capacity exceeded");
    used_ += count * width;
  }
  std::uint64_t used() const noexcept { return used_; }
  std::uint64_t remaining() const noexcept { return capacity_ - used_; }

 private:
  std::uint64_t capacity_, used_ = 0;
};

/// Prototype provider for one authenticated archived rank and one AMR level. It
/// reads only layout metadata, verifies it against the actual target prototypes
/// and sealed source owner table, and constructs ordinary native MultiFabs. It
/// creates no execution lane or emulator.
template <int Dim>
class ContinuationFieldArchive {
 public:
  using Codec = ContinuationCodec<Dim>;
  using Field = typename Codec::Field;
  using Prototypes = typename Codec::Prototypes;
  struct SourceTopology {
    std::span<const int> owners;
    bool replicated;
  };
  using SourceSupports = std::function<SourceTopology(const ContinuationFieldSupport::Ref&)>;

  ContinuationFieldArchive(Prototypes targets, std::span<const int> source_owners, bool replicated,
                           int source_ranks, int source_rank, ContinuationMigrationBudget& budget,
                           SourceSupports source_supports = {})
      : targets_(std::move(targets)),
        owners_(source_owners),
        replicated_(replicated),
        source_ranks_(source_ranks),
        source_rank_(source_rank),
        budget_(budget),
        source_supports_(std::move(source_supports)) {
    if (!targets_ || source_ranks_ < 1 || source_rank_ < 0 || source_rank_ >= source_ranks_ ||
        (replicated_ && !owners_.empty()))
      throw std::invalid_argument("invalid authenticated continuation archive topology");
    for (const int owner : owners_)
      if (owner < 0 || owner >= source_ranks_)
        throw std::invalid_argument(
            "continuation archive owner is outside its source communicator");
  }

  const Field& prototype(int owner, const ContinuationFieldSupport::Ref& support = {}) const {
    const auto found = fields_.find(key_(owner, support));
    if (found == fields_.end())
      throw std::invalid_argument("continuation archive Field was not decoded under its authority");
    return *found->second.field;
  }

  const Field& resolve(const ContinuationType& type, std::span<const std::uint8_t> bytes,
                       const ContinuationFieldSupport::Ref& support = {}) {
    if (support && support->program_owner != type.field_owner)
      throw std::invalid_argument("continuation archive Field support names a foreign owner");
    if (support && !source_supports_)
      throw std::invalid_argument("continuation archive requires authenticated producer topology");
    const auto topology =
        support ? source_supports_(support) : SourceTopology{owners_, replicated_};
    const auto source_owners = topology.owners;
    const bool replicated_source = topology.replicated;
    if (replicated_source && !source_owners.empty())
      throw std::invalid_argument("replicated continuation source has an owner table");
    for (int owner : source_owners)
      if (owner < 0 || owner >= source_ranks_)
        throw std::invalid_argument("continuation producer topology names a foreign rank");
    const Field& target = targets_(type.field_owner, support);
    continuation_detail::Reader in(bytes);
    if (in.word() != static_cast<std::uint64_t>(type.field_components))
      throw std::invalid_argument("continuation archive Field components differ from its schema");
    Index<Dim> local{}, origin{};
    Extent<Dim> ghosts{}, extent{};
    auto index = [&]() {
      const auto value = std::bit_cast<std::int64_t>(in.word());
      if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
        throw std::invalid_argument("continuation archive coordinate exceeds native indices");
      return static_cast<int>(value);
    };
    for (int d = 0; d < Dim; ++d) {
      const auto ghost = in.word();
      if (ghost > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("continuation archive ghost extent exceeds native indices");
      ghosts[d] = static_cast<int>(ghost);
      local[d] = index();
      origin[d] = index();
      const auto size = in.word();
      if (size > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("continuation archive rank extent exceeds native indices");
      extent[d] = static_cast<int>(size);
    }
    const mesh::RankSpace<Dim> ranks(origin, extent);
    if (ranks.size() != static_cast<std::size_t>(source_ranks_) ||
        ranks.linear_rank(local) != static_cast<std::size_t>(source_rank_))
      throw std::invalid_argument("continuation Field archive names a different source rank");
    const auto replicated = in.word(), count = in.word();
    if (replicated > 1 || bool(replicated) != replicated_source ||
        target.distribution().replicated() != replicated_source ||
        count != target.layout().size() || (!replicated_source && source_owners.size() != count))
      throw std::invalid_argument(
          "continuation archive layout differs from authenticated topology");
    const auto width = static_cast<std::uint64_t>(Dim) * (replicated_source ? 16 : 24);
    if (count > (in.bytes.size() - in.at) / width)
      throw std::invalid_argument("continuation archive patch descriptors are truncated");
    budget_.charge(count, sizeof(Index<Dim>));
    std::vector<Index<Dim>> owners;
    if (!replicated_source)
      owners.resize(static_cast<std::size_t>(count));
    for (std::size_t p = 0; p < count; ++p) {
      Index<Dim> owner{};
      for (int d = 0; d < Dim; ++d) {
        if (index() != target.layout()[p].lo[d] || index() != target.layout()[p].hi[d])
          throw std::invalid_argument("continuation migration changes physical patch geometry");
        if (!replicated_source)
          owner[d] = index();
      }
      if (!replicated_source) {
        if (ranks.linear_rank(owner) != static_cast<std::size_t>(source_owners[p]))
          throw std::invalid_argument("continuation Field owner differs from its sealed archive");
        owners[p] = owner;
      }
    }
    const auto distribution =
        replicated_source
            ? mesh::Distribution<Dim>::replicated(target.layout(), ranks)
            : mesh::Distribution<Dim>::partitioned(target.layout(), ranks, std::move(owners));
    const auto found = fields_.find(key_(type.field_owner, support));
    if (found != fields_.end()) {
      if (found->second.field->distribution() != distribution ||
          found->second.field->local_rank() != local)
        throw std::invalid_argument("one continuation owner names incompatible archive layouts");
      return *found->second.field;
    }
    // The prototype supplies geometry only; one component suffices. The
    // existing codec allocates the schema's exact component count after
    // checking the complete payload.
    Extent<Dim> prototype_ghosts{};
    std::uint64_t elements = 0;
    for (std::size_t p = 0; p < count; ++p) {
      if (!distribution.is_local(p, local))
        continue;
      (void)continuation_detail::field_storage_elements(target.layout()[p], ghosts,
                                                        type.field_components, bytes.size() / 8);
      const auto cells = continuation_detail::field_storage_elements(
          target.layout()[p], prototype_ghosts, 1, budget_.remaining() / sizeof(Real));
      if (cells > std::numeric_limits<std::uint64_t>::max() - elements)
        throw std::overflow_error("continuation archive native field size overflows");
      elements += cells;
    }
    budget_.charge(elements, sizeof(Real));
    budget_.charge(sizeof(Field) + sizeof(typename decltype(fields_)::value_type));
    auto field = std::make_unique<Field>(target.layout(), distribution, local, 1, prototype_ghosts);
    auto [entry, inserted] =
        fields_.emplace(key_(type.field_owner, support), CachedField{support, std::move(field)});
    (void)inserted;
    return *entry->second.field;
  }

 private:
  using Key = std::tuple<int, bool, std::string_view, std::string_view, int, int, std::uint64_t,
                         std::uint64_t>;
  static Key key_(int owner, const ContinuationFieldSupport::Ref& support) {
    if (!support)
      return {owner, false, {}, {}, -1, -1, 0, 0};
    support->validate();
    return {owner,
            true,
            support->program_identity,
            support->origin_point,
            support->program_owner,
            support->producer_level,
            support->topology_epoch,
            support->materialization_generation};
  }
  Prototypes targets_;
  std::span<const int> owners_;
  bool replicated_;
  int source_ranks_, source_rank_;
  ContinuationMigrationBudget& budget_;
  SourceSupports source_supports_;
  struct CachedField {
    ContinuationFieldSupport::Ref support;
    std::unique_ptr<Field> field;
  };
  std::map<Key, CachedField> fields_;
};

/// Join every authenticated old-rank image and copy each Field patch to its
/// actual target owner. Scalar/control leaves and finite capabilities retain
/// their exact logical identity. The first profile deliberately rejects
/// distributed arrays; they require GID redistribution.
template <int Dim>
ContinuationValueRef<Dim> rematerialize_continuation_fields(
    std::span<const ContinuationValueRef<Dim>> sources,
    const typename ContinuationCodec<Dim>::Prototypes& targets,
    ContinuationMigrationBudget& budget) {
  using Value = ContinuationValue<Dim>;
  using Ref = typename Value::Ref;
  using Field = typename Value::Field;
  using Bits = std::conditional_t<sizeof(Real) == 8, std::uint64_t, std::uint32_t>;
  if (sources.empty() || !targets)
    throw std::invalid_argument("continuation migration requires the complete source-rank table");
  struct Frame {
    std::vector<const Value*> sources;
    std::vector<Ref> children;
    std::size_t next = 0;
  };
  std::vector<Frame> pending;
  FieldExchangeExpressions::RematerializationCache field_expression_cache;
  auto push = [&](std::vector<const Value*> nodes) {
    budget.charge(sizeof(Frame));
    const auto* first = nodes.front();
    if (!first)
      throw std::invalid_argument("continuation migration contains an absent value");
    for (const auto* node : nodes)
      if (!node || node->kind() != first->kind() ||
          node->members().size() != first->members().size())
        throw std::invalid_argument("continuation archive structures differ between source ranks");

    budget.charge(first->members().size(), sizeof(Ref));
    Frame frame{std::move(nodes), {}, 0};
    frame.children.reserve(first->members().size());
    pending.push_back(std::move(frame));
  };
  budget.charge(sources.size(), sizeof(const Value*));
  std::vector<const Value*> roots;
  roots.reserve(sources.size());
  for (const auto& source : sources)
    roots.push_back(source.get());
  push(std::move(roots));
  Ref result;
  while (!pending.empty()) {
    auto& frame = pending.back();
    const auto& first = *frame.sources.front();
    if (!continuation_distributed_kind(first.kind()) && frame.next < first.members().size()) {
      budget.charge(frame.sources.size(), sizeof(const Value*));
      std::vector<const Value*> children;
      children.reserve(frame.sources.size());
      for (const auto* source : frame.sources)
        children.push_back(source->members()[frame.next].get());
      ++frame.next;
      push(std::move(children));
      continue;
    }
    auto equal_leaf = [&](auto projection) {
      const auto expected = projection(first);
      for (const auto* source : frame.sources)
        if (projection(*source) != expected)
          throw std::invalid_argument("continuation replicated value differs between source ranks");
    };
    if (const auto& domain = first.domain_ref()) {
      if (!domain->realm()->replicated() || !domain->realm()->synchronized())
        throw std::invalid_argument("migration requires an authenticated replicated finite realm");
      for (const auto* source : frame.sources) {
        const auto& other = source->domain_ref();
        if (!other || other != domain)
          throw std::invalid_argument("continuation migration did not intern exact finite aliases");
      }
    }
    budget.charge(sizeof(Value));
    switch (first.kind()) {
      case ContinuationKind::Boolean:
        equal_leaf([](const Value& v) { return v.boolean(); });
        result = Value::boolean(first.boolean());
        break;
      case ContinuationKind::Int64:
        equal_leaf([](const Value& v) { return v.integer(); });
        result = Value::integer(first.integer());
        break;
      case ContinuationKind::Float64:
        equal_leaf([](const Value& v) { return std::bit_cast<std::uint64_t>(v.real()); });
        result = Value::real(first.real());
        break;
      case ContinuationKind::Domain:
        result = Value::domain(first.domain_ref());
        break;
      case ContinuationKind::Index:
        equal_leaf([](const Value& v) { return v.index_offset(); });
        result = Value::index(first.domain_ref(), first.index_offset());
        break;
      case ContinuationKind::Array:
        result = Value::array(first.domain_ref(), std::move(frame.children));
        break;
      case ContinuationKind::Product:
        result = Value::product(std::move(frame.children));
        break;
      case ContinuationKind::Variant:
        equal_leaf([](const Value& v) { return v.arm(); });
        result = Value::variant(first.arm(), std::move(frame.children.at(0)));
        break;
      case ContinuationKind::Field: {
        const Field& target = targets(first.field_owner(), first.field_support());
        const auto count = target.layout().size();
        budget.charge(count, sizeof(std::pair<const Field*, std::size_t>));
        std::vector<std::pair<const Field*, std::size_t>> patches(count, {nullptr, 0});
        for (const auto* source : frame.sources) {
          const Field& old = source->field();
          const auto& support = source->field_support();
          const auto& expected = first.field_support();
          if (bool(support) != bool(expected) || (support && *support != *expected))
            throw std::invalid_argument(
                "continuation Field producer support differs between ranks");
          if (source->field_owner() != first.field_owner() ||
              source->scientific_contract() != first.scientific_contract() ||
              old.layout() != target.layout() || old.ghosts() != first.field().ghosts() ||
              old.ncomp() != first.field().ncomp() ||
              old.distribution().replicated() != target.distribution().replicated())
            throw std::invalid_argument(
                "continuation Field migration changes "
                "scientific/storage authority");
          for (std::size_t p = 0; p < old.local_size(); ++p) {
            const auto global = old.global_index(p);
            auto& entry = patches.at(global);
            if (!entry.first) {
              entry = {&old, p};
              continue;
            }
            if (!old.distribution().replicated())
              throw std::invalid_argument("continuation Field archive has duplicate patch owners");
            const auto& prior = entry.first->fab(entry.second);
            const auto& current = old.fab(p);
            if (prior.size() != current.size())
              throw std::invalid_argument("replicated continuation Field extents disagree");
            budget.charge(prior.size(), 2 * sizeof(Real));
            auto a = prior.create_host_mirror(), b = current.create_host_mirror();
            prior.copy_to_host(a);
            current.copy_to_host(b);
            for (std::size_t i = 0; i < a.size(); ++i)
              if (std::bit_cast<Bits>(a(i)) != std::bit_cast<Bits>(b(i)))
                throw std::invalid_argument("replicated continuation Field bits disagree");
          }
        }
        for (const auto& patch : patches)
          if (!patch.first)
            throw std::invalid_argument("continuation Field archive omits a global patch");
        for (std::size_t p = 0; p < target.local_size(); ++p) {
          const auto& entry = patches[target.global_index(p)];
          // Native destination plus its final owning Value copy and two host
          // mirrors.
          budget.charge(entry.first->fab(entry.second).size(), 4 * sizeof(Real));
        }
        budget.charge(2, sizeof(Field));
        Field next(target.layout(), target.distribution(), target.local_rank(),
                   first.field().ncomp(), first.field().ghosts());
        for (std::size_t p = 0; p < next.local_size(); ++p) {
          const auto& entry = patches[next.global_index(p)];
          const auto& source = entry.first->fab(entry.second);
          auto a = source.create_host_mirror();
          auto b = next.fab(p).create_host_mirror();
          source.copy_to_host(a);
          if (a.size() != b.size())
            throw std::invalid_argument("continuation Field target patch extent changed");
          for (std::size_t i = 0; i < a.size(); ++i)
            b(i) = a(i);
          next.fab(p).copy_from_host(b);
        }
        FieldExchangeExpressions::Ref expression;
        if (std::any_of(frame.sources.begin(), frame.sources.end(), [](const Value* value) {
              return bool(value->field_exchange_expression());
            })) {
          budget.charge(frame.sources.size(), sizeof(FieldExchangeExpressions::Ref));
          std::vector<FieldExchangeExpressions::Ref> source_expressions;
          source_expressions.reserve(frame.sources.size());
          for (const auto* value : frame.sources)
            source_expressions.push_back(value->field_exchange_expression());
          const auto remaining = static_cast<std::size_t>(
              std::min<std::uint64_t>(budget.remaining(), std::numeric_limits<std::size_t>::max()));
          const FieldExchangeExpressions::Limits limits{remaining, remaining, remaining, remaining};
          budget.charge(
              FieldExchangeExpressions::rematerialized_storage_bound(source_expressions, limits));
          expression = FieldExchangeExpressions::rematerialize(
              source_expressions, limits,
              [&](int owner, const FieldExchangeExpressions::Cell& cell) {
                if (cell.size() != Dim)
                  throw std::invalid_argument("field expression cell has a different dimension");
                Index<Dim> coordinate;
                for (int d = 0; d < Dim; ++d) {
                  if (cell[d] < std::numeric_limits<int>::min() ||
                      cell[d] > std::numeric_limits<int>::max())
                    throw std::invalid_argument("field expression cell exceeds native index range");
                  coordinate[d] = static_cast<int>(cell[d]);
                }
                const auto& prototype = targets(owner, first.field_support());
                for (std::size_t p = 0; p < prototype.local_size(); ++p)
                  if (prototype.layout()[prototype.global_index(p)].contains(coordinate))
                    return true;
                return false;
              },
              &field_expression_cache);
        }
        result = Value::field(next, first.field_owner(), first.scientific_contract(),
                              std::move(expression), first.field_support());
        break;
      }
      case ContinuationKind::DistributedDomain:
      case ContinuationKind::DistributedIndex:
      case ContinuationKind::DistributedArray:
        // The collective migration plan has already rebound every source alias.
        // All old-rank occurrences must denote the same newly issued native capability.
        for (const auto* source : frame.sources)
          if (source->distributed_domain_ref() != first.distributed_domain_ref() ||
              source->distributed_array_ref() != first.distributed_array_ref() ||
              (first.kind() == ContinuationKind::DistributedIndex &&
               source->distributed_index_offset() != first.distributed_index_offset()))
            throw std::invalid_argument(
                "distributed migration aliases were not rebound consistently");
        result = first.kind() == ContinuationKind::DistributedArray
                     ? Value::distributed_array(first.distributed_array_ref())
                 : first.kind() == ContinuationKind::DistributedIndex
                     ? Value::distributed_index(first.distributed_domain_ref(),
                                                first.distributed_index_offset())
                     : Value::distributed_domain(first.distributed_domain_ref());
        break;
    }
    pending.pop_back();
    if (!pending.empty())
      pending.back().children.push_back(std::move(result));
  }
  return result;
}

}  // namespace pops::runtime::program
