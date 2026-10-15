#pragma once

#include <pops/runtime/program/continuation_field_migration.hpp>
#include <pops/runtime/program/distributed_array.hpp>
#include <set>

namespace pops::runtime::program {

/// A prepare-only native placement plan. Source identifiers are inert keys;
/// every target Registry/Domain/Array is freshly issued on the actual target lane.
/// The complete mapping is shared by all members of the accepted forest.
template <int Dim>
class ContinuationDistributedMigration {
 public:
  using Codec = ContinuationCodec<Dim>;
  using Image = typename Codec::ArchiveImage;
  using Array = typename Codec::ArchiveArray;
  using Domain = typename Codec::ArchiveDomain;
  using Key = std::array<std::uint64_t, 3>;
  using Origin = std::array<std::uint64_t, 2>;
  using Ref = ContinuationValueRef<Dim>;
  using Value = ContinuationValue<Dim>;
  struct Input {
    const Image* image;
    const Codec* codec;
    std::uint64_t wire_capacity;
    std::uint64_t registry_bytes;
    std::uint64_t registry_versions;
  };
  explicit ContinuationDistributedMigration(ContinuationMigrationBudget& budget)
      : budget_(budget) {}

  static Key key(distributed::Identity id) { return {id.issuer[0], id.issuer[1], id.ordinal}; }

  // Local parsing/cataloguing precedes all collective target construction.
  void prepare(std::span<const Input> inputs) {
    if (prepared_ || inputs.empty())
      throw std::logic_error("distributed migration requires one complete unpublished source plan");
    ranks_ = inputs.front().image->source_ranks;
    for (const auto& input : inputs) {
      const auto& image = *input.image;
      if (image.source_ranks != ranks_ || image.source_rank < 0 || image.source_rank >= ranks_)
        throw std::invalid_argument("distributed migration source cardinalities disagree");
      budget_.charge(sizeof(Cache) + sizeof(const Image*));
      auto [cache, fresh] = caches_.emplace(&image, Cache{});
      if (!fresh)
        throw std::invalid_argument("distributed migration repeats the same input image");
      budget_.charge(image.nodes.size(), sizeof(Ref));
      cache->second.values.resize(image.nodes.size());
      for (const auto& registry : image.registries) {
        if (registry.bytes != input.registry_bytes || registry.versions != input.registry_versions)
          throw std::invalid_argument("distributed migration registry changes installed resources");
        budget_.charge(sizeof(RegistryEntry));
        auto [entry, inserted] = registries_.emplace(registry.issuer, RegistryEntry{registry});
        if (!inserted && entry->second.source != registry)
          throw std::invalid_argument(
              "distributed migration issuer has conflicting resource policy");
      }
      for (const auto& domain : image.domains) {
        budget_.charge(sizeof(DomainEntry));
        auto [entry, inserted] = domains_.emplace(key(domain.identity), DomainEntry{&domain});
        const auto& prior = *entry->second.source;
        if (!inserted && (prior.family != domain.family || prior.capacity != domain.capacity ||
                          prior.offsets != domain.offsets))
          throw std::invalid_argument(
              "distributed migration changes an immutable domain descriptor");
      }
      for (const auto& array : image.arrays) {
        const auto id = key(array.identity);
        budget_.charge(sizeof(ArrayEntry));
        auto [position, inserted] = arrays_.emplace(id, ArrayEntry{});
        auto& entry = position->second;
        if (inserted) {
          budget_.charge(ranks_, sizeof(Source));
          entry.shards.resize(static_cast<std::size_t>(ranks_));
          entry.capacity = input.wire_capacity;
          entry.codec = input.codec;
        } else if (input.wire_capacity < entry.capacity) {
          entry.capacity = input.wire_capacity;
          entry.codec = input.codec;
        }
        auto& shard = entry.shards[image.source_rank];
        if (shard.image) {
          require_same_array_(*shard.image, *shard.array, image, array);
        } else
          shard = {&image, &array};
      }
    }
    for (auto& [id, entry] : arrays_) {
      for (const auto& shard : entry.shards)
        if (!shard.image)
          throw std::invalid_argument("distributed migration omits an Array source-rank shard");
      const auto& first = entry.shards.front();
      const auto& canonical = *first.array;
      for (const auto& shard : entry.shards) {
        const auto& a = *shard.array;
        if (key(shard.image->domains[a.domain].identity) !=
                key(first.image->domains[canonical.domain].identity) ||
            a.element != canonical.element ||
            bool(a.index_domain) != bool(canonical.index_domain) ||
            (a.index_domain && key(shard.image->domains[*a.index_domain].identity) !=
                                   key(first.image->domains[*canonical.index_domain].identity)))
          throw std::invalid_argument(
              "distributed migration Array descriptors differ between ranks");
        for (auto node : a.values)
          Codec::walk_archive(
              *shard.image, node,
              [&](const auto& value) {
                if (value.kind == ContinuationKind::DistributedArray) {
                  budget_.charge(sizeof(Key));
                  entry.dependencies.insert(key(shard.image->arrays[value.slot].identity));
                }
              },
              [&](std::uint64_t bytes) { budget_.charge(bytes); });
      }
    }
    // Validate each exact source registry's total retained bytes, counting each
    // immutable version once even when several forest images retain it.
    for (const auto& [origin, registry] : registries_) {
      std::uint64_t domain_bytes = 0, domain_count = 0;
      for (const auto& [id, domain] : domains_)
        if (domain.source->identity.issuer == origin) {
          const auto charge = distributed::Registry::domain_charge(domain.source->offsets.size());
          if (charge > registry.source.bytes - domain_bytes)
            throw std::length_error(
                "distributed migration source domains exceed installed storage");
          domain_bytes += charge;
          ++domain_count;
        }
      if (domain_count > registry.source.versions)
        throw std::length_error(
            "distributed migration source domain versions exceed installed storage");
      for (int rank = 0; rank < ranks_; ++rank) {
        auto bytes = domain_bytes;
        std::uint64_t count = 0;
        for (const auto& [id, array] : arrays_)
          if (array.shards[rank].array->identity.issuer == origin) {
            const auto charge = array.shards[rank].array->charge;
            if (charge > registry.source.bytes - bytes)
              throw std::length_error(
                  "distributed migration source arrays exceed installed storage");
            bytes += charge;
            ++count;
          }
        if (count > registry.source.versions)
          throw std::length_error(
              "distributed migration source Array versions exceed installed storage");
      }
    }
    // Global topological order, including cross-issuer dependencies which may
    // exist only in one old rank's shard. No local ordering is assumed globally.
    std::set<Key> done;
    while (order_.size() < arrays_.size()) {
      bool changed = false;
      for (const auto& [id, array] : arrays_) {
        if (done.contains(id))
          continue;
        bool ready = true;
        for (const auto& dependency : array.dependencies) {
          if (!arrays_.contains(dependency))
            throw std::invalid_argument("distributed migration has an absent immutable dependency");
          if (!done.contains(dependency))
            ready = false;
        }
        if (ready) {
          budget_.charge(2, sizeof(Key));
          order_.push_back(id);
          done.insert(id);
          changed = true;
        }
      }
      if (!changed)
        throw std::invalid_argument("distributed migration immutable Array graph is cyclic");
    }
    require_reachability_(inputs);
    prepared_ = true;
  }

  void materialize(const ExecutionLane& lane) {
    distributed::phase(lane, [&] {
      if (!prepared_ || materialized_)
        throw std::logic_error("distributed migration plan is not ready for target construction");
      budget_.charge(registries_.size(), sizeof(std::shared_ptr<distributed::Registry>));
      target_registries_.reserve(registries_.size());
    });
    for (auto& [origin, entry] : registries_) {
      // The Registry constructor communicates. Allocate its object storage and
      // converge failures first; ImmutableBorrow's construction is noexcept.
      struct RawRelease {
        void operator()(void* p) const noexcept { ::operator delete(p); }
      };
      std::unique_ptr<void, RawRelease> storage;
      distributed::phase(lane, [&] {
        budget_.charge(sizeof(distributed::Registry));
        storage.reset(::operator new(sizeof(distributed::Registry)));
      });
      auto* raw = new (storage.get()) distributed::Registry(lane);
      (void)storage.release();
      std::unique_ptr<distributed::Registry> owned(raw);
      distributed::phase(lane, [&] {
        entry.target = std::shared_ptr<distributed::Registry>(std::move(owned));
        if (registries_.contains(entry.target->issuer()) ||
            std::any_of(
                target_registries_.begin(), target_registries_.end(),
                [&](const auto& prior) { return prior->issuer() == entry.target->issuer(); }))
          throw std::logic_error("distributed migration target reused a source authority");
        target_registries_.push_back(entry.target);
      });
      entry.target->configure(entry.source.bytes, entry.source.versions);
    }
    for (auto& [id, entry] : domains_) {
      const auto& source = *entry.source;
      std::vector<std::int64_t> offsets;
      distributed::phase(lane, [&] {
        budget_.charge(std::uint64_t(lane.size()) + 1, sizeof(std::int64_t));
        offsets = distributed::partition_balanced(source.offsets.back(), lane.size(),
                                                  (std::uint64_t(lane.size()) + 1) * 8);
      });
      entry.target = registries_.at(source.identity.issuer)
                         .target->issue(source.family, source.capacity, offsets);
    }
    ContinuationTransportSession<Dim> transport;
    distributed::phase(lane, [&] {
      for (const auto& registry : target_registries_)
        transport.bind_distributed_registry(
            *registry, [registry](distributed::Identity version, distributed::DomainRef domain,
                                  const ContinuationType& element) {
              return distributed::resolve_array<Dim>(*registry, version, domain, element);
            });
    });
    for (const auto& id : order_) {
      auto& entry = arrays_.at(id);
      const auto& source = *entry.shards.front().array;
      const auto& image = *entry.shards.front().image;
      const auto domain = domains_.at(key(image.domains[source.domain].identity)).target;
      auto& registry = *registries_.at(source.identity.issuer).target;
      std::shared_ptr<ContinuationDistributedArray<Dim>> next;
      distributed::phase(lane, [&] {
        const auto type_identity =
            continuation_transport_type_identity(source.element, true, entry.capacity);
        budget_.charge(type_identity.size());
        std::uint64_t charge = 256 + 8 * domain->partition_offsets().size();
        const auto add = [&](std::uint64_t bytes) {
          if (bytes > UINT64_MAX - charge)
            throw std::length_error("target Array charge overflows");
          charge += bytes;
        };
        add(type_identity.size());
        const auto& offsets = image.domains[source.domain].offsets;
        for (std::int64_t gid = domain->begin(); gid < domain->end(); ++gid) {
          const auto rank = static_cast<std::size_t>(
              std::upper_bound(offsets.begin(), offsets.end(), gid) - offsets.begin() - 1);
          const auto& shard = entry.shards.at(rank);
          const auto local = static_cast<std::size_t>(gid - offsets[rank]);
          add(32);
          add(entry.codec->archive_transport_size(
              *shard.image, shard.array->values.at(local), source.element,
              [&](std::uint64_t bytes) { budget_.charge(bytes); },
              [&](distributed::Identity id) -> std::span<const std::int64_t> {
                return domains_.at(key(id)).target->partition_offsets();
              }));
        }
        if (registry.retained_values() >= registry.storage_versions() ||
            registry.retained_bytes() > registry.storage_bytes() ||
            charge > registry.storage_bytes() - registry.retained_bytes())
          throw std::length_error("target Array cannot fit the authenticated registry budget");
        budget_.charge(charge);
      });
      distributed::phase(lane, [&] {
        budget_.charge(sizeof(ContinuationDistributedArray<Dim>));
        budget_.charge(domain->local_size(), sizeof(Ref));
        next = std::make_shared<ContinuationDistributedArray<Dim>>();
        next->domain = domain;
        next->element = source.element;
        if (source.index_domain)
          next->index_domain =
              domains_.at(key(image.domains[*source.index_domain].identity)).target;
        next->values.reserve(static_cast<std::size_t>(domain->local_size()));
        const auto& offsets = image.domains[source.domain].offsets;
        for (std::int64_t gid = domain->begin(); gid < domain->end(); ++gid) {
          const auto upper = std::upper_bound(offsets.begin(), offsets.end(), gid);
          const auto rank = static_cast<std::size_t>(upper - offsets.begin() - 1);
          const auto& shard = entry.shards.at(rank);
          const auto local = static_cast<std::size_t>(gid - offsets[rank]);
          next->values.push_back(value(*shard.image, shard.array->values.at(local)));
        }
      });
      entry.target =
          distributed::publish_array<Dim>(registry, std::move(next), entry.capacity, &transport);
    }
    distributed::phase(lane, [&] { materialized_ = true; });
  }

  Ref value(const Image& image, std::size_t root) {
    auto& cache = caches_.at(&image).values;
    std::vector<std::pair<std::size_t, bool>> pending;
    budget_.charge(sizeof(decltype(pending)::value_type));
    pending.emplace_back(root, false);
    while (!pending.empty()) {
      const auto [id, exit] = pending.back();
      pending.pop_back();
      if (cache.at(id))
        continue;
      const auto& node = image.nodes.at(id);
      if (!exit && !node.children.empty()) {
        budget_.charge(node.children.size() + 1, sizeof(decltype(pending)::value_type));
        pending.emplace_back(id, true);
        for (auto i = node.children.rbegin(); i != node.children.rend(); ++i)
          pending.emplace_back(*i, false);
        continue;
      }
      if (node.leaf) {
        cache[id] = node.leaf;
        continue;
      }
      budget_.charge(sizeof(Value));
      if (node.kind == ContinuationKind::DistributedArray) {
        const auto& array = arrays_.at(key(image.arrays.at(node.slot).identity)).target;
        if (!array)
          throw std::logic_error("distributed migration dependency is not materialized");
        cache[id] = Value::distributed_array(array);
      } else if (continuation_distributed_kind(node.kind)) {
        const auto& domain = domains_.at(key(image.domains.at(node.slot).identity)).target;
        if (!domain)
          throw std::logic_error("distributed migration domain is not materialized");
        cache[id] = node.kind == ContinuationKind::DistributedDomain
                        ? Value::distributed_domain(domain)
                        : Value::distributed_index(domain, node.gid);
      } else {
        budget_.charge(node.children.size(), sizeof(Ref));
        std::vector<Ref> children;
        children.reserve(node.children.size());
        for (const auto child : node.children)
          children.push_back(cache.at(child));
        if (node.kind == ContinuationKind::Product)
          cache[id] = Value::product(std::move(children));
        else if (node.kind == ContinuationKind::Array)
          cache[id] = Value::array(node.finite, std::move(children));
        else if (node.kind == ContinuationKind::Variant)
          cache[id] =
              Value::variant(static_cast<std::uint32_t>(node.arm), std::move(children.at(0)));
        else
          throw std::logic_error("distributed migration has an unrecognized structural value");
      }
    }
    return cache.at(root);
  }
  std::shared_ptr<distributed::Registry> primary(const Image& image) const {
    return image.primary_registry
               ? registries_.at(image.registries.at(*image.primary_registry).issuer).target
               : nullptr;
  }
  const auto& authorities() const noexcept { return target_registries_; }

 private:
  struct Source {
    const Image* image = nullptr;
    const Array* array = nullptr;
  };
  struct ArrayEntry {
    std::vector<Source> shards;
    std::set<Key> dependencies;
    std::uint64_t capacity = 0;
    const Codec* codec = nullptr;
    ContinuationDistributedArrayRef<Dim> target;
  };
  struct DomainEntry {
    const Domain* source;
    distributed::DomainRef target;
  };
  struct RegistryEntry {
    typename Codec::ArchiveRegistry source;
    std::shared_ptr<distributed::Registry> target;
  };
  struct Cache {
    std::vector<Ref> values;
  };
  ContinuationMigrationBudget& budget_;
  int ranks_ = 0;
  bool prepared_ = false, materialized_ = false;
  std::map<Origin, RegistryEntry> registries_;
  std::map<Key, DomainEntry> domains_;
  std::map<Key, ArrayEntry> arrays_;
  std::map<const Image*, Cache> caches_;
  std::vector<Key> order_;
  std::vector<std::shared_ptr<distributed::Registry>> target_registries_;

  void require_same_array_(const Image& a, const Array& x, const Image& b, const Array& y) {
    if (key(a.domains[x.domain].identity) != key(b.domains[y.domain].identity) ||
        x.element != y.element || x.charge != y.charge || x.values.size() != y.values.size() ||
        bool(x.index_domain) != bool(y.index_domain) ||
        (x.index_domain &&
         key(a.domains[*x.index_domain].identity) != key(b.domains[*y.index_domain].identity)))
      throw std::invalid_argument("distributed migration aliases conflicting immutable versions");
    std::vector<std::pair<std::size_t, std::size_t>> pending;
    budget_.charge(x.values.size(), sizeof(decltype(pending)::value_type));
    for (std::size_t i = 0; i < x.values.size(); ++i)
      pending.emplace_back(x.values[i], y.values[i]);
    while (!pending.empty()) {
      const auto [ai, bi] = pending.back();
      pending.pop_back();
      const auto& an = a.nodes.at(ai);
      const auto& bn = b.nodes.at(bi);
      if (an.kind != bn.kind || an.arm != bn.arm || an.gid != bn.gid ||
          an.children.size() != bn.children.size() || an.finite != bn.finite)
        throw std::invalid_argument("distributed migration immutable value contents conflict");
      if (continuation_distributed_kind(an.kind)) {
        const auto ak = an.kind == ContinuationKind::DistributedArray
                            ? key(a.arrays.at(an.slot).identity)
                            : key(a.domains.at(an.slot).identity);
        const auto bk = bn.kind == ContinuationKind::DistributedArray
                            ? key(b.arrays.at(bn.slot).identity)
                            : key(b.domains.at(bn.slot).identity);
        if (ak != bk)
          throw std::invalid_argument("distributed migration immutable reference conflicts");
      } else if (an.leaf) {
        bool same = bool(bn.leaf) && an.leaf->domain_ref() == bn.leaf->domain_ref();
        if (same)
          switch (an.kind) {
            case ContinuationKind::Boolean:
              same = an.leaf->boolean() == bn.leaf->boolean();
              break;
            case ContinuationKind::Int64:
              same = an.leaf->integer() == bn.leaf->integer();
              break;
            case ContinuationKind::Float64:
              same = std::bit_cast<std::uint64_t>(an.leaf->real()) ==
                     std::bit_cast<std::uint64_t>(bn.leaf->real());
              break;
            case ContinuationKind::Index:
              same = an.leaf->index_offset() == bn.leaf->index_offset();
              break;
            case ContinuationKind::Domain:
              break;
            default:
              same = false;
          }
        if (!same)
          throw std::invalid_argument("distributed migration immutable leaf conflicts");
      }
      budget_.charge(an.children.size(), sizeof(decltype(pending)::value_type));
      for (std::size_t i = 0; i < an.children.size(); ++i)
        pending.emplace_back(an.children[i], bn.children[i]);
    }
  }
  void require_reachability_(std::span<const Input> inputs) {
    std::set<Key> reachable_arrays, reachable_domains;
    std::vector<Key> pending;
    const auto inspect = [&](const Image& image, std::size_t root) {
      Codec::walk_archive(
          image, root,
          [&](const auto& node) {
            if (!continuation_distributed_kind(node.kind))
              return;
            const auto& domain = image.domains.at(node.kind == ContinuationKind::DistributedArray
                                                      ? image.arrays.at(node.slot).domain
                                                      : node.slot);
            budget_.charge(sizeof(Key));
            reachable_domains.insert(key(domain.identity));
            if (node.kind == ContinuationKind::DistributedArray) {
              const auto id = key(image.arrays.at(node.slot).identity);
              budget_.charge(2, sizeof(Key));
              if (reachable_arrays.insert(id).second)
                pending.push_back(id);
            }
          },
          [&](std::uint64_t bytes) { budget_.charge(bytes); });
    };
    for (const auto& input : inputs)
      inspect(*input.image, input.image->root);
    while (!pending.empty()) {
      const auto id = pending.back();
      pending.pop_back();
      for (const auto& shard : arrays_.at(id).shards) {
        if (shard.array->index_domain) {
          budget_.charge(sizeof(Key));
          reachable_domains.insert(key(shard.image->domains[*shard.array->index_domain].identity));
        }
        for (const auto node : shard.array->values)
          inspect(*shard.image, node);
      }
    }
    if (reachable_arrays.size() != arrays_.size() || reachable_domains.size() != domains_.size())
      throw std::invalid_argument("distributed migration source tables contain unreachable values");
  }
};
}  // namespace pops::runtime::program
