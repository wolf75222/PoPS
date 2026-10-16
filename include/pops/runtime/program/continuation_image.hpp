#pragma once

#include <pops/mesh/storage/multifab.hpp>
#include <pops/runtime/program/distributed_domain.hpp>
#include <pops/runtime/program/continuation_field_support.hpp>
#include <pops/runtime/program/field_exchange_expression.hpp>
#include <pops/runtime/program/field_representation_obligation.hpp>
#include <pops/runtime/program/structural_traversal.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::runtime::program {

// Both descriptions travel with an opaque Field. Neither description by itself
// grants permission to read its storage in a consuming execution context.
struct ContinuationFieldMetadata {
  FieldExchangeExpressions::Ref expression;
  ContinuationFieldSupport::Ref support;
  FieldRepresentationObligation::Ref representation;
};

// Structural language types only: no physical-model or numerical-method dispatch.
enum class ContinuationKind : std::uint8_t {
  Boolean,
  Int64,
  Float64,
  Product,
  Variant,
  Domain,
  Index,
  Array,
  Field,
  DistributedDomain,
  DistributedIndex,
  DistributedArray
};

#include <pops/runtime/program/continuation_transport_wire.inc>

inline bool continuation_domain_kind(ContinuationKind kind) noexcept {
  return kind == ContinuationKind::Domain || kind == ContinuationKind::DistributedDomain;
}
inline bool continuation_index_kind(ContinuationKind kind) noexcept {
  return kind == ContinuationKind::Index || kind == ContinuationKind::DistributedIndex;
}
inline bool continuation_array_kind(ContinuationKind kind) noexcept {
  return kind == ContinuationKind::Array || kind == ContinuationKind::DistributedArray;
}
inline bool continuation_distributed_kind(ContinuationKind kind) noexcept {
  return kind == ContinuationKind::DistributedDomain ||
         kind == ContinuationKind::DistributedIndex || kind == ContinuationKind::DistributedArray;
}

// A path traverses Product/Variant members by ordinal.  The sentinel traverses
// every element of an Array. Domain binders themselves are fixed Product paths.
inline constexpr std::size_t continuation_each = std::numeric_limits<std::size_t>::max();
struct ContinuationDomainBinding {
  std::vector<std::size_t> value_path;
  std::vector<std::size_t> domain_path;
  bool operator==(const ContinuationDomainBinding&) const = default;
};

struct ContinuationType {
  ContinuationKind kind = ContinuationKind::Product;
  std::vector<ContinuationType> children;
  std::vector<std::string> names;
  std::int64_t family = -1;
  std::int64_t capacity = 0;
  int field_owner = -1;
  int field_components = 0;
  std::string scientific_contract;
  std::vector<ContinuationDomainBinding> domain_bindings;
  bool dependent_bundle = false;
  std::string field_provider;
  ContinuationType(ContinuationKind selected = ContinuationKind::Product,
                   std::vector<ContinuationType> members = {}, std::vector<std::string> labels = {},
                   std::int64_t selected_family = -1, std::int64_t selected_capacity = 0,
                   int owner = -1, int components = 0, std::string contract = {},
                   std::vector<ContinuationDomainBinding> bindings = {}, bool dependent = false,
                   std::string provider = {})
      : kind(selected),
        children(std::move(members)),
        names(std::move(labels)),
        family(selected_family),
        capacity(selected_capacity),
        field_owner(owner),
        field_components(components),
        scientific_contract(std::move(contract)),
        domain_bindings(std::move(bindings)),
        dependent_bundle(dependent),
        field_provider(std::move(provider)) {}

  ContinuationType(ContinuationType&&) noexcept = default;
  ContinuationType& operator=(ContinuationType&&) noexcept = default;
  ContinuationType(const ContinuationType& other) {
    struct Pair {
      const ContinuationType* source;
      ContinuationType* destination;
    };
    std::vector<Pair> pending{{&other, this}};
    while (!pending.empty()) {
      const auto [source, destination] = pending.back();
      pending.pop_back();
      destination->kind = source->kind;
      destination->names = source->names;
      destination->family = source->family;
      destination->capacity = source->capacity;
      destination->field_owner = source->field_owner;
      destination->field_components = source->field_components;
      destination->scientific_contract = source->scientific_contract;
      destination->domain_bindings = source->domain_bindings;
      destination->dependent_bundle = source->dependent_bundle;
      destination->field_provider = source->field_provider;
      destination->children.resize(source->children.size());
      for (std::size_t i = source->children.size(); i != 0; --i)
        pending.push_back({&source->children[i - 1], &destination->children[i - 1]});
    }
  }
  ContinuationType& operator=(const ContinuationType& other) {
    if (this != &other) {
      ContinuationType copy(other);
      *this = std::move(copy);
    }
    return *this;
  }
  ~ContinuationType() noexcept {
    // Children are uniquely owned values. Drain their vectors from leaves upward
    // without allocation; pop_back then invokes only an already empty destructor.
    auto* node = this;
    for (;;) {
      if (!node->children.empty()) {
        auto* next = &node->children.back();
        next->retirement_parent_ = node;
        node = next;
      } else {
        if (node == this)
          break;
        auto* parent = node->retirement_parent_;
        parent->children.pop_back();
        node = parent;
      }
    }
  }
  bool operator==(const ContinuationType& other) const {
    using Pair = std::pair<const ContinuationType*, const ContinuationType*>;
    std::vector<Pair> pending{{this, &other}};
    while (!pending.empty()) {
      const auto [left, right] = pending.back();
      pending.pop_back();
      if (left->kind != right->kind || left->names != right->names ||
          left->family != right->family || left->capacity != right->capacity ||
          left->field_owner != right->field_owner ||
          left->field_components != right->field_components ||
          left->scientific_contract != right->scientific_contract ||
          left->domain_bindings != right->domain_bindings ||
          left->dependent_bundle != right->dependent_bundle ||
          left->field_provider != right->field_provider ||
          left->children.size() != right->children.size())
        return false;
      for (std::size_t i = left->children.size(); i != 0; --i)
        pending.emplace_back(&left->children[i - 1], &right->children[i - 1]);
    }
    return true;
  }

 private:
  ContinuationType* retirement_parent_ = nullptr;

 public:
  const ContinuationType& dependent_at(const std::vector<std::size_t>& path,
                                       bool binder = false) const {
    const auto* value = this;
    for (auto item : path) {
      if (!binder && continuation_array_kind(value->kind) && item == continuation_each) {
        if (value->children.size() != 1)
          throw std::invalid_argument("invalid Bundle array path");
        value = &value->children[0];
      } else if ((value->kind == ContinuationKind::Product ||
                  (!binder && value->kind == ContinuationKind::Variant)) &&
                 item < value->children.size()) {
        value = &value->children[item];
      } else
        throw std::invalid_argument("Bundle dependency path changes structural type");
    }
    return *value;
  }

  void validate_dependencies(continuation_detail::StructuralBudget& budget) const {
    if (!dependent_bundle) {
      if (!domain_bindings.empty())
        throw std::invalid_argument("domain bindings require a dependent Bundle");
      return;
    }
    if (kind != ContinuationKind::Product ||
        names != std::vector<std::string>{"instance", "fields", "geometry", "maps", "histories"} ||
        children.size() != 5 || !continuation_domain_kind(children[0].kind))
      throw std::invalid_argument("Bundle must retain its complete five-role schema");
    for (std::size_t i = 1; i < children.size(); ++i)
      if (children[i].kind != ContinuationKind::Product)
        throw std::invalid_argument("Bundle data roles must be Products");
    std::vector<std::vector<std::size_t>> required, supplied;
    struct Pending {
      const ContinuationType* type;
      std::vector<std::size_t> path;
      bool root;
    };
    std::vector<Pending> pending{{this, {}, true}};
    while (!pending.empty()) {
      auto frame = std::move(pending.back());
      pending.pop_back();
      budget.consume();
      budget.consume(frame.path.size());
      const auto& type = *frame.type;
      if (!frame.root && type.dependent_bundle)
        continue;
      if (continuation_domain_kind(type.kind) || continuation_index_kind(type.kind) ||
          continuation_array_kind(type.kind))
        required.push_back(frame.path);
      if (type.kind == ContinuationKind::Field)
        throw std::invalid_argument("legacy Field is not a finite Bundle representation");
      for (std::size_t i = type.children.size(); i != 0; --i) {
        auto path = frame.path;
        path.push_back(continuation_array_kind(type.kind) ? continuation_each : i - 1);
        budget.frames(pending.size() + 1, sizeof(Pending));
        pending.push_back({&type.children[i - 1], std::move(path), false});
      }
    }
    for (const auto& binding : domain_bindings) {
      const auto& dependent = dependent_at(binding.value_path);
      const auto& domain = dependent_at(binding.domain_path, true);
      if (!continuation_domain_kind(domain.kind) ||
          (!continuation_domain_kind(dependent.kind) && !continuation_index_kind(dependent.kind) &&
           !continuation_array_kind(dependent.kind)) ||
          dependent.family != domain.family || dependent.capacity != domain.capacity ||
          continuation_distributed_kind(dependent.kind) !=
              continuation_distributed_kind(domain.kind) ||
          (continuation_domain_kind(dependent.kind) && binding.value_path != binding.domain_path))
        throw std::invalid_argument("Bundle binding changes its domain family or bounds");
      supplied.push_back(binding.value_path);
    }
    std::sort(required.begin(), required.end());
    std::sort(supplied.begin(), supplied.end());
    if (required != supplied)
      throw std::invalid_argument("Bundle does not bind every dependent value exactly once");
  }

  void validate_node_() const {
    if (static_cast<unsigned>(kind) > static_cast<unsigned>(ContinuationKind::DistributedArray))
      throw std::invalid_argument("continuation schema has an unknown kind");
    const bool record = kind == ContinuationKind::Product || kind == ContinuationKind::Variant;
    if ((record && (names.size() != children.size() ||
                    (kind == ContinuationKind::Variant && names.empty()))) ||
        (!record && !names.empty()) || (continuation_array_kind(kind) && children.size() != 1) ||
        (!record && !continuation_array_kind(kind) && !children.empty()))
      throw std::invalid_argument("continuation schema has inconsistent structural children");
    for (std::size_t i = 0; i < names.size(); ++i)
      if (names[i].empty() ||
          std::find(names.begin(), names.begin() + i, names[i]) != names.begin() + i)
        throw std::invalid_argument("continuation schema member names are empty or duplicated");
    if ((continuation_domain_kind(kind) || continuation_index_kind(kind) ||
         continuation_array_kind(kind)) &&
        (family < 0 || capacity < 0))
      throw std::invalid_argument("continuation domain schema lacks bounded family authority");
    if (kind == ContinuationKind::Field && (!((field_owner >= 0 && field_provider.empty()) ||
                                              (field_owner == -1 && !field_provider.empty())) ||
                                            field_components < 1 || scientific_contract.empty()))
      throw std::invalid_argument(
          "continuation field schema lacks exact owner/components/scientific authority");
  }
  void validate(std::uint64_t capacity = std::numeric_limits<std::uint64_t>::max()) const {
    continuation_detail::StructuralBudget budget(capacity);
    continuation_detail::walk_structure(
        this, [](const ContinuationType& node) { return node.children.size(); },
        [](const ContinuationType& node, std::size_t i) { return &node.children[i]; },
        [](const ContinuationType& node) {
          node.validate_node_();
          return true;
        },
        [&](const ContinuationType& node) { node.validate_dependencies(budget); }, budget);
  }
};
inline std::string continuation_transport_type_identity(
    const ContinuationType&, bool = false,
    std::uint64_t = std::numeric_limits<std::uint64_t>::max());

template <int Dim>
class ContinuationCodec;
template <int Dim>
class ContinuationTransportSession;
class FiniteDomainRegistry;
class FiniteDomainRealm {
 public:
  bool visible() const noexcept { return live_.load(std::memory_order_acquire); }
  std::array<std::uint64_t, 2> transport_identity() const {
    if (!visible())
      throw std::logic_error("cannot transport a revoked finite realm");
    exported_ = true;
    return identity_;
  }
  const std::array<std::uint64_t, 2>& unexported_identity() const noexcept { return identity_; }
  bool replicated() const noexcept { return replicated_; }
  bool synchronized() const noexcept { return synchronized_; }

 private:
  mutable std::atomic<bool> live_{true};
  mutable bool exported_ = false;
  mutable bool replicated_ = false;
  mutable bool synchronized_ = false;
  mutable std::array<std::uint64_t, 2> identity_{};
  void revoke() const noexcept { live_.store(false, std::memory_order_release); }
  friend class FiniteDomainRegistry;
  template <int>
  friend class ContinuationCodec;
  template <int>
  friend class ContinuationTransportSession;
  FiniteDomainRealm() {
    std::random_device random;
    for (auto& word : identity_)
      word = (std::uint64_t(random()) << 32) ^ std::uint64_t(random());
  }
  explicit FiniteDomainRealm(std::array<std::uint64_t, 2> identity, bool replicated)
      : exported_(true), replicated_(replicated), synchronized_(true), identity_(identity) {}
};
using FiniteDomainRealmRef = std::shared_ptr<const FiniteDomainRealm>;
class FiniteDomain {
 public:
  const std::int64_t size, capacity, family, version;
  const FiniteDomainRealmRef& realm() const noexcept { return realm_; }
  bool visible() const noexcept {
    return live_.load(std::memory_order_acquire) && realm_ && realm_->visible();
  }
  void require_live() const {
    if (!visible())
      throw std::logic_error("finite domain belongs to a revoked runtime realm");
  }

 private:
  friend class FiniteDomainRegistry;
  template <int>
  friend class ContinuationCodec;
  template <int>
  friend class ContinuationTransportSession;
  FiniteDomain(std::int64_t count, std::int64_t cap, std::int64_t fam, std::int64_t ordinal,
               FiniteDomainRealmRef realm)
      : size(count), capacity(cap), family(fam), version(ordinal), realm_(std::move(realm)) {}
  FiniteDomainRealmRef realm_;
  mutable std::atomic<bool> live_{true};
  void revoke() const noexcept { live_.store(false, std::memory_order_release); }
};
using FiniteDomainRef = std::shared_ptr<const FiniteDomain>;

// Live allocator: the ordinal is burned even if the enclosing mathematical trial rolls back.
// The realm is native possession, never reconstructed from an integer or pointer in an archive.
class FiniteDomainRegistry {
 public:
  explicit FiniteDomainRegistry(bool replicated = true)
      : realm_(new FiniteDomainRealm), replicated_(replicated) {
    realm_->replicated_ = replicated_;
  }
  ~FiniteDomainRegistry() noexcept { revoke_on_restart(); }
  FiniteDomainRegistry(const FiniteDomainRegistry&) = delete;
  FiniteDomainRegistry& operator=(const FiniteDomainRegistry&) = delete;
  const FiniteDomainRealmRef& realm() const noexcept { return realm_; }
  // The caller must agree these words on its execution lane before this local operation.
  // A realm already exported by transport can never silently change its identity.
  void synchronize_identity(const FiniteDomainRealmRef& realm,
                            std::array<std::uint64_t, 2> identity) {
    if (!realm || !realm->visible() || (realm->exported_ && realm->identity_ != identity))
      throw std::logic_error("finite realm identity cannot be rebound after transport");
    realm->identity_ = identity;
    realm->replicated_ = true;
    realm->synchronized_ = true;
  }
  // A decoded realm is adopted while allocation failure is still rollbackable. The current
  // realm is also retained separately, so successful finalization can retain a newly decoded
  // realm without allocating a tracking entry.
  void adopt(const FiniteDomainRealmRef& realm) {
    if (!realm || !realm->visible())
      throw std::logic_error("cannot adopt a revoked finite realm");
    if (realm == realm_)
      return;
    for (const auto& known : adopted_)
      if (known.lock() == realm)
        return;
    adopted_.emplace_back(realm);
  }
  void select_realm(const FiniteDomainRealmRef& realm) {
    if (!realm || !realm->visible() || realm->replicated() != replicated_)
      throw std::logic_error("finite issuance realm has foreign liveness or distribution");
    if (realm == realm_)
      return;
    adopt(realm);
    if (realm_)
      adopted_.emplace_back(realm_);
    realm_ = realm;
  }
  struct PreparedAdoption {
    FiniteDomainRealmRef realm;
    std::vector<std::weak_ptr<const FiniteDomainRealm>> realms;
    std::vector<std::weak_ptr<const FiniteDomain>> domains;
    std::int64_t high_water = 0;
  };
  // All allocations precede accepted-image publication. Preparing this value
  // neither changes issuer authority nor revokes domains in another store.
  PreparedAdoption prepare_adoption(std::span<const FiniteDomainRef> domains,
                                    FiniteDomainRealmRef selected = {}) const {
    PreparedAdoption out{selected ? selected : realm_, adopted_, issued_, next_};
    const auto add_realm = [&](const FiniteDomainRealmRef& realm) {
      if (!realm || !realm->visible() || realm->replicated() != replicated_)
        throw std::logic_error("finite adoption has unavailable or foreign realm");
      if (std::none_of(out.realms.begin(), out.realms.end(),
                       [&](const auto& weak) { return weak.lock() == realm; }))
        out.realms.emplace_back(realm);
    };
    if (selected)
      add_realm(selected);
    if (realm_ && realm_->visible())
      add_realm(realm_);
    for (const auto& domain : domains) {
      if (!domain || !domain->visible())
        throw std::logic_error("cannot prepare adoption of an unavailable domain");
      add_realm(domain->realm());
      out.high_water = std::max(out.high_water, domain->version);
      if (std::none_of(out.domains.begin(), out.domains.end(),
                       [&](const auto& weak) { return weak.lock() == domain; }))
        out.domains.emplace_back(domain);
    }
    return out;
  }
  void install_adoption(PreparedAdoption&& prepared) noexcept {
    realm_.swap(prepared.realm);
    adopted_.swap(prepared.realms);
    issued_.swap(prepared.domains);
    next_ = std::max(next_, prepared.high_water);
  }
  // Adopt only domains decoded or selected by the accepted-image authority.
  // Imported ordinals are burned before later issuance, just like native ones.
  void adopt_domains(std::span<const FiniteDomainRef> domains) {
    for (const auto& domain : domains) {
      if (!domain || !domain->visible())
        throw std::logic_error("cannot adopt an unavailable finite domain");
      import_high_water(domain->version);
      adopt(domain->realm());
      bool known = false;
      for (const auto& weak : issued_)
        known = known || weak.lock() == domain;
      if (!known)
        issued_.emplace_back(domain);
    }
  }
  // The keep set is supplied by an authenticated accepted-image owner, never
  // inferred from arbitrary algebra values. Counters remain outside rollback.
  void revoke_unretained(std::span<const FiniteDomainRef> keep = {}) noexcept {
    for (const auto& weak : issued_)
      if (auto domain = weak.lock()) {
        bool retained = false;
        for (const auto& accepted : keep)
          retained = retained || domain == accepted;
        if (!retained)
          domain->revoke();
      }
    std::erase_if(issued_, [](const auto& weak) {
      const auto domain = weak.lock();
      return !domain || !domain->visible();
    });
  }
  void revoke_on_restart(FiniteDomainRealmRef keep = {}) noexcept {
    revoke_unretained();
    if (realm_ && realm_ != keep)
      realm_->revoke();
    for (const auto& known : adopted_)
      if (auto realm = known.lock(); realm && realm != keep)
        realm->revoke();
    adopted_.clear();
    realm_ = std::move(keep);  // shared_ptr assignment is nonallocating
  }
  std::int64_t next_version() {
    if (next_ == std::numeric_limits<std::int64_t>::max())
      throw std::overflow_error("native finite-domain version exhausted");
    return ++next_;
  }
  void import_high_water(std::int64_t version) {
    if (version < 0)
      throw std::invalid_argument("invalid finite-domain high-water mark");
    next_ = std::max(next_, version);
  }
  FiniteDomainRef issue(std::int64_t count, std::int64_t capacity, std::int64_t family,
                        FiniteDomainRealmRef realm = {}) {
    const auto version = next_version();
    if (count < 0 || capacity < count || family < 0)
      throw std::invalid_argument("invalid native finite-domain bounds or family");
    if (realm)
      adopt(realm);
    else {
      if (!realm_ || !realm_->visible()) {
        realm_ = FiniteDomainRealmRef(new FiniteDomainRealm);
        realm_->replicated_ = replicated_;
      }
      realm = realm_;
    }
    auto domain =
        FiniteDomainRef(new FiniteDomain(count, capacity, family, version, std::move(realm)));
    issued_.emplace_back(domain);
    return domain;
  }

 private:
  FiniteDomainRealmRef realm_;
  std::vector<std::weak_ptr<const FiniteDomainRealm>> adopted_;
  std::vector<std::weak_ptr<const FiniteDomain>> issued_;
  std::int64_t next_ = 0;
  bool replicated_ = true;
};

template <int Dim>
class ContinuationValue;
template <int Dim>
using ContinuationValueRef = std::shared_ptr<const ContinuationValue<Dim>>;
template <int Dim>
class ContinuationCodec;
template <int Dim>
struct ContinuationDistributedArray {
  distributed::DomainRef domain;
  ContinuationType element;
  std::vector<ContinuationValueRef<Dim>> values;
  distributed::Identity version{};
  distributed::DomainRef index_domain;  // exact target witness, including empty Index shards

  // An aggregate remains usable for native staging. Copying or moving its
  // opaque proof never grants a mutable object the immutable-publication fact.
  class PublicationProof {
   public:
    PublicationProof() = default;
    PublicationProof(const PublicationProof&) noexcept {}
    PublicationProof& operator=(const PublicationProof&) noexcept {
      scalar_only_ = false;
      return *this;
    }

   private:
    friend struct ContinuationDistributedArray;
    explicit PublicationProof(bool scalar_only) : scalar_only_(scalar_only) {}
    bool scalar_only_ = false;
  } publication_proof;

  bool has_immutable_scalar_values() const noexcept { return publication_proof.scalar_only_; }
  static std::shared_ptr<const ContinuationDistributedArray> immutable_publication(
      const ContinuationDistributedArray& source, distributed::Identity identity,
      std::uint64_t workspace_capacity) {
    // The old vector remains owned by pending. Charge the additional pointer
    // table against the operation workspace before allocating it; shared Value
    // payloads retain their existing registry charge and are not duplicated.
    if (source.values.size() > workspace_capacity / sizeof(ContinuationValueRef<Dim>))
      throw std::length_error("immutable Array publication exceeds its pointer workspace");
    bool scalar_only = true;
    for (const auto& value : source.values)
      scalar_only = scalar_only && value && value->is_scalar_structure();
    // Copy the vector even when source is uniquely owned: native callers can
    // retain references to its elements. Only immutable Value leaves are shared.
    // Direct aggregate initialization elides the proof copy into an actually
    // const object; every subsequent copy/move of that object drops the proof.
    return std::shared_ptr<const ContinuationDistributedArray>(
        new const ContinuationDistributedArray{source.domain, source.element, source.values,
                                               identity, source.index_domain,
                                               PublicationProof{scalar_only}});
  }
};
template <int Dim>
using ContinuationDistributedArrayRef = std::shared_ptr<const ContinuationDistributedArray<Dim>>;
template <int Dim>
std::uint64_t continuation_distributed_array_charge(
    const ContinuationDistributedArrayRef<Dim>&, distributed::Registry&, std::uint64_t,
    std::span<const std::shared_ptr<distributed::Registry>> = {},
    const ContinuationTransportSession<Dim>* = nullptr);
template <int Dim>
struct ContinuationClosure {
  std::vector<distributed::DomainRef> domains;
  std::vector<ContinuationDistributedArrayRef<Dim>> arrays;
  FiniteDomainRealmRef replicated_realm;
  // Explicit native capabilities; issuer bytes alone never authorize a registry.
  std::vector<std::shared_ptr<distributed::Registry>> registries;
};

/// Immutable owned finite value. All construction here is rank-local; the caller converges
/// allocation/copy/fence failures before a subsequent native collective.
template <int Dim>
class ContinuationValue {
 public:
  using Ref = ContinuationValueRef<Dim>;
  using Field = MultiFab<Dim>;
  static Ref boolean(bool value) { return scalar_(ContinuationKind::Boolean, value ? 1 : 0); }
  static Ref integer(std::int64_t value) {
    return scalar_(ContinuationKind::Int64, std::bit_cast<std::uint64_t>(value));
  }
  static Ref real(double value) {
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
    return scalar_(ContinuationKind::Float64, std::bit_cast<std::uint64_t>(value));
  }
  static Ref product(std::vector<Ref> values) {
    auto out = fresh_(ContinuationKind::Product);
    out->scalar_structure_ = std::all_of(values.begin(), values.end(), [](const Ref& value) {
      return value && value->is_scalar_structure();
    });
    out->members_ = std::move(values);
    return out;
  }
  static Ref variant(std::uint32_t arm, Ref payload) {
    auto out = fresh_(ContinuationKind::Variant);
    out->scalar_structure_ = payload && payload->is_scalar_structure();
    out->word_ = arm;
    out->members_.push_back(std::move(payload));
    return out;
  }
  static Ref domain(FiniteDomainRef value) {
    require_domain_(value);
    auto out = fresh_(ContinuationKind::Domain);
    out->domain_ = std::move(value);
    return out;
  }
  static Ref index(FiniteDomainRef value, std::int64_t offset) {
    require_domain_(value);
    auto out = fresh_(ContinuationKind::Index);
    out->domain_ = std::move(value);
    out->word_ = std::bit_cast<std::uint64_t>(offset);
    return out;
  }
  static Ref array(FiniteDomainRef domain, std::vector<Ref> values) {
    require_domain_(domain);
    auto out = fresh_(ContinuationKind::Array);
    out->domain_ = std::move(domain);
    out->members_ = std::move(values);
    return out;
  }
  static Ref distributed_domain(distributed::DomainRef domain) {
    if (!domain)
      throw std::invalid_argument("distributed continuation domain is absent");
    domain->require_live();
    auto out = fresh_(ContinuationKind::DistributedDomain);
    out->distributed_domain_ = std::move(domain);
    return out;
  }
  static Ref distributed_index(distributed::DomainRef domain, std::int64_t gid) {
    if (!domain)
      throw std::invalid_argument("distributed continuation domain is absent");
    domain->require_live();
    if (gid < 0 || gid >= domain->size())
      throw std::invalid_argument("distributed continuation Index is out of bounds");
    auto out = fresh_(ContinuationKind::DistributedIndex);
    out->distributed_domain_ = std::move(domain);
    out->word_ = std::bit_cast<std::uint64_t>(gid);
    return out;
  }
  static Ref distributed_array(ContinuationDistributedArrayRef<Dim> array) {
    if (!array || !array->domain)
      throw std::invalid_argument("distributed continuation Array is absent");
    array->domain->require_live();
    if (array->values.size() != static_cast<std::size_t>(array->domain->local_size()))
      throw std::invalid_argument("distributed continuation Array shard differs from ownership");
    auto out = fresh_(ContinuationKind::DistributedArray);
    out->distributed_domain_ = array->domain;
    out->distributed_array_ = std::move(array);
    return out;
  }
  static Ref field(const Field& value, int owner, std::string contract,
                   FieldExchangeExpressions::Ref expression = {},
                   ContinuationFieldSupport::Ref support = {}, std::string provider = {},
                   FieldRepresentationObligation::Ref representation = {}) {
    require_field_authority_(owner, provider, support);
    if (representation)
      representation->validate();
    // Fab's copy constructor deep-copies Kokkos storage. No scratch borrow survives this call.
    return owned_field_(std::make_shared<const Field>(value), owner, std::move(contract),
                        std::move(expression), std::move(support), std::move(provider),
                        std::move(representation));
  }
  static Ref with_field_exchange_expression(const Ref& value,
                                            FieldExchangeExpressions::Ref expression) {
    if (!value)
      throw std::invalid_argument("field expression requires an owning continuation value");
    value->require_(ContinuationKind::Field);
    return owned_field_(value->field_, value->field_owner_, value->contract_, std::move(expression),
                        value->field_support_, value->field_provider_,
                        value->field_representation_);
  }
  static Ref with_field_metadata(const Ref& value, ContinuationFieldMetadata metadata) {
    if (!value)
      throw std::invalid_argument("field metadata requires an owning continuation value");
    value->require_(ContinuationKind::Field);
    return owned_field_(value->field_, value->field_owner_, value->contract_,
                        std::move(metadata.expression), std::move(metadata.support),
                        value->field_provider_, std::move(metadata.representation));
  }
  static Ref with_field_representation(const Ref& value,
                                       FieldRepresentationObligation::Ref representation) {
    if (!value)
      throw std::invalid_argument("field representation requires an owning continuation value");
    value->require_(ContinuationKind::Field);
    return owned_field_(value->field_, value->field_owner_, value->contract_,
                        value->field_exchange_expression_, value->field_support_,
                        value->field_provider_, std::move(representation));
  }
  ContinuationKind kind() const noexcept { return kind_; }
  // Closed immutable grammar only: scalars, Products and the selected Variant
  // payload. Domains, Arrays, Fields and future kinds remain conservative so
  // pruning cannot erase a lifetime check or a distributed ownership edge.
  bool is_scalar_structure() const noexcept { return scalar_structure_; }
  const distributed::DomainRef& distributed_domain_ref() const {
    if (distributed_domain_)
      distributed_domain_->require_live();
    return distributed_domain_;
  }
  const ContinuationDistributedArrayRef<Dim>& distributed_array_ref() const {
    if (distributed_domain_)
      distributed_domain_->require_live();
    return distributed_array_;
  }
  std::int64_t distributed_index_offset() const {
    require_(ContinuationKind::DistributedIndex);
    distributed_domain_->require_live();
    return std::bit_cast<std::int64_t>(word_);
  }
  bool boolean() const {
    require_(ContinuationKind::Boolean);
    return word_ != 0;
  }
  std::int64_t integer() const {
    require_(ContinuationKind::Int64);
    return std::bit_cast<std::int64_t>(word_);
  }
  double real() const {
    require_(ContinuationKind::Float64);
    return std::bit_cast<double>(word_);
  }
  const std::vector<Ref>& members() const {
    if (kind_ == ContinuationKind::Array)
      require_domain_(domain_);
    if (kind_ == ContinuationKind::DistributedArray) {
      distributed_domain_->require_live();
      return distributed_array_->values;
    }
    return members_;
  }
  std::uint32_t arm() const {
    require_(ContinuationKind::Variant);
    return static_cast<std::uint32_t>(word_);
  }
  const FiniteDomainRef& domain_ref() const {
    if (domain_)
      domain_->require_live();
    return domain_;
  }
  std::int64_t index_offset() const {
    require_(ContinuationKind::Index);
    require_domain_(domain_);
    return std::bit_cast<std::int64_t>(word_);
  }
  const Field& field() const {
    require_(ContinuationKind::Field);
    return *field_;
  }
  const std::shared_ptr<const Field>& field_storage_owner() const {
    require_(ContinuationKind::Field);
    return field_;
  }
  int field_owner() const noexcept { return field_owner_; }
  const std::string& field_provider() const noexcept { return field_provider_; }
  const FieldRepresentationObligation::Ref& field_representation() const noexcept {
    return field_representation_;
  }
  const FieldExchangeExpressions::Ref& field_exchange_expression() const noexcept {
    return field_exchange_expression_;
  }
  const ContinuationFieldSupport::Ref& field_support() const noexcept { return field_support_; }
  ContinuationFieldMetadata field_metadata() const noexcept {
    return {field_exchange_expression_, field_support_, field_representation_};
  }
  const std::string& scientific_contract() const noexcept { return contract_; }

 private:
  friend class ContinuationCodec<Dim>;
  friend class ContinuationTransportSession<Dim>;
  ContinuationKind kind_;
  std::uint64_t word_ = 0;
  std::vector<Ref> members_;
  distributed::DomainRef distributed_domain_;
  ContinuationDistributedArrayRef<Dim> distributed_array_;
  FiniteDomainRef domain_;
  std::shared_ptr<const Field> field_;
  FieldExchangeExpressions::Ref field_exchange_expression_;
  ContinuationFieldSupport::Ref field_support_;
  int field_owner_ = -1;
  std::string field_provider_;
  FieldRepresentationObligation::Ref field_representation_;
  std::string contract_;
  ContinuationValue* retirement_next_ = nullptr;
  bool scalar_structure_ = false;
  explicit ContinuationValue(ContinuationKind kind) : kind_(kind) {}
  static std::shared_ptr<ContinuationValue> fresh_(ContinuationKind kind) {
    return std::shared_ptr<ContinuationValue>(new ContinuationValue(kind), &retire_);
  }
  static void retire_(ContinuationValue* value) noexcept {
    static thread_local ContinuationValue* pending = nullptr;
    static thread_local bool draining = false;
    value->retirement_next_ = pending;
    pending = value;
    if (draining)
      return;
    draining = true;
    while (pending) {
      auto* retired = pending;
      pending = retired->retirement_next_;
      delete retired;  // Child deleters enqueue; they never recurse through the DAG.
    }
    draining = false;
  }
  static Ref scalar_(ContinuationKind kind, std::uint64_t word) {
    auto out = fresh_(kind);
    out->scalar_structure_ = kind == ContinuationKind::Boolean || kind == ContinuationKind::Int64 ||
                             kind == ContinuationKind::Float64;
    out->word_ = word;
    return out;
  }
  static Ref owned_field_(std::shared_ptr<const Field> field, int owner, std::string contract,
                          FieldExchangeExpressions::Ref expression = {},
                          ContinuationFieldSupport::Ref support = {}, std::string provider = {},
                          FieldRepresentationObligation::Ref representation = {}) {
    require_field_authority_(owner, provider, support);
    if (representation)
      representation->validate();
    auto out = fresh_(ContinuationKind::Field);
    out->field_ = std::move(field);
    out->field_owner_ = owner;
    out->field_provider_ = std::move(provider);
    out->field_representation_ = std::move(representation);
    out->contract_ = std::move(contract);
    out->field_exchange_expression_ = std::move(expression);
    out->field_support_ = std::move(support);
    return out;
  }
  static void require_field_authority_(int owner, const std::string& provider,
                                       const ContinuationFieldSupport::Ref& support) {
    if (!((owner >= 0 && provider.empty()) || (owner == -1 && !provider.empty())))
      throw std::invalid_argument("Field requires exactly one State or provider authority");
    if (support) {
      support->validate();
      if (support->program_owner != owner || support->field_provider != provider)
        throw std::invalid_argument("Field producer support changes its installed owner/provider");
    }
  }
  static void require_domain_(const FiniteDomainRef& domain) {
    if (!domain)
      throw std::invalid_argument("finite value has no domain authority");
    domain->require_live();
  }
  void require_(ContinuationKind kind) const {
    if (kind_ != kind)
      throw std::invalid_argument("continuation value projection changes its structural type");
  }
};

namespace continuation_detail {
// Halo width and component count belong to the value's native storage. The
// owner authenticates its geometry/distribution; the schema authenticates components.
// Validate native index arithmetic and the complete allocation size before any Fab allocation.
template <int Dim>
std::uint64_t field_storage_elements(const Box<Dim>& box, const Extent<Dim>& ghosts, int components,
                                     std::uint64_t limit) {
  if (components <= 0)
    throw std::invalid_argument("continuation Field has no components");
  std::uint64_t elements = static_cast<std::uint64_t>(components);
  for (int d = 0; d < Dim; ++d) {
    const auto low = std::int64_t(box.lo[d]) - ghosts[d];
    const auto high = std::int64_t(box.hi[d]) + ghosts[d];
    const auto width = high - low + 1;
    if (ghosts[d] < 0 || low < std::numeric_limits<int>::min() ||
        high > std::numeric_limits<int>::max() || width <= 0 ||
        width > std::numeric_limits<int>::max())
      throw std::invalid_argument("continuation Field grown box exceeds native storage indices");
    const auto extent = static_cast<std::uint64_t>(width);
    if (elements > limit / extent)
      throw std::length_error("continuation Field storage exceeds its bounded payload/workspace");
    elements *= extent;
  }
  return elements;
}

template <int Dim, class Enter, class Exit>
void walk_value(const ContinuationValue<Dim>* root, std::uint64_t capacity, Enter enter,
                Exit exit) {
  using Value = ContinuationValue<Dim>;
  StructuralBudget budget(capacity);
  walk_structure(
      root, [](const Value& value) { return value.members().size(); },
      [](const Value& value, std::size_t i) { return value.members()[i].get(); }, enter, exit,
      budget);
}

template <int Dim, class Enter, class Exit>
void walk_value(const ContinuationValueRef<Dim>& root, std::uint64_t capacity, Enter enter,
                Exit exit) {
  walk_value<Dim>(root.get(), capacity, enter, exit);
}

template <int Dim>
struct DecodeFrame {
  using Value = ContinuationValue<Dim>;
  const ContinuationType* type = nullptr;
  ContinuationValueRef<Dim> ready;
  FiniteDomainRef domain;
  std::uint64_t arm = 0;
  std::size_t count = 0;
  std::vector<ContinuationValueRef<Dim>> values;

  static DecodeFrame leaf(ContinuationValueRef<Dim> value) {
    DecodeFrame frame;
    frame.ready = std::move(value);
    return frame;
  }
  static DecodeFrame container(const ContinuationType& type, FiniteDomainRef domain,
                               std::uint64_t arm, std::size_t count) {
    DecodeFrame frame;
    frame.type = &type;
    frame.domain = std::move(domain);
    frame.arm = arm;
    frame.count = count;
    frame.values.reserve(count);
    return frame;
  }
  ContinuationValueRef<Dim> finish() {
    if (type->kind == ContinuationKind::Product)
      return Value::product(std::move(values));
    if (type->kind == ContinuationKind::Variant)
      return Value::variant(static_cast<std::uint32_t>(arm), std::move(values[0]));
    return Value::array(std::move(domain), std::move(values));
  }
};

template <int Dim, class Header>
ContinuationValueRef<Dim> decode_structure(const ContinuationType& type, std::uint64_t capacity,
                                           Header header) {
  using Frame = DecodeFrame<Dim>;
  StructuralBudget budget(capacity);
  std::vector<Frame> stack;
  auto push = [&](const ContinuationType& schema) {
    budget.consume();
    budget.frames(stack.size() + 1, sizeof(Frame));
    stack.push_back(header(schema));
  };
  push(type);
  for (;;) {
    auto& frame = stack.back();
    if (!frame.ready && frame.values.size() != frame.count) {
      const auto& schema = *frame.type;
      const auto selected = schema.kind == ContinuationKind::Product   ? frame.values.size()
                            : schema.kind == ContinuationKind::Variant ? frame.arm
                                                                       : 0;
      const auto* child = &schema.children[selected];
      push(*child);
      continue;
    }
    auto result = frame.ready ? std::move(frame.ready) : frame.finish();
    stack.pop_back();
    if (stack.empty())
      return result;
    stack.back().values.push_back(std::move(result));
  }
}

class Writer {
 public:
  explicit Writer(std::uint64_t capacity) : capacity_(capacity) {}
  std::vector<std::uint8_t> bytes;
  std::uint64_t capacity() const noexcept { return capacity_; }
  void word(std::uint64_t value) {
    room(8);
    for (unsigned n = 0; n < 8; ++n)
      bytes.push_back(static_cast<std::uint8_t>(value >> (8 * n)));
  }
  void text(const std::string& value) {
    word(value.size());
    room(value.size());
    bytes.insert(bytes.end(), value.begin(), value.end());
  }
  void blob(std::span<const std::uint8_t> value) {
    word(value.size());
    room(value.size());
    bytes.insert(bytes.end(), value.begin(), value.end());
  }

 private:
  std::uint64_t capacity_;
  void room(std::uint64_t count) const {
    if (bytes.size() > capacity_ || count > capacity_ - bytes.size())
      throw std::length_error("continuation image exceeds its installed byte capacity");
  }
};
class Reader {
 public:
  explicit Reader(std::span<const std::uint8_t> data) : bytes(data) {}
  std::span<const std::uint8_t> bytes;
  std::size_t at = 0;
  std::uint64_t word() {
    if (bytes.size() - at < 8)
      throw std::invalid_argument("continuation image is truncated");
    std::uint64_t value = 0;
    for (unsigned n = 0; n < 8; ++n)
      value |= std::uint64_t(bytes[at++]) << (8 * n);
    return value;
  }
  std::string text() {
    const auto size = word();
    if (size > bytes.size() - at)
      throw std::invalid_argument("continuation text exceeds its byte image");
    std::string value(reinterpret_cast<const char*>(bytes.data() + at),
                      static_cast<std::size_t>(size));
    at += static_cast<std::size_t>(size);
    return value;
  }
  std::span<const std::uint8_t> blob() {
    const auto size = word();
    if (size > bytes.size() - at)
      throw std::invalid_argument("continuation blob exceeds its byte image");
    const auto value = bytes.subspan(at, static_cast<std::size_t>(size));
    at += static_cast<std::size_t>(size);
    return value;
  }
};
inline void domain(const FiniteDomainRef& value, const ContinuationType& type) {
  if (!value || !value->visible() || value->size < 0 || value->size > value->capacity ||
      value->capacity != type.capacity || value->family != type.family || value->version <= 0)
    throw std::invalid_argument("continuation domain differs from its bounded installed family");
}
}  // namespace continuation_detail

/// Structural local codec. Its prototype provider is the installed native storage authority,
/// not a checkpoint-supplied constructor. First profile is exact-layout/rank restart.
template <int Dim>
class ContinuationCodec {
 public:
  using Value = ContinuationValue<Dim>;
  using Ref = typename Value::Ref;
  using Field = MultiFab<Dim>;
  class Prototypes {
   public:
    Prototypes() = default;
    template <class Provider>
    Prototypes(Provider provider) {
      if constexpr (std::is_invocable_r_v<const Field&, Provider, const ContinuationType&,
                                          const ContinuationFieldSupport::Ref&>) {
        provider_ = std::move(provider);
      } else if constexpr (std::is_invocable_r_v<const Field&, Provider, int,
                                                 const ContinuationFieldSupport::Ref&>) {
        provider_ = [provider = std::move(provider)](
                        const ContinuationType& type,
                        const ContinuationFieldSupport::Ref& support) -> const Field& {
          require_state_owner_(type);
          return provider(type.field_owner, support);
        };
      } else {
        static_assert(std::is_invocable_r_v<const Field&, Provider, int>);
        provider_ = [provider = std::move(provider)](
                        const ContinuationType& type,
                        const ContinuationFieldSupport::Ref& support) -> const Field& {
          require_state_owner_(type);
          if (support)
            throw std::invalid_argument(
                "Field producer support requires an exact prototype provider");
          return provider(type.field_owner);
        };
      }
    }
    explicit operator bool() const noexcept { return bool(provider_); }
    const Field& operator()(const ContinuationType& type,
                            const ContinuationFieldSupport::Ref& support = {}) const {
      if (support) {
        support->validate();
        if (support->program_owner != type.field_owner ||
            support->field_provider != type.field_provider)
          throw std::invalid_argument("Field support changes its exact prototype authority");
      }
      return provider_(type, support);
    }
    const Field& operator()(int owner, const ContinuationFieldSupport::Ref& support = {}) const {
      ContinuationType type(ContinuationKind::Field);
      type.field_owner = owner;
      return (*this)(type, support);
    }

   private:
    static void require_state_owner_(const ContinuationType& type) {
      if (type.field_owner < 0 || !type.field_provider.empty())
        throw std::invalid_argument("independent Field requires an exact provider prototype");
    }
    std::function<const Field&(const ContinuationType&, const ContinuationFieldSupport::Ref&)>
        provider_;
  };

  // Installed archive migration may reconstruct the old native distribution
  // after the complete source-rank table has been authenticated. The ordinary
  // exact-layout decoder leaves this empty. The resolver must enforce the
  // source and target storage authority; decode_ still checks the exact
  // resulting layout and all payload framing below.
  class ArchiveFieldResolver {
   public:
    ArchiveFieldResolver() = default;
    template <class Provider>
    ArchiveFieldResolver(Provider provider) {
      if constexpr (std::is_invocable_r_v<const Field&, Provider, const ContinuationType&,
                                          std::span<const std::uint8_t>,
                                          const ContinuationFieldSupport::Ref&>)
        provider_ = std::move(provider);
      else {
        static_assert(std::is_invocable_r_v<const Field&, Provider, const ContinuationType&,
                                            std::span<const std::uint8_t>>);
        provider_ = [provider = std::move(provider)](
                        const ContinuationType& type, std::span<const std::uint8_t> bytes,
                        const ContinuationFieldSupport::Ref& support) -> const Field& {
          if (support)
            throw std::invalid_argument(
                "archive Field producer support requires an exact resolver");
          return provider(type, bytes);
        };
      }
    }
    explicit operator bool() const noexcept { return bool(provider_); }
    const Field& operator()(const ContinuationType& type, std::span<const std::uint8_t> bytes,
                            const ContinuationFieldSupport::Ref& support = {}) const {
      return provider_(type, bytes, support);
    }

   private:
    std::function<const Field&(const ContinuationType&, std::span<const std::uint8_t>,
                               const ContinuationFieldSupport::Ref&)>
        provider_;
  };

  struct Decoded {
    Ref root;
    std::uint64_t publication = 0;
    std::int64_t max_domain_version = 0;
    FiniteDomainRealmRef realm;
    std::vector<FiniteDomainRealmRef> realms;
    std::shared_ptr<distributed::Registry> distributed_registry;
    std::shared_ptr<const ContinuationClosure<Dim>> closure;
  };

#include <pops/runtime/program/continuation_decode_context_declarations.inc>
#include <pops/runtime/program/continuation_archive_codec.inc>

  ContinuationCodec(ContinuationType type, std::string identity, std::uint64_t capacity,
                    Prototypes prototypes, const ExecutionLane* lane = nullptr,
                    const distributed::Registry* registry = nullptr,
                    const ContinuationClosure<Dim>* closure = nullptr,

                    DecodeContext* decode_context = nullptr,
                    ArchiveFieldResolver archive_fields = {},
                    std::uint64_t output_byte_capacity = std::numeric_limits<std::uint64_t>::max())

      : type_(std::move(type)),
        identity_(std::move(identity)),
        capacity_(capacity),

        output_capacity_(std::min(capacity, output_byte_capacity)),

        prototypes_(std::move(prototypes)),
        lane_(lane),
        registry_(registry),
        closure_(closure),

        decode_context_(decode_context),
        archive_fields_(std::move(archive_fields)) {
    type_.validate(capacity_);
    if (identity_.empty() || capacity_ < 32 || !prototypes_)
      throw std::invalid_argument(
          "continuation codec requires installed identity, capacity and native prototypes");
  }
  const ContinuationType& type() const noexcept { return type_; }
  const std::string& identity() const noexcept { return identity_; }
  std::uint64_t capacity() const noexcept { return capacity_; }

  void validate(const Ref& root) const { validate_(type_, root); }
  std::vector<std::uint8_t> encode(const Ref& root, std::uint64_t publication) const {
    validate(root);
    if (requires_extended_(root) ||
        (closure_ &&
         (closure_->replicated_realm || !closure_->domains.empty() || !closure_->arrays.empty())))
      return encode_extended_(root, publication);

    continuation_detail::Writer out(output_capacity_);

    out.text("POPSCONT1");
    out.text(identity_);
    out.word(Dim);
    out.word(publication);
    std::vector<FiniteDomainRef> domains;
    collect_domains_(root, domains, capacity_);
    for (const auto& domain : domains)
      if (!domain->realm() || domain->realm() != domains.front()->realm())
        throw std::invalid_argument("continuation combines domains from distinct native realms");
    for (std::size_t i = 0; i < domains.size(); ++i)
      for (std::size_t j = 0; j < i; ++j)
        if (domains[i]->version == domains[j]->version)
          throw std::invalid_argument("continuation domains reuse a runtime instance version");
    out.word(domains.size());
    for (const auto& d : domains) {
      out.word(d->size);
      out.word(d->capacity);
      out.word(d->family);
      out.word(d->version);
    }
    encode_(root.get(), domains, out);
    return std::move(out.bytes);
  }
  // Exact rank-independent image of replicated leaves and structural ownership. Field cell
  // values are distributed data, so only their common native layout/owner contract appears here.
  std::string collective_identity(const Ref& root, std::uint64_t publication) const {
    validate(root);

    continuation_detail::Writer out(output_capacity_);

    out.text(identity_);
    out.word(publication);
    continuation_detail::walk_value<Dim>(
        root, capacity_, [&](const Value& node) { return collective_node_(node, out); },
        [](const Value&) {});
    return std::string(reinterpret_cast<const char*>(out.bytes.data()), out.bytes.size());
  }
  Decoded decode(std::span<const std::uint8_t> bytes) const {
    if (bytes.size() > capacity_)
      throw std::length_error("continuation archive exceeds installed capacity");
    continuation_detail::Reader in(bytes);
    const auto wire = in.text();
    if (wire == "POPSCONT2" || wire == "POPSCONT3" || wire == "POPSCONT4" || wire == "POPSCONT5" ||
        wire == "POPSCONT6")
      return decode_extended_(in, wire != "POPSCONT2", wire == "POPSCONT4",
                              wire == "POPSCONT5" || wire == "POPSCONT6", wire == "POPSCONT6");
    if (wire != "POPSCONT1" || in.text() != identity_ || in.word() != Dim)
      throw std::invalid_argument(
          "continuation image has a different schema, dimension or wire version");
    Decoded out;
    out.publication = in.word();
    const auto count = in.word();
    if (count > (bytes.size() - in.at) / 32)
      throw std::invalid_argument("continuation domain count exceeds its byte image");
    // CONT1 carries an allocator only through its domain table. An empty table
    // must not invent a realm which a later read would synchronize and persist.
    if (count != 0) {
      out.realm = FiniteDomainRealmRef(new FiniteDomainRealm);
      out.realm->replicated_ = true;
      out.realms.push_back(out.realm);
    }
    std::vector<FiniteDomainRef> domains;
    domains.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t n = 0; n < count; ++n) {
      const auto size = signed_word_(in), capacity = signed_word_(in), family = signed_word_(in),
                 version = signed_word_(in);
      if (size < 0 || capacity < size || family < 0 || version <= 0)
        throw std::invalid_argument("continuation domain table has invalid bounds or version");
      for (const auto& prior : domains)
        if (prior->version == version)
          throw std::invalid_argument(
              "continuation domain table duplicates a runtime instance version");
      domains.push_back(
          FiniteDomainRef(new FiniteDomain(size, capacity, family, version, out.realm)));
      out.max_domain_version = std::max(out.max_domain_version, version);
    }
    out.root = decode_(type_, domains, in);
    if (in.at != bytes.size())
      throw std::invalid_argument("continuation image has trailing bytes");
    validate(out.root);
    std::vector<FiniteDomainRef> used;
    collect_domains_(out.root, used, capacity_);
    if (used.size() != domains.size())
      throw std::invalid_argument("continuation image contains unreferenced domains");
    return out;
  }

 private:
  friend class ContinuationTransportSession<Dim>;
  ContinuationType type_;
  std::string identity_;
  std::uint64_t capacity_;

  std::uint64_t output_capacity_;

  Prototypes prototypes_;
  const ExecutionLane* lane_ = nullptr;
  const distributed::Registry* registry_ = nullptr;
  const ContinuationClosure<Dim>* closure_ = nullptr;
  DecodeContext* decode_context_ = nullptr;

  ArchiveFieldResolver archive_fields_;

#include <pops/runtime/program/continuation_codec_extended.inc>
  static void encode_field_support_(const ContinuationFieldSupport::Ref& support,
                                    continuation_detail::Writer& out, bool provider_wire = false) {
    out.word(bool(support));
    if (!support)
      return;
    support->validate();
    if (!provider_wire && !support->field_provider.empty())
      throw std::invalid_argument("provider Field support requires CONT6 framing");
    out.text(support->program_identity);
    out.text(support->origin_point);
    out.word(support->program_owner);
    out.word(support->producer_level);
    out.word(support->topology_epoch);
    out.word(support->materialization_generation);
    if (provider_wire)
      out.text(support->field_provider);
  }
  static ContinuationFieldSupport::Ref decode_field_support_(
      continuation_detail::Reader& in, const std::function<void(std::uint64_t)>& reserve = {},
      bool provider_wire = false) {
    const auto present = in.word();
    if (present > 1)
      throw std::invalid_argument("invalid Field producer support tag");
    if (!present)
      return {};
    auto check = in;
    const auto identity = check.blob(), point = check.blob();
    const auto owner = check.word(), level = check.word();
    (void)check.word();
    (void)check.word();
    const auto provider = provider_wire ? check.blob() : std::span<const std::uint8_t>{};
    const bool valid_owner =
        (owner <= std::uint64_t(std::numeric_limits<int>::max()) && provider.empty()) ||
        (provider_wire && owner == std::numeric_limits<std::uint64_t>::max() && !provider.empty());
    if (identity.empty() || point.empty() || !valid_owner ||
        level > std::uint64_t(std::numeric_limits<int>::max()))
      throw std::invalid_argument("invalid Field producer support descriptor");
    if (reserve)
      reserve(sizeof(ContinuationFieldSupport) + identity.size() + point.size() + provider.size());
    auto identity_text = in.text(), point_text = in.text();
    const auto encoded_owner = in.word();
    const int decoded_owner = encoded_owner == std::numeric_limits<std::uint64_t>::max()
                                  ? -1
                                  : static_cast<int>(encoded_owner);
    const int decoded_level = static_cast<int>(in.word());
    const auto epoch = in.word(), generation = in.word();
    auto provider_text = provider_wire ? in.text() : std::string{};
    auto support = std::make_shared<const ContinuationFieldSupport>(
        ContinuationFieldSupport{std::move(identity_text), std::move(point_text), decoded_owner,
                                 decoded_level, epoch, generation, std::move(provider_text)});
    support->validate();
    return support;
  }
  static void encode_field_traits_(const Value& value, continuation_detail::Writer& out) {
    out.text(value.field_provider());
    const auto& representation = value.field_representation();
    out.word(bool(representation));
    if (representation) {
      representation->validate();
      out.text(representation->contract);
      out.text(representation->origin);
    }
  }
  static FieldRepresentationObligation::Ref decode_field_traits_(
      const ContinuationType& type, continuation_detail::Reader& in,
      const std::function<void(std::uint64_t)>& reserve) {
    const auto provider = in.blob();
    if (std::string_view(reinterpret_cast<const char*>(provider.data()), provider.size()) !=
        type.field_provider)
      throw std::invalid_argument("continuation Field provider differs from installed schema");
    reserve(provider.size());  // The restored Value owns its provider key.
    const auto present = in.word();
    if (present > 1)
      throw std::invalid_argument("invalid Field representation obligation tag");
    if (!present)
      return {};
    auto check = in;
    const auto contract = check.blob(), origin = check.blob();
    if (contract.empty() || origin.empty())
      throw std::invalid_argument("Field representation obligation is incomplete");
    reserve(sizeof(FieldRepresentationObligation) + contract.size() + origin.size());
    auto contract_text = in.text();
    auto origin_text = in.text();
    return std::make_shared<const FieldRepresentationObligation>(
        FieldRepresentationObligation{std::move(contract_text), std::move(origin_text)});
  }
  bool collective_node_(const Value& node, continuation_detail::Writer& out) const {
    const auto* value = &node;
    out.word(static_cast<unsigned>(value->kind()));
    if (const auto& domain = value->domain_ref()) {
      out.word(domain->family);
      out.word(domain->capacity);
      out.word(domain->size);
      out.word(domain->version);
      // Legacy POPSCONT1 reconstructs its one replicated realm locally. Its aliases
      // have already been checked structurally; the execution context synchronizes a
      // fresh transport origin before the first subsequent use, not during local decode.
      const auto& realm = domain->realm();
      out.word(realm->replicated());
      const bool named = !realm->replicated() || realm->synchronized();
      out.word(named);
      if (named) {
        const auto origin = realm->transport_identity();
        out.word(origin[0]);
        out.word(origin[1]);
      }
    }
    if (const auto& domain = value->distributed_domain_ref()) {
      const auto id = domain->identity();
      out.word(id.issuer[0]);
      out.word(id.issuer[1]);
      out.word(id.ordinal);
      out.word(domain->family());
      out.word(domain->capacity());
      out.word(domain->partition_offsets().size());
      for (auto offset : domain->partition_offsets())
        out.word(offset);
      if (value->kind() == ContinuationKind::DistributedIndex)
        out.word(value->word_);
      if (value->kind() == ContinuationKind::DistributedArray) {
        const auto& array = value->distributed_array_ref();
        out.word(array->version.issuer[0]);
        out.word(array->version.issuer[1]);
        out.word(array->version.ordinal);
        out.word(bool(array->index_domain));
        if (array->index_domain) {
          const auto target = array->index_domain->identity();
          out.word(target.issuer[0]);
          out.word(target.issuer[1]);
          out.word(target.ordinal);
        }
      }
      return false;  // Local shards are distributed scientific data, not replicated control.
    }
    switch (value->kind()) {
      case ContinuationKind::Boolean:
      case ContinuationKind::Int64:
      case ContinuationKind::Float64:
      case ContinuationKind::Index:
      case ContinuationKind::Variant:
        out.word(value->word_);
        break;
      case ContinuationKind::Field: {
        out.word(value->field_owner());
        out.text(value->scientific_contract());
        if (value->field_exchange_expression()) {
          out.text("pops.field-exchange-expression@1");
          out.text(FieldExchangeExpressions::collective_identity(
              value->field_exchange_expression(), field_expression_limits_(out.capacity())));
        }
        const bool extended_traits =
            !value->field_provider().empty() || value->field_representation();
        if (extended_traits) {
          out.text("pops.field-traits@1");
          encode_field_traits_(*value, out);
        }
        if (value->field_support()) {
          out.text(extended_traits ? "pops.field-producer-support@2"
                                   : "pops.field-producer-support@1");
          encode_field_support_(value->field_support(), out, extended_traits);
        }
        const auto& field = value->field();
        out.word(field.ncomp());
        out.word(field.layout().size());
        out.word(field.distribution().replicated());
        for (int d = 0; d < Dim; ++d) {
          out.word(field.ghosts()[d]);
          out.word(std::int64_t(field.distribution().rank_space().origin()[d]));
          out.word(field.distribution().rank_space().extent()[d]);
        }
        for (std::size_t p = 0; p < field.layout().size(); ++p)
          for (int d = 0; d < Dim; ++d) {
            out.word(std::int64_t(field.layout()[p].lo[d]));
            out.word(std::int64_t(field.layout()[p].hi[d]));
            if (!field.distribution().replicated())
              out.word(std::int64_t(field.distribution().owner(p)[d]));
          }
        if (field.distribution().replicated()) {
          using Bits = std::conditional_t<sizeof(Real) == 8, std::uint64_t, std::uint32_t>;
          for (std::size_t p = 0; p < field.local_size(); ++p) {
            const auto& fab = field.fab(p);
            auto host = fab.create_host_mirror();
            fab.copy_to_host(host);
            const auto view = fab.view();
            const auto extent = fab.box().extent();
            for (std::int64_t cell = 0; cell < fab.box().numPts(); ++cell) {
              auto remainder = cell;
              std::size_t offset = 0;
              for (int d = 0; d < Dim; ++d) {
                const auto coordinate = fab.box().lo[d] + remainder % extent[d];
                remainder /= extent[d];
                offset += static_cast<std::size_t>(coordinate - view.origin[d]) * view.strides[d];
              }
              for (int c = 0; c < field.ncomp(); ++c)
                out.word(std::bit_cast<Bits>(host(offset + c * view.component_stride)));
            }
          }
        }
        break;
      }
      case ContinuationKind::Product:
      case ContinuationKind::Domain:
      case ContinuationKind::Array:
      case ContinuationKind::DistributedDomain:
      case ContinuationKind::DistributedIndex:
      case ContinuationKind::DistributedArray:
        break;
    }
    out.word(value->members().size());
    return true;
  }
  static std::int64_t signed_word_(continuation_detail::Reader& in) {
    return std::bit_cast<std::int64_t>(in.word());
  }
  void validate_(const ContinuationType& type, const Ref& value) const {
    struct Frame {
      const ContinuationType* type;
      const Ref* value;
      std::size_t next;
    };
    continuation_detail::StructuralBudget budget(capacity_);
    std::vector<Frame> pending;
    std::unordered_set<const Value*> active;
    auto inspect = [&](const ContinuationType& type, const Ref& value) {
      if (!value || value->kind() != type.kind)
        throw std::invalid_argument(
            "continuation value differs from its installed structural type");
      switch (type.kind) {
        case ContinuationKind::Boolean:
        case ContinuationKind::Int64:
        case ContinuationKind::Float64:
          return;
        case ContinuationKind::Product:
          if (value->members().size() != type.children.size())
            throw std::invalid_argument("continuation Product has missing or excess members");
          return;
        case ContinuationKind::Variant:
          if (value->arm() >= type.children.size() || value->members().size() != 1)
            throw std::invalid_argument("continuation Variant has an invalid active arm");
          return;
        case ContinuationKind::Domain:
          continuation_detail::domain(value->domain_ref(), type);
          return;
        case ContinuationKind::Index:
          continuation_detail::domain(value->domain_ref(), type);
          if (value->index_offset() < 0 || value->index_offset() >= value->domain_ref()->size)
            throw std::invalid_argument("continuation Index is outside its exact domain");
          return;
        case ContinuationKind::Array:
          continuation_detail::domain(value->domain_ref(), type);
          if (value->members().size() != static_cast<std::uint64_t>(value->domain_ref()->size))
            throw std::invalid_argument("continuation Array differs from its domain cardinality");
          return;
        case ContinuationKind::DistributedDomain:
        case ContinuationKind::DistributedIndex:
        case ContinuationKind::DistributedArray: {
          const auto& domain = value->distributed_domain_ref();
          if (!domain || domain->family() != type.family || domain->capacity() != type.capacity)
            throw std::invalid_argument(
                "distributed continuation domain differs from installed family");
          if (type.kind == ContinuationKind::DistributedIndex &&
              (value->distributed_index_offset() < 0 ||
               value->distributed_index_offset() >= domain->size()))
            throw std::invalid_argument("distributed continuation Index is outside its domain");
          if (type.kind == ContinuationKind::DistributedArray) {
            const auto& array = value->distributed_array_ref();
            if (!array || array->element != type.children[0] ||
                array->values.size() != static_cast<std::size_t>(domain->local_size()))
              throw std::invalid_argument(
                  "distributed continuation Array has a foreign element/shard schema");
            if (array->index_domain) {
              array->index_domain->require_live();
              if (array->element.kind != ContinuationKind::DistributedIndex ||
                  array->index_domain->family() != array->element.family ||
                  array->index_domain->capacity() != array->element.capacity)
                throw std::invalid_argument(
                    "distributed Array index target changes its installed schema");
              for (const auto& item : array->values)
                if (item->distributed_domain_ref() != array->index_domain)
                  throw std::invalid_argument("distributed Array mixes Index target instances");
            }
          }
          return;
        }
        case ContinuationKind::Field: {
          if (value->field_owner() != type.field_owner ||
              value->field_provider() != type.field_provider ||
              value->scientific_contract() != type.scientific_contract)
            throw std::invalid_argument(
                "continuation Field changes installed scientific authority");
          if (!type.field_provider.empty() && !value->field_support())
            throw std::invalid_argument("independent Field lacks its exact producer support");
          if (value->field_representation())
            value->field_representation()->validate();
          const auto& prototype = prototypes_(type, value->field_support());
          const auto& field = value->field();
          if (value->field_exchange_expression())
            FieldExchangeExpressions::require_expression(value->field_exchange_expression(),
                                                         field_expression_limits_(capacity_));
          if (value->field_owner() != type.field_owner ||
              value->scientific_contract() != type.scientific_contract ||
              field.ncomp() != type.field_components || field.layout() != prototype.layout() ||
              field.distribution() != prototype.distribution() ||
              field.local_rank() != prototype.local_rank())
            throw std::invalid_argument(
                "continuation Field differs from its installed storage/scientific authority");
          return;
        }
      }
      throw std::invalid_argument("unknown continuation structural kind");
    };
    auto push = [&](const ContinuationType& schema, const Ref& owned) {
      budget.consume();
      budget.frames(pending.size() + 1, sizeof(Frame));
      if (!owned || !active.insert(owned.get()).second)
        throw std::invalid_argument("continuation validation contains a missing or cyclic value");
      inspect(schema, owned);
      pending.push_back({&schema, &owned, 0});
    };
    push(type, value);
    while (!pending.empty()) {
      auto& frame = pending.back();
      const auto& members = (*frame.value)->members();
      if (frame.next != members.size()) {
        const auto ordinal = frame.next++;
        const auto selected = frame.type->kind == ContinuationKind::Product ? ordinal
                              : frame.type->kind == ContinuationKind::Variant
                                  ? (*frame.value)->arm()
                                  : 0;
        push(frame.type->children[selected], members[ordinal]);
      } else {
        validate_bindings_(*frame.type, *frame.value, budget);
        active.erase(frame.value->get());
        pending.pop_back();
      }
    }
  }
  static void validate_bindings_(const ContinuationType& type, const Ref& root,
                                 continuation_detail::StructuralBudget& budget) {
    for (const auto& binding : type.domain_bindings) {
      auto domain_value = root;
      for (auto item : binding.domain_path)
        domain_value = domain_value->members().at(item);
      const auto& domain = domain_value->domain_ref();
      const auto& distributed_domain = domain_value->distributed_domain_ref();
      if (!domain && !distributed_domain)
        throw std::invalid_argument("Bundle domain binder has no native authority");
      struct Frame {
        const Ref* value;
        std::size_t at;
      };
      std::vector<Frame> pending{{&root, 0}};
      while (!pending.empty()) {
        budget.consume();
        const auto frame = pending.back();
        pending.pop_back();
        const auto& value = *frame.value;
        const auto at = frame.at;
        if (at == binding.value_path.size()) {
          if (value->domain_ref() != domain ||
              value->distributed_domain_ref() != distributed_domain)
            throw std::invalid_argument(
                "Bundle dependency belongs to another native domain instance");
          continue;
        }
        auto push = [&](const Ref& child) {
          budget.frames(pending.size() + 1, sizeof(Frame));
          pending.push_back({&child, at + 1});
        };
        const auto item = binding.value_path[at];
        if (continuation_array_kind(value->kind()) && item == continuation_each) {
          if (at + 1 == binding.value_path.size() &&
              value->kind() == ContinuationKind::DistributedArray) {
            const auto& witness = value->distributed_array_ref()->index_domain;
            if (witness && witness != distributed_domain)
              throw std::invalid_argument(
                  "Bundle empty Index Array witness belongs to another instance");
          }
          for (auto i = value->members().size(); i != 0; --i)
            push(value->members()[i - 1]);
        } else if (value->kind() == ContinuationKind::Variant) {
          if (value->arm() == item)
            push(value->members().at(0));
        } else
          push(value->members().at(item));
      }
    }
  }
  static void collect_domains_(const Ref& value, std::vector<FiniteDomainRef>& domains,
                               std::uint64_t capacity) {
    continuation_detail::walk_value<Dim>(
        value, capacity,
        [&](const Value& node) {
          const auto* value = &node;
          if (value->domain_ref() &&
              std::find(domains.begin(), domains.end(), value->domain_ref()) == domains.end())
            domains.push_back(value->domain_ref());
          return true;
        },
        [](const Value&) {});
  }
  static void field_layout_(const Field& field, continuation_detail::Writer& out,
                            int components = -1,
                            const typename Field::ghost_type* ghosts = nullptr) {
    out.word(components < 0 ? field.ncomp() : components);
    for (int d = 0; d < Dim; ++d) {
      out.word(ghosts ? (*ghosts)[d] : field.ghosts()[d]);
      out.word(std::int64_t(field.local_rank()[d]));
      out.word(std::int64_t(field.distribution().rank_space().origin()[d]));
      out.word(field.distribution().rank_space().extent()[d]);
    }
    out.word(field.distribution().replicated());
    out.word(field.layout().size());
    for (std::size_t p = 0; p < field.layout().size(); ++p)
      for (int d = 0; d < Dim; ++d) {
        out.word(std::int64_t(field.layout()[p].lo[d]));
        out.word(std::int64_t(field.layout()[p].hi[d]));
        if (!field.distribution().replicated())
          out.word(std::int64_t(field.distribution().owner(p)[d]));
      }
  }
  static void encode_(const Value* value, const std::vector<FiniteDomainRef>& domains,
                      continuation_detail::Writer& out) {
    continuation_detail::walk_value<Dim>(
        value, out.capacity(),
        [&](const Value& node) {
          const auto* value = &node;
          out.word(static_cast<unsigned>(value->kind()));
          switch (value->kind()) {
            case ContinuationKind::Boolean:
            case ContinuationKind::Int64:
            case ContinuationKind::Float64:
              out.word(value->word_);
              return false;
            case ContinuationKind::Variant:
              out.word(value->arm());
              break;
            case ContinuationKind::Domain:
            case ContinuationKind::Index:
            case ContinuationKind::Array: {
              const auto found = std::find(domains.begin(), domains.end(), value->domain_ref());
              out.word(static_cast<std::uint64_t>(found - domains.begin()));
              if (value->kind() == ContinuationKind::Index)
                out.word(value->word_);
              if (value->kind() != ContinuationKind::Array)
                return false;
              break;
            }
            case ContinuationKind::Field: {
              const auto& field = value->field();
              field_layout_(field, out);
              out.word(field.local_size());
              using Bits = std::conditional_t<sizeof(Real) == 8, std::uint64_t, std::uint32_t>;
              out.word(sizeof(Real));
              for (std::size_t p = 0; p < field.local_size(); ++p) {
                out.word(field.global_index(p));
                auto host = field.fab(p).create_host_mirror();
                field.fab(p).copy_to_host(host);
                out.word(host.size());
                for (std::size_t n = 0; n < host.size(); ++n)
                  out.word(std::bit_cast<Bits>(host(n)));
              }
              return false;
            }
            case ContinuationKind::Product:
              break;
            case ContinuationKind::DistributedDomain:
            case ContinuationKind::DistributedIndex:
            case ContinuationKind::DistributedArray:
              throw std::logic_error("distributed checkpoint requires the extended alias codec");
          }
          out.word(value->members().size());
          return true;
        },
        [](const Value&) {});
  }
  Ref decode_(const ContinuationType& type, const std::vector<FiniteDomainRef>& domains,
              continuation_detail::Reader& in, ContinuationFieldSupport::Ref support = {},
              FieldRepresentationObligation::Ref representation = {}) const {
    using Frame = continuation_detail::DecodeFrame<Dim>;
    return continuation_detail::decode_structure<Dim>(
        type, capacity_, [&](const ContinuationType& type) -> Frame {
          if (continuation_distributed_kind(type.kind))
            throw std::invalid_argument(
                "distributed continuation values require the extended wire");
          if (in.word() != static_cast<unsigned>(type.kind))
            throw std::invalid_argument("continuation wire kind differs from installed schema");
          if (type.kind == ContinuationKind::Boolean) {
            const auto word = in.word();
            if (word > 1)
              throw std::invalid_argument("invalid continuation Bool bits");
            return Frame::leaf(Value::boolean(word != 0));
          }
          if (type.kind == ContinuationKind::Int64)
            return Frame::leaf(Value::integer(signed_word_(in)));
          if (type.kind == ContinuationKind::Float64)
            return Frame::leaf(Value::real(std::bit_cast<double>(in.word())));
          std::uint64_t arm = 0;
          if (type.kind == ContinuationKind::Variant) {
            arm = in.word();
            if (arm >= type.children.size())
              throw std::invalid_argument("invalid continuation sum arm");
          }
          FiniteDomainRef domain;
          if (type.kind == ContinuationKind::Domain || type.kind == ContinuationKind::Index ||
              type.kind == ContinuationKind::Array) {
            const auto d = in.word();
            if (d >= domains.size())
              throw std::invalid_argument("invalid continuation domain reference");
            domain = domains[d];
            continuation_detail::domain(domain, type);
            if (type.kind == ContinuationKind::Domain)
              return Frame::leaf(Value::domain(domain));
            if (type.kind == ContinuationKind::Index)
              return Frame::leaf(Value::index(domain, signed_word_(in)));
          }
          if (type.kind == ContinuationKind::Field) {
            if (support && (support->program_owner != type.field_owner ||
                            support->field_provider != type.field_provider))
              throw std::invalid_argument(
                  "continuation Field support owner/provider differs from schema");
            if (!type.field_provider.empty() && !support)
              throw std::invalid_argument("independent Field lacks its exact producer support");
            const auto& prototype = archive_fields_
                                        ? archive_fields_(type, in.bytes.subspan(in.at), support)
                                        : prototypes_(type, support);

            const auto begin = in.at;
            if (in.word() != static_cast<std::uint64_t>(type.field_components))
              throw std::invalid_argument("continuation field components differ from schema");
            typename Field::ghost_type ghosts{};
            for (int d = 0; d < Dim; ++d) {
              auto g = in.word();
              if (g > std::uint64_t(std::numeric_limits<int>::max()))
                throw std::invalid_argument(
                    "continuation field ghost extent exceeds native storage indices");
              ghosts[d] = static_cast<int>(g);
              (void)in.word();
              (void)in.word();
              (void)in.word();
            }
            for (std::size_t patch = 0; patch < prototype.layout().size(); ++patch)
              (void)continuation_detail::field_storage_elements(
                  prototype.layout()[patch], ghosts, 1, std::numeric_limits<std::uint64_t>::max());
            const auto replicated = in.word(), count = in.word();
            if (replicated > 1 || count != prototype.layout().size())
              throw std::invalid_argument(
                  "continuation field layout differs from installed prototype");
            for (std::size_t p = 0; p < count; ++p)
              for (int d = 0; d < Dim; ++d) {
                (void)in.word();
                (void)in.word();
                if (!replicated)
                  (void)in.word();
              }
            continuation_detail::Writer expected(capacity_);
            field_layout_(prototype, expected, type.field_components, &ghosts);
            if (expected.bytes.size() != in.at - begin ||
                !std::equal(expected.bytes.begin(), expected.bytes.end(), in.bytes.begin() + begin))
              throw std::invalid_argument("continuation field exact layout/ownership/rank changed");
            // Reject malformed/truncated field images before allocating any native field buffer.
            auto preflight = in;
            if (preflight.word() != prototype.local_size() || preflight.word() != sizeof(Real))
              throw std::invalid_argument("continuation field storage or precision changed");
            for (std::size_t p = 0; p < prototype.local_size(); ++p) {
              const auto elements = continuation_detail::field_storage_elements(
                  prototype.layout()[prototype.global_index(p)], ghosts, type.field_components,
                  (preflight.bytes.size() - preflight.at) / 8);
              if (preflight.word() != prototype.global_index(p) || preflight.word() != elements ||
                  elements > (preflight.bytes.size() - preflight.at) / 8)
                throw std::invalid_argument("continuation field payload exceeds its bounded image");
              preflight.at += static_cast<std::size_t>(elements * 8);
            }
            auto restored =
                std::make_shared<Field>(prototype.layout(), prototype.distribution(),
                                        prototype.local_rank(), type.field_components, ghosts);
            (void)in.word();
            (void)in.word();
            using Bits = std::conditional_t<sizeof(Real) == 8, std::uint64_t, std::uint32_t>;
            for (std::size_t p = 0; p < restored->local_size(); ++p) {
              auto& fab = restored->fab(p);
              auto host = fab.create_host_mirror();
              if (in.word() != restored->global_index(p) || in.word() != host.size() ||
                  host.size() > (in.bytes.size() - in.at) / 8)
                throw std::invalid_argument(
                    "continuation field patch payload differs from its allocation");
              for (std::size_t n = 0; n < host.size(); ++n) {
                const auto bits = in.word();
                if constexpr (sizeof(Bits) == 4)
                  if (bits > UINT32_MAX)
                    throw std::invalid_argument("continuation field bits exceed native precision");
                host(n) = std::bit_cast<Real>(static_cast<Bits>(bits));
              }
              fab.copy_from_host(host);
            }
            return Frame::leaf(Value::owned_field_(std::move(restored), type.field_owner,
                                                   type.scientific_contract, {}, support,
                                                   type.field_provider, representation));
          }
          const auto count = in.word();
          const auto expected = type.kind == ContinuationKind::Product ? type.children.size()
                                : type.kind == ContinuationKind::Variant
                                    ? 1
                                    : static_cast<std::uint64_t>(domain->size);
          if (count != expected || count > (in.bytes.size() - in.at) / 8)
            throw std::invalid_argument(
                "continuation member count differs from schema/cardinality or bytes");

          return Frame::container(type, std::move(domain), arm, static_cast<std::size_t>(count));
        });
  }
};

inline std::string continuation_transport_type_identity(const ContinuationType& type,
                                                        bool finite_dependencies,
                                                        std::uint64_t capacity) {
  type.validate(capacity);
  continuation_detail::Writer out(capacity);
  out.text(finite_dependencies ? "pops.continuation-transport-type@2"
                               : "pops.continuation-transport-type@1");
  continuation_detail::StructuralBudget budget(capacity);
  continuation_detail::walk_structure(
      &type, [](const ContinuationType& node) { return node.children.size(); },
      [](const ContinuationType& node, std::size_t i) { return &node.children[i]; },
      [&](const ContinuationType& node) {
        if (node.kind != ContinuationKind::Boolean && node.kind != ContinuationKind::Int64 &&
            node.kind != ContinuationKind::Float64 && node.kind != ContinuationKind::Product &&
            node.kind != ContinuationKind::Variant &&
            !(finite_dependencies &&
              (node.kind == ContinuationKind::Domain || node.kind == ContinuationKind::Index ||
               node.kind == ContinuationKind::Array || continuation_distributed_kind(node.kind))))
          throw std::invalid_argument(
              "transport accepts only Bool/Int64/Float64/Product/Variant payloads");
        out.word(static_cast<unsigned>(node.kind));
        out.word(node.names.size());
        for (const auto& name : node.names)
          out.text(name);
        if (finite_dependencies) {
          out.word(std::bit_cast<std::uint64_t>(node.family));
          out.word(node.capacity);
          out.word(node.dependent_bundle);
          out.word(node.domain_bindings.size());
          for (const auto& binding : node.domain_bindings) {
            out.word(binding.value_path.size());
            for (auto p : binding.value_path)
              out.word(p);
            out.word(binding.domain_path.size());
            for (auto p : binding.domain_path)
              out.word(p);
          }
        }
        out.word(node.children.size());
        return true;
      },
      [](const ContinuationType&) {}, budget);
  return std::string(reinterpret_cast<const char*>(out.bytes.data()), out.bytes.size());
}

/// Context-owned alias session. No method performs communication: its caller must converge
/// local allocation/codec failures before entering the next collective transport phase.
/// A decoded domain is interned across messages and operations, and a captured domain is
/// returned by exact shared identity, never replaced by an equal-cardinality lookalike.
template <int Dim>
class ContinuationTransportSession {
 public:
  using Value = ContinuationValue<Dim>;
  using Ref = ContinuationValueRef<Dim>;
  using Key = std::array<std::uint64_t, 3>;
  using ArrayResolver = std::function<ContinuationDistributedArrayRef<Dim>(
      distributed::Identity, distributed::DomainRef, const ContinuationType&)>;
  ContinuationTransportSession() = default;
  ContinuationTransportSession(const ContinuationTransportSession&) = delete;
  ContinuationTransportSession& operator=(const ContinuationTransportSession&) = delete;
  ~ContinuationTransportSession() noexcept { clear(); }

  void bind_distributed_registry(distributed::Registry& registry, ArrayResolver resolver) {
    if (!resolver)
      throw std::invalid_argument("transport requires an authenticated Array resolver");
    const auto found = registries_.find(registry.issuer());
    if (found != registries_.end() && found->second.registry != &registry)
      throw std::logic_error("transport session has conflicting distributed issuer authorities");
    if (!registries_.empty() && &registries_.begin()->second.registry->lane() != &registry.lane())
      throw std::logic_error("transport session registries belong to different execution lanes");
    registries_.insert_or_assign(registry.issuer(),
                                 RegistryBinding{&registry, std::move(resolver)});
  }

  void bind_authorities_to(ContinuationTransportSession& destination) const {
    for (const auto& [issuer, binding] : registries_)
      destination.bind_distributed_registry(*binding.registry, binding.resolver);
  }
  void authenticate_distributed_domain(const distributed::DomainRef& domain) const {
    if (!domain)
      throw std::logic_error("distributed transport domain is absent");
    binding_(domain->identity()).registry->authenticate(domain);
  }
  distributed::DomainRef resolve_distributed_domain(distributed::Identity id) const {
    return binding_(id).registry->resolve_domain(id);
  }

  void clear() noexcept {
    for (const auto& [identity, weak] : foreign_realms_)
      if (auto realm = weak.lock())
        realm->revoke();
    foreign_realms_.clear();
    known_realms_.clear();
    domains_.clear();
    registries_.clear();
  }

  // Rank-local housekeeping after a successful collective liveness frontier.
  // Never revoke a live authority while pruning these weak alias caches.
  void collect_expired_aliases() noexcept {
    for (auto i = domains_.begin(); i != domains_.end();)
      if (i->second.expired())
        i = domains_.erase(i);
      else
        ++i;
    for (auto i = foreign_realms_.begin(); i != foreign_realms_.end();)
      if (i->second.expired())
        i = foreign_realms_.erase(i);
      else
        ++i;
    for (auto i = known_realms_.begin(); i != known_realms_.end();)
      if (i->second.expired())
        i = known_realms_.erase(i);
      else
        ++i;
  }
  void capture(const Ref& value,
               std::uint64_t capacity = std::numeric_limits<std::uint64_t>::max()) {
    continuation_detail::walk_value<Dim>(
        value, capacity,
        [&](const Value& node) {
          const auto* value = &node;
          if (!value)
            throw std::invalid_argument("transport capture is absent");
          if (const auto& domain = value->domain_ref())
            capture_domain_(domain);
          if (const auto& domain = value->distributed_domain_ref()) {
            binding_(domain->identity()).registry->authenticate(domain);
          }
          if (const auto& array = value->distributed_array_ref(); array && array->index_domain) {
            binding_(array->index_domain->identity()).registry->authenticate(array->index_domain);
          }
          return true;
        },
        [](const Value&) {});
  }

  std::vector<std::uint8_t> encode(const ContinuationType& type, const Ref& value,
                                   std::uint64_t capacity) {
    auto codec = codec_(type, capacity);
    codec.validate(value);
    capture(value, capacity);
    std::vector<FiniteDomainRef> domains;
    collect_domains_(value, domains, capacity);
    continuation_detail::Writer out(capacity);
    continuation_detail::transport_header(out, codec.identity(), Dim, domains);
    encode_value_(value, domains, out);
    return std::move(out.bytes);
  }

  Ref decode(const ContinuationType& type, std::span<const std::uint8_t> bytes,
             std::uint64_t capacity) {
    if (bytes.size() > capacity)
      throw std::length_error("transport image exceeds capacity");
    auto codec = codec_(type, capacity);
    continuation_detail::Reader in(bytes);
    if (in.text() != "POPSTRNS2" || in.text() != codec.identity() || in.word() != Dim)
      throw std::invalid_argument("transport schema, dimension or wire differs");
    const auto count = in.word();
    if (count > (bytes.size() - in.at) / 56)
      throw std::invalid_argument("transport domain table is truncated");
    std::vector<FiniteDomainRef> domains;
    domains.reserve(static_cast<std::size_t>(count));
    std::vector<Key> keys;
    keys.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
      const std::array<std::uint64_t, 2> origin{in.word(), in.word()};
      const auto replicated = in.word();
      const auto size = codec.signed_word_(in), cap = codec.signed_word_(in),
                 family = codec.signed_word_(in), version = codec.signed_word_(in);
      if (replicated > 1 || size < 0 || cap < size || family < 0 || version <= 0)
        throw std::invalid_argument("transport domain descriptor is invalid");
      const Key key{origin[0], origin[1], static_cast<std::uint64_t>(version)};
      if (std::find(keys.begin(), keys.end(), key) != keys.end())
        throw std::invalid_argument("transport repeats an instance in its alias table");
      keys.push_back(key);
      auto found = domains_.find(key);
      FiniteDomainRef domain = found == domains_.end() ? FiniteDomainRef{} : found->second.lock();
      if (domain && !domain->visible())
        throw std::logic_error("transport refers to a revoked domain");
      if (domain) {
        if (domain->size != size || domain->capacity != cap || domain->family != family ||
            domain->realm()->replicated() != bool(replicated))
          throw std::invalid_argument("transport changed an existing domain descriptor");
      } else {
        auto realm = known_realms_[origin].lock();
        auto& known_realm = foreign_realms_[origin];
        if (!realm)
          realm = known_realm.lock();
        if (!realm) {
          realm = FiniteDomainRealmRef(new FiniteDomainRealm(origin, bool(replicated)));
          known_realm = realm;
        } else if (!realm->visible() || realm->replicated() != bool(replicated))
          throw std::logic_error("transport reuses a revoked or inconsistent domain origin");
        domain = FiniteDomainRef(new FiniteDomain(size, cap, family, version, realm));
        domains_[key] = domain;
      }
      domains.push_back(std::move(domain));
    }
    auto value = decode_value_(type, domains, in, capacity);
    if (in.at != bytes.size())
      throw std::invalid_argument("transport has trailing bytes");
    codec.validate(value);
    std::vector<FiniteDomainRef> used;
    collect_domains_(value, used, capacity);
    if (used.size() != domains.size())
      throw std::invalid_argument("transport includes unused domains");
    return value;
  }

 private:
  std::map<Key, std::weak_ptr<const FiniteDomain>> domains_;
  std::map<std::array<std::uint64_t, 2>, std::weak_ptr<const FiniteDomainRealm>> foreign_realms_;
  std::map<std::array<std::uint64_t, 2>, std::weak_ptr<const FiniteDomainRealm>> known_realms_;
  struct RegistryBinding {
    distributed::Registry* registry;
    ArrayResolver resolver;
  };
  std::map<std::array<std::uint64_t, 2>, RegistryBinding> registries_;
  const RegistryBinding& binding_(distributed::Identity identity) const {
    const auto found = registries_.find(identity.issuer);
    if (found == registries_.end())
      throw std::logic_error("distributed transport has no authorized issuer registry");
    return found->second;
  }
  static void collect_domains_(const Ref& value, std::vector<FiniteDomainRef>& domains,
                               std::uint64_t capacity) {
    continuation_detail::walk_value<Dim>(
        value, capacity,
        [&](const Value& node) {
          const auto* value = &node;
          if (value->kind() == ContinuationKind::DistributedArray)
            return false;  // authenticated vault reference
          if (const auto& domain = value->domain_ref();
              domain && std::find(domains.begin(), domains.end(), domain) == domains.end())
            domains.push_back(domain);
          return true;
        },
        [](const Value&) {});
  }
  static ContinuationCodec<Dim> codec_(const ContinuationType& type, std::uint64_t capacity) {
    return ContinuationCodec<Dim>(type, continuation_transport_type_identity(type, true, capacity),
                                  capacity, [](int) -> const MultiFab<Dim>& {
                                    throw std::logic_error("transport has no field prototype");
                                  });
  }
  void capture_domain_(const FiniteDomainRef& domain) {
    domain->require_live();
    const auto origin = domain->realm()->transport_identity();
    auto& known_realm = known_realms_[origin];
    if (auto previous = known_realm.lock();
        previous && previous != domain->realm() && previous->visible())
      throw std::logic_error("transport capture conflicts with a live finite realm");
    known_realm = domain->realm();
    const Key key{origin[0], origin[1], static_cast<std::uint64_t>(domain->version)};
    auto& known = domains_[key];
    if (auto previous = known.lock(); previous && previous != domain && previous->visible())
      throw std::logic_error("transport capture conflicts with a live domain authority");
    known = domain;
  }

  void encode_value_(const Ref& value, const std::vector<FiniteDomainRef>& domains,
                     continuation_detail::Writer& out) const {
    continuation_detail::walk_value<Dim>(
        value, out.capacity(),
        [&](const Value& node) {
          if (continuation_distributed_kind(node.kind())) {
            const auto& domain = node.distributed_domain_ref();
            const auto& binding = binding_(domain->identity());
            binding.registry->authenticate(domain);
            distributed::Identity version{};
            if (node.kind() == ContinuationKind::DistributedArray) {
              const auto& array = node.distributed_array_ref();
              if (array->version.issuer != domain->identity().issuer ||
                  binding.resolver(array->version, domain, array->element) != array)
                throw std::logic_error(
                    "distributed Array is not an authenticated published version");
              version = array->version;
            }
            continuation_detail::transport_distributed_node(
                out, node.kind(), domain->identity(), domain->family(), domain->capacity(),
                domain->partition_offsets(), std::bit_cast<std::int64_t>(node.word_), version);
            return false;
          }
          std::uint64_t slot = 0;
          if (node.domain_ref()) {
            const auto found = std::find(domains.begin(), domains.end(), node.domain_ref());
            if (found == domains.end())
              throw std::logic_error("transport lost a finite alias");
            slot = static_cast<std::uint64_t>(found - domains.begin());
          }
          return continuation_detail::transport_finite_node(out, node.kind(), node.word_, slot,
                                                            node.members().size());
        },
        [](const Value&) {});
  }

  Ref decode_value_(const ContinuationType& type, const std::vector<FiniteDomainRef>& domains,
                    continuation_detail::Reader& in, std::uint64_t capacity) const {
    using Frame = continuation_detail::DecodeFrame<Dim>;
    return continuation_detail::decode_structure<Dim>(
        type, capacity, [&](const ContinuationType& type) -> Frame {
          if (in.word() != static_cast<unsigned>(type.kind))
            throw std::invalid_argument("transport value kind differs");
          if (continuation_distributed_kind(type.kind)) {
            const distributed::Identity id{{in.word(), in.word()}, in.word()};
            const auto& binding = binding_(id);
            const auto family = std::bit_cast<std::int64_t>(in.word()),
                       cap = std::bit_cast<std::int64_t>(in.word());
            const auto n = in.word();
            if (n != static_cast<std::size_t>(binding.registry->lane().size() + 1) ||
                n > (in.bytes.size() - in.at) / 8)
              throw std::invalid_argument(
                  "distributed transport ownership differs from its execution lane");
            std::vector<std::int64_t> offsets;
            offsets.reserve(static_cast<std::size_t>(n));
            for (std::uint64_t j = 0; j < n; ++j)
              offsets.push_back(std::bit_cast<std::int64_t>(in.word()));
            if (family != type.family || cap != type.capacity)
              throw std::invalid_argument(
                  "distributed transport family differs from installed schema");
            auto domain = binding.registry->resolve(id, family, cap, offsets);
            if (type.kind == ContinuationKind::DistributedDomain)
              return Frame::leaf(Value::distributed_domain(std::move(domain)));
            if (type.kind == ContinuationKind::DistributedIndex)
              return Frame::leaf(Value::distributed_index(std::move(domain),
                                                          std::bit_cast<std::int64_t>(in.word())));
            const distributed::Identity version{{in.word(), in.word()}, in.word()};
            if (version.issuer != id.issuer)
              throw std::logic_error("distributed Array transport changes its domain issuer");
            return Frame::leaf(Value::distributed_array(
                binding.resolver(version, std::move(domain), type.children[0])));
          }
          if (type.kind == ContinuationKind::Boolean) {
            const auto v = in.word();
            if (v > 1)
              throw std::invalid_argument("transport Bool is invalid");
            return Frame::leaf(Value::boolean(v != 0));
          }
          if (type.kind == ContinuationKind::Int64)
            return Frame::leaf(Value::integer(std::bit_cast<std::int64_t>(in.word())));
          if (type.kind == ContinuationKind::Float64)
            return Frame::leaf(Value::real(std::bit_cast<double>(in.word())));
          std::uint64_t arm = 0;
          if (type.kind == ContinuationKind::Variant) {
            arm = in.word();
            if (arm >= type.children.size())
              throw std::invalid_argument("transport Variant tag is invalid");
          }
          FiniteDomainRef domain;
          if (type.kind == ContinuationKind::Domain || type.kind == ContinuationKind::Index ||
              type.kind == ContinuationKind::Array) {
            const auto slot = in.word();
            if (slot >= domains.size())
              throw std::invalid_argument("transport domain alias is invalid");
            domain = domains[slot];
            continuation_detail::domain(domain, type);
            if (type.kind == ContinuationKind::Domain)
              return Frame::leaf(Value::domain(std::move(domain)));
            if (type.kind == ContinuationKind::Index)
              return Frame::leaf(
                  Value::index(std::move(domain), std::bit_cast<std::int64_t>(in.word())));
          }
          const auto count = in.word();
          const auto expected = type.kind == ContinuationKind::Product   ? type.children.size()
                                : type.kind == ContinuationKind::Variant ? 1
                                : domain ? static_cast<std::uint64_t>(domain->size)
                                         : UINT64_MAX;
          if (count != expected || count > (in.bytes.size() - in.at) / 8)
            throw std::invalid_argument("transport member count differs from its installed schema");

          return Frame::container(type, std::move(domain), arm, static_cast<std::size_t>(count));
        });
  }
};

#include <pops/runtime/program/continuation_decode_context.inc>

// Deterministic rank-local retained-byte policy shared by publication and restore.
// The enclosing caller owns the collective phase; this routine never communicates.
template <int Dim>
std::uint64_t continuation_distributed_array_charge(
    const ContinuationDistributedArrayRef<Dim>& array, distributed::Registry& registry,
    std::uint64_t capacity,
    std::span<const std::shared_ptr<distributed::Registry>> authorized_registries,
    const ContinuationTransportSession<Dim>* transport) {
  if (!array || !array->domain || array->values.size() != std::size_t(array->domain->local_size()))
    throw std::invalid_argument("distributed Array charge has inconsistent ownership");
  registry.authenticate(array->domain);
  ContinuationTransportSession<Dim> session;
  const auto bind = [&](distributed::Registry& authority) {
    session.bind_distributed_registry(
        authority, [&authority](distributed::Identity version, distributed::DomainRef domain,
                                const ContinuationType& element) {
          authority.authenticate(domain);
          auto held = std::static_pointer_cast<const ContinuationDistributedArray<Dim>>(
              authority.resolve_value(version, distributed::array_type_tag<Dim>()));
          if (!held || held->domain != domain || held->element != element ||
              held->version.issuer != version.issuer || held->version.ordinal != version.ordinal ||
              held->values.size() != std::size_t(domain->local_size()))
            throw std::invalid_argument(
                "Array charge dependency differs from its immutable vault version");
          return held;
        });
  };
  bind(registry);
  if (transport)
    transport->bind_authorities_to(session);
  for (const auto& authority : authorized_registries) {
    if (!authority)
      throw std::invalid_argument("Array charge has an absent issuer authority");
    bind(*authority);
  }
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  const auto type_bytes =
      continuation_transport_type_identity(array->element, true, capacity).size();
  if (array->domain->partition_offsets().size() > (maximum - 256) / 8)
    throw std::length_error("distributed Array charge overflow");
  std::uint64_t charge = 256 + 8 * array->domain->partition_offsets().size();
  if (type_bytes > maximum - charge)
    throw std::length_error("distributed Array charge overflow");
  charge += type_bytes;
  for (const auto& value : array->values) {
    const auto bytes = session.encode(array->element, value, capacity);
    if (charge > maximum - 32 || bytes.size() > maximum - charge - 32)
      throw std::length_error("distributed Array charge overflow");
    charge += 32 + bytes.size();
  }
  return charge;
}

template <int Dim>
std::vector<std::uint8_t> encode_continuation_transport(
    const ContinuationType& type, ContinuationValueRef<Dim> value, std::uint64_t capacity,
    ContinuationTransportSession<Dim>& session) {
  return session.encode(type, value, capacity);
}
template <int Dim>
ContinuationValueRef<Dim> decode_continuation_transport(
    const ContinuationType& type, std::span<const std::uint8_t> bytes, std::uint64_t capacity,
    ContinuationTransportSession<Dim>& session) {
  return session.decode(type, bytes, capacity);
}

/// Local-only payload codec for native transports. No collective, pointer, domain or field wire.
template <int Dim>
std::vector<std::uint8_t> encode_continuation_transport(const ContinuationType& type,
                                                        ContinuationValueRef<Dim> value,
                                                        std::uint64_t capacity) {
  ContinuationCodec<Dim> codec(type, continuation_transport_type_identity(type, false, capacity),
                               capacity, [](int) -> const MultiFab<Dim>& {
                                 throw std::logic_error("transport has no field prototype");
                               });
  return codec.encode(value, 0);
}
template <int Dim>
ContinuationValueRef<Dim> decode_continuation_transport(const ContinuationType& type,
                                                        std::span<const std::uint8_t> bytes,
                                                        std::uint64_t capacity) {
  ContinuationCodec<Dim> codec(type, continuation_transport_type_identity(type, false, capacity),
                               capacity, [](int) -> const MultiFab<Dim>& {
                                 throw std::logic_error("transport has no field prototype");
                               });
  auto result = codec.decode(bytes);
  if (result.publication != 0)
    throw std::invalid_argument("transport payload carries publication authority");
  return std::move(result.root);
}

}  // namespace pops::runtime::program
