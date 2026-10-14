#pragma once

#include <pops/mesh/storage/multifab.hpp>
#include <pops/runtime/program/distributed_domain.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::runtime::program {

// Structural language types only: no physical-model or numerical-method dispatch.
enum class ContinuationKind : std::uint8_t {
  Boolean, Int64, Float64, Product, Variant, Domain, Index, Array, Field,
  DistributedDomain, DistributedIndex, DistributedArray
};

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
  return kind == ContinuationKind::DistributedDomain || kind == ContinuationKind::DistributedIndex ||
         kind == ContinuationKind::DistributedArray;
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
  bool operator==(const ContinuationType&) const = default;

  const ContinuationType& dependent_at(const std::vector<std::size_t>& path,
                                     bool binder = false) const {
    const auto* value = this;
    for (auto item : path) {
      if (!binder && continuation_array_kind(value->kind) && item == continuation_each) {
        if (value->children.size() != 1) throw std::invalid_argument("invalid Bundle array path");
        value = &value->children[0];
      } else if ((value->kind == ContinuationKind::Product ||
                  (!binder && value->kind == ContinuationKind::Variant)) && item < value->children.size()) {
        value = &value->children[item];
      } else throw std::invalid_argument("Bundle dependency path changes structural type");
    }
    return *value;
  }

  void validate_dependencies() const {
    if (!dependent_bundle) {
      if (!domain_bindings.empty()) throw std::invalid_argument("domain bindings require a dependent Bundle");
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
    auto collect = [&](auto&& self, const ContinuationType& type,
                       std::vector<std::size_t> path, bool root) -> void {
      if (!root && type.dependent_bundle) return;
      if (continuation_domain_kind(type.kind) || continuation_index_kind(type.kind) ||
          continuation_array_kind(type.kind)) required.push_back(path);
      if (type.kind == ContinuationKind::Field)
        throw std::invalid_argument("legacy Field is not a finite Bundle representation");
      for (std::size_t i = 0; i < type.children.size(); ++i) {
        auto child_path = path;
        child_path.push_back(continuation_array_kind(type.kind) ? continuation_each : i);
        self(self, type.children[i], std::move(child_path), false);
      }
    };
    collect(collect, *this, {}, true);
    for (const auto& binding : domain_bindings) {
      const auto& dependent = dependent_at(binding.value_path);
      const auto& domain = dependent_at(binding.domain_path, true);
      if (!continuation_domain_kind(domain.kind) ||
          (!continuation_domain_kind(dependent.kind) && !continuation_index_kind(dependent.kind) &&
           !continuation_array_kind(dependent.kind)) ||
          dependent.family != domain.family || dependent.capacity != domain.capacity ||
          continuation_distributed_kind(dependent.kind) != continuation_distributed_kind(domain.kind) ||
          (continuation_domain_kind(dependent.kind) && binding.value_path != binding.domain_path))
        throw std::invalid_argument("Bundle binding changes its domain family or bounds");
      supplied.push_back(binding.value_path);
    }
    std::sort(required.begin(), required.end()); std::sort(supplied.begin(), supplied.end());
    if (required != supplied)
      throw std::invalid_argument("Bundle does not bind every dependent value exactly once");
  }

  void validate(unsigned depth = 0) const {
    if (depth > 128 || static_cast<unsigned>(kind) > static_cast<unsigned>(ContinuationKind::DistributedArray))
      throw std::invalid_argument("continuation schema exceeds structural depth or has an unknown kind");
    const bool record = kind == ContinuationKind::Product || kind == ContinuationKind::Variant;
    if ((record && (names.size() != children.size() ||
                    (kind == ContinuationKind::Variant && names.empty()))) ||
        (!record && !names.empty()) ||
        (continuation_array_kind(kind) && children.size() != 1) ||
        (!record && !continuation_array_kind(kind) && !children.empty()))
      throw std::invalid_argument("continuation schema has inconsistent structural children");
    for (std::size_t i = 0; i < names.size(); ++i)
      if (names[i].empty() || std::find(names.begin(), names.begin() + i, names[i]) != names.begin() + i)
        throw std::invalid_argument("continuation schema member names are empty or duplicated");
    if ((continuation_domain_kind(kind) || continuation_index_kind(kind) || continuation_array_kind(kind))
        && (family < 0 || capacity < 0))
      throw std::invalid_argument("continuation domain schema lacks bounded family authority");
    if (kind == ContinuationKind::Field &&
        (field_owner < 0 || field_components < 1 || scientific_contract.empty()))
      throw std::invalid_argument("continuation field schema lacks exact owner/components/scientific authority");
    for (const auto& child : children) child.validate(depth + 1);
    validate_dependencies();
  }
};
inline std::string continuation_transport_type_identity(const ContinuationType&, bool);

template<int Dim> class ContinuationCodec;
template<int Dim> class ContinuationTransportSession;
class FiniteDomainRegistry;
class FiniteDomainRealm {
 public:
  bool visible() const noexcept { return live_.load(std::memory_order_acquire); }
  std::array<std::uint64_t, 2> transport_identity() const {
    if (!visible()) throw std::logic_error("cannot transport a revoked finite realm");
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
  template<int> friend class ContinuationCodec;
  template<int> friend class ContinuationTransportSession;
  FiniteDomainRealm() {
    std::random_device random;
    for (auto& word : identity_) word = (std::uint64_t(random()) << 32) ^ std::uint64_t(random());
  }
  explicit FiniteDomainRealm(std::array<std::uint64_t, 2> identity, bool replicated)
      : exported_(true), replicated_(replicated), synchronized_(true), identity_(identity) {}
};
using FiniteDomainRealmRef = std::shared_ptr<const FiniteDomainRealm>;
class FiniteDomain {
 public:
  const std::int64_t size, capacity, family, version;
  const FiniteDomainRealmRef& realm() const noexcept { return realm_; }
  bool visible() const noexcept { return realm_ && realm_->visible(); }
  void require_live() const {
    if (!visible()) throw std::logic_error("finite domain belongs to a revoked runtime realm");
  }
 private:
  friend class FiniteDomainRegistry;
  template<int> friend class ContinuationCodec;
  template<int> friend class ContinuationTransportSession;
  FiniteDomain(std::int64_t count, std::int64_t cap, std::int64_t fam,
               std::int64_t ordinal, FiniteDomainRealmRef realm)
      : size(count), capacity(cap), family(fam), version(ordinal), realm_(std::move(realm)) {}
  FiniteDomainRealmRef realm_;
};
using FiniteDomainRef = std::shared_ptr<const FiniteDomain>;

// Live allocator: the ordinal is burned even if the enclosing mathematical trial rolls back.
// The realm is native possession, never reconstructed from an integer or pointer in an archive.
class FiniteDomainRegistry {
 public:
  explicit FiniteDomainRegistry(bool replicated = true) : realm_(new FiniteDomainRealm), replicated_(replicated) {
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
    if (!realm || !realm->visible()) throw std::logic_error("cannot adopt a revoked finite realm");
    if (realm == realm_) return;
    for (const auto& known : adopted_) if (known.lock() == realm) return;
    adopted_.emplace_back(realm);
  }
  void revoke_on_restart(FiniteDomainRealmRef keep = {}) noexcept {
    if (realm_ && realm_ != keep) realm_->revoke();
    for (const auto& known : adopted_)
      if (auto realm = known.lock(); realm && realm != keep) realm->revoke();
    adopted_.clear();
    realm_ = std::move(keep); // shared_ptr assignment is nonallocating
  }
  std::int64_t next_version() {
    if (next_ == std::numeric_limits<std::int64_t>::max())
      throw std::overflow_error("native finite-domain version exhausted");
    return ++next_;
  }
  void import_high_water(std::int64_t version) {
    if (version < 0) throw std::invalid_argument("invalid finite-domain high-water mark");
    next_ = std::max(next_, version);
  }
  FiniteDomainRef issue(std::int64_t count, std::int64_t capacity, std::int64_t family,
                        FiniteDomainRealmRef realm = {}) {
    const auto version = next_version();
    if (count < 0 || capacity < count || family < 0)
      throw std::invalid_argument("invalid native finite-domain bounds or family");
    if (realm) adopt(realm);
    else {
      if (!realm_ || !realm_->visible()) {
        realm_ = FiniteDomainRealmRef(new FiniteDomainRealm); realm_->replicated_ = replicated_;
      }
      realm = realm_;
    }
    return FiniteDomainRef(new FiniteDomain(count, capacity, family, version, std::move(realm)));
  }
 private:
  FiniteDomainRealmRef realm_;
  std::vector<std::weak_ptr<const FiniteDomainRealm>> adopted_;
  std::int64_t next_ = 0;
  bool replicated_ = true;
};

template<int Dim> class ContinuationValue;
template<int Dim> using ContinuationValueRef = std::shared_ptr<const ContinuationValue<Dim>>;
template<int Dim> class ContinuationCodec;
template<int Dim> struct ContinuationDistributedArray {
  distributed::DomainRef domain;
  ContinuationType element;
  std::vector<ContinuationValueRef<Dim>> values;
  distributed::Identity version{};
  distributed::DomainRef index_domain; // exact target witness, including empty Index shards
};
template<int Dim> using ContinuationDistributedArrayRef = std::shared_ptr<const ContinuationDistributedArray<Dim>>;
template<int Dim> std::uint64_t continuation_distributed_array_charge(
    const ContinuationDistributedArrayRef<Dim>&, distributed::Registry&, std::uint64_t);
template<int Dim> struct ContinuationClosure {
  std::vector<distributed::DomainRef> domains;
  std::vector<ContinuationDistributedArrayRef<Dim>> arrays;
  FiniteDomainRealmRef replicated_realm;
};

/// Immutable owned finite value. All construction here is rank-local; the caller converges
/// allocation/copy/fence failures before a subsequent native collective.
template<int Dim> class ContinuationValue {
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
    auto out = fresh_(ContinuationKind::Product); out->members_ = std::move(values); return out;
  }
  static Ref variant(std::uint32_t arm, Ref payload) {
    auto out = fresh_(ContinuationKind::Variant); out->word_ = arm;
    out->members_.push_back(std::move(payload)); return out;
  }
  static Ref domain(FiniteDomainRef value) {
    require_domain_(value);
    auto out = fresh_(ContinuationKind::Domain); out->domain_ = std::move(value); return out;
  }
  static Ref index(FiniteDomainRef value, std::int64_t offset) {
    require_domain_(value);
    auto out = fresh_(ContinuationKind::Index); out->domain_ = std::move(value);
    out->word_ = std::bit_cast<std::uint64_t>(offset); return out;
  }
  static Ref array(FiniteDomainRef domain, std::vector<Ref> values) {
    require_domain_(domain);
    auto out = fresh_(ContinuationKind::Array); out->domain_ = std::move(domain);
    out->members_ = std::move(values); return out;
  }
  static Ref distributed_domain(distributed::DomainRef domain) {
    if (!domain) throw std::invalid_argument("distributed continuation domain is absent");
    domain->require_live(); auto out = fresh_(ContinuationKind::DistributedDomain);
    out->distributed_domain_ = std::move(domain); return out;
  }
  static Ref distributed_index(distributed::DomainRef domain, std::int64_t gid) {
    if (!domain) throw std::invalid_argument("distributed continuation domain is absent");
    domain->require_live();
    if (gid < 0 || gid >= domain->size()) throw std::invalid_argument("distributed continuation Index is out of bounds");
    auto out = fresh_(ContinuationKind::DistributedIndex);
    out->distributed_domain_ = std::move(domain); out->word_ = std::bit_cast<std::uint64_t>(gid); return out;
  }
  static Ref distributed_array(ContinuationDistributedArrayRef<Dim> array) {
    if (!array || !array->domain) throw std::invalid_argument("distributed continuation Array is absent");
    array->domain->require_live();
    if (array->values.size() != static_cast<std::size_t>(array->domain->local_size()))
      throw std::invalid_argument("distributed continuation Array shard differs from ownership");
    auto out = fresh_(ContinuationKind::DistributedArray);
    out->distributed_domain_ = array->domain; out->distributed_array_ = std::move(array); return out;
  }
  static Ref field(const Field& value, int owner, std::string contract) {
    // Fab's copy constructor deep-copies Kokkos storage. No scratch borrow survives this call.
    return owned_field_(std::make_shared<const Field>(value), owner, std::move(contract));
  }
  ContinuationKind kind() const noexcept { return kind_; }
  const distributed::DomainRef& distributed_domain_ref() const {
    if (distributed_domain_) distributed_domain_->require_live(); return distributed_domain_;
  }
  const ContinuationDistributedArrayRef<Dim>& distributed_array_ref() const {
    if (distributed_domain_) distributed_domain_->require_live(); return distributed_array_;
  }
  std::int64_t distributed_index_offset() const {
    require_(ContinuationKind::DistributedIndex); distributed_domain_->require_live();
    return std::bit_cast<std::int64_t>(word_);
  }
  bool boolean() const { require_(ContinuationKind::Boolean); return word_ != 0; }
  std::int64_t integer() const { require_(ContinuationKind::Int64); return std::bit_cast<std::int64_t>(word_); }
  double real() const { require_(ContinuationKind::Float64); return std::bit_cast<double>(word_); }
  const std::vector<Ref>& members() const {
    if (kind_ == ContinuationKind::Array) require_domain_(domain_);
    if (kind_ == ContinuationKind::DistributedArray) {
      distributed_domain_->require_live(); return distributed_array_->values;
    }
    return members_;
  }
  std::uint32_t arm() const { require_(ContinuationKind::Variant); return static_cast<std::uint32_t>(word_); }
  const FiniteDomainRef& domain_ref() const {
    if (domain_) domain_->require_live();
    return domain_;
  }
  std::int64_t index_offset() const { require_(ContinuationKind::Index); require_domain_(domain_); return std::bit_cast<std::int64_t>(word_); }
  const Field& field() const { require_(ContinuationKind::Field); return *field_; }
  int field_owner() const noexcept { return field_owner_; }
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
  int field_owner_ = -1;
  std::string contract_;
  explicit ContinuationValue(ContinuationKind kind) : kind_(kind) {}
  static std::shared_ptr<ContinuationValue> fresh_(ContinuationKind kind) {
    return std::shared_ptr<ContinuationValue>(new ContinuationValue(kind));
  }
  static Ref scalar_(ContinuationKind kind, std::uint64_t word) {
    auto out = fresh_(kind); out->word_ = word; return out;
  }
  static Ref owned_field_(std::shared_ptr<const Field> field, int owner, std::string contract) {
    auto out = fresh_(ContinuationKind::Field); out->field_ = std::move(field);
    out->field_owner_ = owner; out->contract_ = std::move(contract); return out;
  }
  static void require_domain_(const FiniteDomainRef& domain) {
    if (!domain) throw std::invalid_argument("finite value has no domain authority");
    domain->require_live();
  }
  void require_(ContinuationKind kind) const {
    if (kind_ != kind) throw std::invalid_argument("continuation value projection changes its structural type");
  }
};

namespace continuation_detail {
class Writer {
 public:
  explicit Writer(std::uint64_t capacity) : capacity_(capacity) {}
  std::vector<std::uint8_t> bytes;
  void word(std::uint64_t value) {
    room(8); for (unsigned n = 0; n < 8; ++n) bytes.push_back(static_cast<std::uint8_t>(value >> (8 * n)));
  }
  void text(const std::string& value) {
    word(value.size()); room(value.size()); bytes.insert(bytes.end(), value.begin(), value.end());
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
    if (bytes.size() - at < 8) throw std::invalid_argument("continuation image is truncated");
    std::uint64_t value = 0;
    for (unsigned n = 0; n < 8; ++n) value |= std::uint64_t(bytes[at++]) << (8 * n);
    return value;
  }
  std::string text() {
    const auto size = word();
    if (size > bytes.size() - at) throw std::invalid_argument("continuation text exceeds its byte image");
    std::string value(reinterpret_cast<const char*>(bytes.data() + at), static_cast<std::size_t>(size));
    at += static_cast<std::size_t>(size); return value;
  }
};
inline void domain(const FiniteDomainRef& value, const ContinuationType& type) {
  if (!value || !value->visible() || value->size < 0 || value->size > value->capacity || value->capacity != type.capacity ||
      value->family != type.family || value->version <= 0)
    throw std::invalid_argument("continuation domain differs from its bounded installed family");
}
} // namespace continuation_detail

/// Structural local codec. Its prototype provider is the installed native storage authority,
/// not a checkpoint-supplied constructor. First profile is exact-layout/rank restart.
template<int Dim> class ContinuationCodec {
 public:
  using Value = ContinuationValue<Dim>;
  using Ref = typename Value::Ref;
  using Field = MultiFab<Dim>;
  using Prototypes = std::function<const Field&(int)>;
  struct Decoded {
    Ref root; std::uint64_t publication = 0; std::int64_t max_domain_version = 0;
    FiniteDomainRealmRef realm;
    std::vector<FiniteDomainRealmRef> realms;
    std::shared_ptr<distributed::Registry> distributed_registry;
    std::shared_ptr<const ContinuationClosure<Dim>> closure;
  };

  ContinuationCodec(ContinuationType type, std::string identity, std::uint64_t capacity,
                    Prototypes prototypes, const ExecutionLane* lane = nullptr,
                    const distributed::Registry* registry = nullptr,
                    const ContinuationClosure<Dim>* closure = nullptr)
      : type_(std::move(type)), identity_(std::move(identity)), capacity_(capacity),
        prototypes_(std::move(prototypes)), lane_(lane), registry_(registry), closure_(closure) {
    type_.validate();
    if (identity_.empty() || capacity_ < 32 || !prototypes_)
      throw std::invalid_argument("continuation codec requires installed identity, capacity and native prototypes");
  }
  const ContinuationType& type() const noexcept { return type_; }
  const std::string& identity() const noexcept { return identity_; }
  std::uint64_t capacity() const noexcept { return capacity_; }

  void validate(const Ref& root) const { validate_(type_, root); }
  std::vector<std::uint8_t> encode(const Ref& root, std::uint64_t publication) const {
    validate(root);
    if (requires_extended_(root) ||
        (closure_ && (!closure_->domains.empty() || !closure_->arrays.empty())))
      return encode_extended_(root, publication);
    continuation_detail::Writer out(capacity_);
    out.text("POPSCONT1"); out.text(identity_); out.word(Dim); out.word(publication);
    std::vector<FiniteDomainRef> domains;
    collect_domains_(root, domains);
    for (const auto& domain : domains)
      if (!domain->realm() || domain->realm() != domains.front()->realm())
        throw std::invalid_argument("continuation combines domains from distinct native realms");
    for (std::size_t i = 0; i < domains.size(); ++i)
      for (std::size_t j = 0; j < i; ++j)
        if (domains[i]->version == domains[j]->version)
          throw std::invalid_argument("continuation domains reuse a runtime instance version");
    out.word(domains.size());
    for (const auto& d : domains) {
      out.word(d->size); out.word(d->capacity); out.word(d->family); out.word(d->version);
    }
    encode_(root, domains, out);
    return std::move(out.bytes);
  }
  // Exact rank-independent image of replicated leaves and structural ownership. Field cell
  // values are distributed data, so only their common native layout/owner contract appears here.
  std::string collective_identity(const Ref& root, std::uint64_t publication) const {
    validate(root);
    continuation_detail::Writer out(capacity_); out.text(identity_); out.word(publication);
    auto visit = [&](auto&& self, const Ref& value) -> void {
      out.word(static_cast<unsigned>(value->kind()));
      if (const auto& domain = value->domain_ref()) {
        out.word(domain->family); out.word(domain->capacity); out.word(domain->size); out.word(domain->version);
        // Legacy POPSCONT1 reconstructs its one replicated realm locally. Its aliases
        // have already been checked structurally; the execution context synchronizes a
        // fresh transport origin before the first subsequent use, not during local decode.
        const auto& realm=domain->realm();out.word(realm->replicated());
        const bool named=!realm->replicated() || realm->synchronized();out.word(named);
        if(named) {const auto origin=realm->transport_identity();out.word(origin[0]);out.word(origin[1]);}
      }
      if (const auto& domain = value->distributed_domain_ref()) {
        const auto id = domain->identity();
        out.word(id.issuer[0]); out.word(id.issuer[1]); out.word(id.ordinal);
        out.word(domain->family()); out.word(domain->capacity());
        out.word(domain->partition_offsets().size());
        for (auto offset : domain->partition_offsets()) out.word(offset);
        if (value->kind() == ContinuationKind::DistributedIndex) out.word(value->word_);
        if (value->kind() == ContinuationKind::DistributedArray) {
          const auto& array = value->distributed_array_ref();
          out.word(array->version.issuer[0]); out.word(array->version.issuer[1]); out.word(array->version.ordinal);
          out.word(bool(array->index_domain));
          if (array->index_domain) {
            const auto target = array->index_domain->identity();
            out.word(target.issuer[0]); out.word(target.issuer[1]); out.word(target.ordinal);
          }
        }
        return; // Local shards are distributed scientific data, not replicated control.
      }
      switch (value->kind()) {
        case ContinuationKind::Boolean: case ContinuationKind::Int64: case ContinuationKind::Float64:
        case ContinuationKind::Index: case ContinuationKind::Variant: out.word(value->word_); break;
        case ContinuationKind::Field: {
          out.word(value->field_owner()); out.text(value->scientific_contract());
          const auto& field = value->field(); out.word(field.ncomp()); out.word(field.layout().size());
          out.word(field.distribution().replicated());
          for (int d = 0; d < Dim; ++d) {
            out.word(field.ghosts()[d]); out.word(std::int64_t(field.distribution().rank_space().origin()[d]));
            out.word(field.distribution().rank_space().extent()[d]);
          }
          for (std::size_t p = 0; p < field.layout().size(); ++p)
            for (int d = 0; d < Dim; ++d) {
              out.word(std::int64_t(field.layout()[p].lo[d])); out.word(std::int64_t(field.layout()[p].hi[d]));
              if (!field.distribution().replicated()) out.word(std::int64_t(field.distribution().owner(p)[d]));
            }
          if (field.distribution().replicated()) {
            using Bits = std::conditional_t<sizeof(Real) == 8, std::uint64_t, std::uint32_t>;
            for (std::size_t p = 0; p < field.local_size(); ++p) {
              const auto& fab = field.fab(p); auto host = fab.create_host_mirror(); fab.copy_to_host(host);
              const auto view = fab.view(); const auto extent = fab.box().extent();
              for (std::int64_t cell = 0; cell < fab.box().numPts(); ++cell) {
                auto remainder = cell; std::size_t offset = 0;
                for (int d = 0; d < Dim; ++d) {
                  const auto coordinate = fab.box().lo[d] + remainder % extent[d]; remainder /= extent[d];
                  offset += static_cast<std::size_t>(coordinate - view.origin[d]) * view.strides[d];
                }
                for (int c = 0; c < field.ncomp(); ++c)
                  out.word(std::bit_cast<Bits>(host(offset + c * view.component_stride)));
              }
            }
          }
          break;
        }
        case ContinuationKind::Product: case ContinuationKind::Domain: case ContinuationKind::Array:
        case ContinuationKind::DistributedDomain: case ContinuationKind::DistributedIndex:
        case ContinuationKind::DistributedArray: break;
      }
      out.word(value->members().size());
      for (const auto& child : value->members()) self(self, child);
    };
    visit(visit, root);
    return std::string(reinterpret_cast<const char*>(out.bytes.data()), out.bytes.size());
  }
  Decoded decode(std::span<const std::uint8_t> bytes) const {
    if (bytes.size() > capacity_) throw std::length_error("continuation archive exceeds installed capacity");
    continuation_detail::Reader in(bytes);
    const auto wire = in.text();
    if (wire == "POPSCONT2") return decode_extended_(in);
    if (wire != "POPSCONT1" || in.text() != identity_ || in.word() != Dim)
      throw std::invalid_argument("continuation image has a different schema, dimension or wire version");
    Decoded out; out.publication = in.word(); out.realm = FiniteDomainRealmRef(new FiniteDomainRealm);
    out.realm->replicated_ = true; out.realms.push_back(out.realm);
    const auto count = in.word();
    if (count > (bytes.size() - in.at) / 32)
      throw std::invalid_argument("continuation domain count exceeds its byte image");
    std::vector<FiniteDomainRef> domains;
    domains.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t n = 0; n < count; ++n) {
      const auto size = signed_word_(in), capacity = signed_word_(in), family = signed_word_(in), version = signed_word_(in);
      if (size < 0 || capacity < size || family < 0 || version <= 0)
        throw std::invalid_argument("continuation domain table has invalid bounds or version");
      for (const auto& prior : domains)
        if (prior->version == version)
          throw std::invalid_argument("continuation domain table duplicates a runtime instance version");
      domains.push_back(FiniteDomainRef(new FiniteDomain(size, capacity, family, version, out.realm)));
      out.max_domain_version = std::max(out.max_domain_version, version);
    }
    out.root = decode_(type_, domains, in);
    if (in.at != bytes.size()) throw std::invalid_argument("continuation image has trailing bytes");
    validate(out.root);
    std::vector<FiniteDomainRef> used; collect_domains_(out.root, used);
    if (used.size() != domains.size()) throw std::invalid_argument("continuation image contains unreferenced domains");
    return out;
  }

 private:
  friend class ContinuationTransportSession<Dim>;
  ContinuationType type_;
  std::string identity_;
  std::uint64_t capacity_;
  Prototypes prototypes_;
  const ExecutionLane* lane_ = nullptr;
  const distributed::Registry* registry_ = nullptr;
  const ContinuationClosure<Dim>* closure_ = nullptr;
#include <pops/runtime/program/continuation_codec_extended.inc>
  static std::int64_t signed_word_(continuation_detail::Reader& in) {
    return std::bit_cast<std::int64_t>(in.word());
  }
  void validate_(const ContinuationType& type, const Ref& value) const {
    if (!value || value->kind() != type.kind)
      throw std::invalid_argument("continuation value differs from its installed structural type");
    switch (type.kind) {
      case ContinuationKind::Boolean: case ContinuationKind::Int64: case ContinuationKind::Float64: return;
      case ContinuationKind::Product:
        if (value->members().size() != type.children.size())
          throw std::invalid_argument("continuation Product has missing or excess members");
        for (std::size_t n = 0; n < type.children.size(); ++n) validate_(type.children[n], value->members()[n]);
        validate_bindings_(type, value);
        return;
      case ContinuationKind::Variant:
        if (value->arm() >= type.children.size() || value->members().size() != 1)
          throw std::invalid_argument("continuation Variant has an invalid active arm");
        validate_(type.children[value->arm()], value->members()[0]); return;
      case ContinuationKind::Domain: continuation_detail::domain(value->domain_ref(), type); return;
      case ContinuationKind::Index:
        continuation_detail::domain(value->domain_ref(), type);
        if (value->index_offset() < 0 || value->index_offset() >= value->domain_ref()->size)
          throw std::invalid_argument("continuation Index is outside its exact domain");
        return;
      case ContinuationKind::Array:
        continuation_detail::domain(value->domain_ref(), type);
        if (value->members().size() != static_cast<std::uint64_t>(value->domain_ref()->size))
          throw std::invalid_argument("continuation Array differs from its domain cardinality");
        for (const auto& item : value->members()) validate_(type.children[0], item);
        return;
      case ContinuationKind::DistributedDomain:
      case ContinuationKind::DistributedIndex:
      case ContinuationKind::DistributedArray: {
        const auto& domain = value->distributed_domain_ref();
        if (!domain || domain->family() != type.family || domain->capacity() != type.capacity)
          throw std::invalid_argument("distributed continuation domain differs from installed family");
        if (type.kind == ContinuationKind::DistributedIndex &&
            (value->distributed_index_offset() < 0 || value->distributed_index_offset() >= domain->size()))
          throw std::invalid_argument("distributed continuation Index is outside its domain");
        if (type.kind == ContinuationKind::DistributedArray) {
          const auto& array = value->distributed_array_ref();
          if (!array || array->element != type.children[0] ||
              array->values.size() != static_cast<std::size_t>(domain->local_size()))
            throw std::invalid_argument("distributed continuation Array has a foreign element/shard schema");
          if (array->index_domain) {
            array->index_domain->require_live();
            if (array->element.kind != ContinuationKind::DistributedIndex ||
                array->index_domain->family() != array->element.family ||
                array->index_domain->capacity() != array->element.capacity)
              throw std::invalid_argument("distributed Array index target changes its installed schema");
            for (const auto& item : array->values)
              if (item->distributed_domain_ref() != array->index_domain)
                throw std::invalid_argument("distributed Array mixes Index target instances");
          }
          for (const auto& item : array->values) validate_(type.children[0], item);
        }
        return;
      }
      case ContinuationKind::Field: {
        const auto& prototype = prototypes_(type.field_owner);
        const auto& field = value->field();
        if (value->field_owner() != type.field_owner || value->scientific_contract() != type.scientific_contract ||
            field.ncomp() != type.field_components || field.layout() != prototype.layout() ||
            field.distribution() != prototype.distribution() || field.local_rank() != prototype.local_rank() ||
            field.ghosts() != prototype.ghosts())
          throw std::invalid_argument("continuation Field differs from its installed storage/scientific authority");
        return;
      }
    }
    throw std::invalid_argument("unknown continuation structural kind");
  }
  static void validate_bindings_(const ContinuationType& type, const Ref& root) {
    for (const auto& binding : type.domain_bindings) {
      auto domain_value = root;
      for (auto item : binding.domain_path) domain_value = domain_value->members().at(item);
      const auto& domain = domain_value->domain_ref();
      const auto& distributed_domain = domain_value->distributed_domain_ref();
      if (!domain && !distributed_domain) throw std::invalid_argument("Bundle domain binder has no native authority");
      auto visit = [&](auto&& self, const Ref& value, std::size_t at) -> void {
        if (at == binding.value_path.size()) {
          if (value->domain_ref() != domain || value->distributed_domain_ref() != distributed_domain)
            throw std::invalid_argument("Bundle dependency belongs to another native domain instance");
          return;
        }
        const auto item = binding.value_path[at];
        if (continuation_array_kind(value->kind()) && item == continuation_each) {
          if (at+1 == binding.value_path.size() && value->kind() == ContinuationKind::DistributedArray) {
            const auto& witness = value->distributed_array_ref()->index_domain;
            if (witness && witness != distributed_domain)
              throw std::invalid_argument("Bundle empty Index Array witness belongs to another instance");
          }
          for (const auto& child : value->members()) self(self, child, at + 1);
        } else if (value->kind() == ContinuationKind::Variant) {
          if (value->arm() == item) self(self, value->members().at(0), at + 1);
        } else self(self, value->members().at(item), at + 1);
      };
      visit(visit, root, 0);
    }
  }
  static void collect_domains_(const Ref& value, std::vector<FiniteDomainRef>& domains) {
    if (value->domain_ref() && std::find(domains.begin(), domains.end(), value->domain_ref()) == domains.end())
      domains.push_back(value->domain_ref());
    for (const auto& member : value->members()) collect_domains_(member, domains);
  }
  static void field_layout_(const Field& field, continuation_detail::Writer& out, int components = -1) {
    out.word(components < 0 ? field.ncomp() : components);
    for (int d = 0; d < Dim; ++d) {
      out.word(field.ghosts()[d]); out.word(std::int64_t(field.local_rank()[d]));
      out.word(std::int64_t(field.distribution().rank_space().origin()[d]));
      out.word(field.distribution().rank_space().extent()[d]);
    }
    out.word(field.distribution().replicated()); out.word(field.layout().size());
    for (std::size_t p = 0; p < field.layout().size(); ++p)
      for (int d = 0; d < Dim; ++d) {
        out.word(std::int64_t(field.layout()[p].lo[d])); out.word(std::int64_t(field.layout()[p].hi[d]));
        if (!field.distribution().replicated()) out.word(std::int64_t(field.distribution().owner(p)[d]));
      }
  }
  static void encode_(const Ref& value, const std::vector<FiniteDomainRef>& domains,
                      continuation_detail::Writer& out) {
    out.word(static_cast<unsigned>(value->kind()));
    switch (value->kind()) {
      case ContinuationKind::Boolean: case ContinuationKind::Int64: case ContinuationKind::Float64:
        out.word(value->word_); return;
      case ContinuationKind::Variant: out.word(value->arm()); break;
      case ContinuationKind::Domain: case ContinuationKind::Index: case ContinuationKind::Array: {
        const auto found = std::find(domains.begin(), domains.end(), value->domain_ref());
        out.word(static_cast<std::uint64_t>(found - domains.begin()));
        if (value->kind() == ContinuationKind::Index) out.word(value->word_);
        if (value->kind() != ContinuationKind::Array) return;
        break;
      }
      case ContinuationKind::Field: {
        const auto& field = value->field();
        field_layout_(field, out); out.word(field.local_size());
        using Bits = std::conditional_t<sizeof(Real) == 8, std::uint64_t, std::uint32_t>;
        out.word(sizeof(Real));
        for (std::size_t p = 0; p < field.local_size(); ++p) {
          out.word(field.global_index(p));
          auto host = field.fab(p).create_host_mirror(); field.fab(p).copy_to_host(host);
          out.word(host.size());
          for (std::size_t n = 0; n < host.size(); ++n) out.word(std::bit_cast<Bits>(host(n)));
        }
        return;
      }
      case ContinuationKind::Product: break;
      case ContinuationKind::DistributedDomain: case ContinuationKind::DistributedIndex:
      case ContinuationKind::DistributedArray:
        throw std::logic_error("distributed checkpoint requires the extended alias codec");
    }
    out.word(value->members().size());
    for (const auto& member : value->members()) encode_(member, domains, out);
  }
  Ref decode_(const ContinuationType& type, const std::vector<FiniteDomainRef>& domains,
              continuation_detail::Reader& in) const {
    if (continuation_distributed_kind(type.kind))
      throw std::invalid_argument("distributed continuation values require the extended wire");
    if (in.word() != static_cast<unsigned>(type.kind))
      throw std::invalid_argument("continuation wire kind differs from installed schema");
    if (type.kind == ContinuationKind::Boolean) {
      const auto word = in.word(); if (word > 1) throw std::invalid_argument("invalid continuation Bool bits");
      return Value::boolean(word != 0);
    }
    if (type.kind == ContinuationKind::Int64) return Value::integer(signed_word_(in));
    if (type.kind == ContinuationKind::Float64) return Value::real(std::bit_cast<double>(in.word()));
    std::uint64_t arm = 0;
    if (type.kind == ContinuationKind::Variant) {
      arm = in.word(); if (arm >= type.children.size()) throw std::invalid_argument("invalid continuation sum arm");
    }
    FiniteDomainRef domain;
    if (type.kind == ContinuationKind::Domain || type.kind == ContinuationKind::Index || type.kind == ContinuationKind::Array) {
      const auto d = in.word(); if (d >= domains.size()) throw std::invalid_argument("invalid continuation domain reference");
      domain = domains[d]; continuation_detail::domain(domain, type);
      if (type.kind == ContinuationKind::Domain) return Value::domain(domain);
      if (type.kind == ContinuationKind::Index) return Value::index(domain, signed_word_(in));
    }
    if (type.kind == ContinuationKind::Field) {
      const auto& prototype = prototypes_(type.field_owner);
      const auto begin = in.at;
      if (in.word() != static_cast<std::uint64_t>(type.field_components))
        throw std::invalid_argument("continuation field components differ from schema");
      typename Field::ghost_type ghosts{};
      for (int d = 0; d < Dim; ++d) {
        auto g = in.word();
        if (g > std::uint64_t(std::numeric_limits<int>::max()) || g != std::uint64_t(prototype.ghosts()[d]))
          throw std::invalid_argument("continuation field ghost layout differs from installed prototype");
        ghosts[d] = static_cast<int>(g);
        (void)in.word(); (void)in.word(); (void)in.word();
      }
      const auto replicated = in.word(), count = in.word();
      if (replicated > 1 || count != prototype.layout().size())
        throw std::invalid_argument("continuation field layout differs from installed prototype");
      for (std::size_t p = 0; p < count; ++p)
        for (int d = 0; d < Dim; ++d) { (void)in.word(); (void)in.word(); if (!replicated) (void)in.word(); }
      continuation_detail::Writer expected(capacity_); field_layout_(prototype, expected, type.field_components);
      if (expected.bytes.size() != in.at - begin || !std::equal(expected.bytes.begin(), expected.bytes.end(), in.bytes.begin() + begin))
        throw std::invalid_argument("continuation field exact layout/ownership/rank changed");
      // Reject malformed/truncated field images before allocating any native field buffer.
      auto preflight = in;
      if (preflight.word() != prototype.local_size() || preflight.word() != sizeof(Real))
        throw std::invalid_argument("continuation field storage or precision changed");
      for (std::size_t p = 0; p < prototype.local_size(); ++p) {
        const auto cells = prototype.fab(p).size() / static_cast<std::uint64_t>(prototype.ncomp());
        if (cells > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(type.field_components))
          throw std::overflow_error("continuation field allocation extent overflow");
        const auto elements = cells * static_cast<std::uint64_t>(type.field_components);
        if (preflight.word() != prototype.global_index(p) || preflight.word() != elements ||
            elements > (preflight.bytes.size() - preflight.at) / 8)
          throw std::invalid_argument("continuation field payload exceeds its bounded image");
        preflight.at += static_cast<std::size_t>(elements * 8);
      }
      auto restored = std::make_shared<Field>(prototype.layout(), prototype.distribution(), prototype.local_rank(), type.field_components, ghosts);
      (void)in.word(); (void)in.word();
      using Bits = std::conditional_t<sizeof(Real) == 8, std::uint64_t, std::uint32_t>;
      for (std::size_t p = 0; p < restored->local_size(); ++p) {
        auto& fab = restored->fab(p); auto host = fab.create_host_mirror();
        if (in.word() != restored->global_index(p) || in.word() != host.size() || host.size() > (in.bytes.size() - in.at) / 8)
          throw std::invalid_argument("continuation field patch payload differs from its allocation");
        for (std::size_t n = 0; n < host.size(); ++n) {
          const auto bits = in.word();
          if constexpr (sizeof(Bits) == 4)
            if (bits > UINT32_MAX) throw std::invalid_argument("continuation field bits exceed native precision");
          host(n) = std::bit_cast<Real>(static_cast<Bits>(bits));
        }
        fab.copy_from_host(host);
      }
      return Value::owned_field_(std::move(restored), type.field_owner, type.scientific_contract);
    }
    const auto count = in.word();
    const auto expected = type.kind == ContinuationKind::Product ? type.children.size() :
                          type.kind == ContinuationKind::Variant ? 1 : static_cast<std::uint64_t>(domain->size);
    if (count != expected || count > (in.bytes.size() - in.at) / 8)
      throw std::invalid_argument("continuation member count differs from schema/cardinality or bytes");
    std::vector<Ref> values; values.reserve(static_cast<std::size_t>(count));
    for (std::size_t n = 0; n < count; ++n)
      values.push_back(decode_(type.children[type.kind == ContinuationKind::Product ? n :
                       type.kind == ContinuationKind::Variant ? arm : 0], domains, in));
    if (type.kind == ContinuationKind::Product) return Value::product(std::move(values));
    if (type.kind == ContinuationKind::Variant) return Value::variant(static_cast<std::uint32_t>(arm), std::move(values[0]));
    return Value::array(std::move(domain), std::move(values));
  }
};

inline std::string continuation_transport_type_identity(const ContinuationType& type,
                                                        bool finite_dependencies = false) {
  type.validate();
  continuation_detail::Writer out(std::numeric_limits<std::uint64_t>::max());
  const auto write = [&](auto&& self, const ContinuationType& node) -> void {
    if (node.kind != ContinuationKind::Boolean && node.kind != ContinuationKind::Int64 &&
        node.kind != ContinuationKind::Float64 && node.kind != ContinuationKind::Product &&
        node.kind != ContinuationKind::Variant &&
        !(finite_dependencies && (node.kind == ContinuationKind::Domain ||
          node.kind == ContinuationKind::Index || node.kind == ContinuationKind::Array ||
          continuation_distributed_kind(node.kind))))
      throw std::invalid_argument("transport accepts only Bool/Int64/Float64/Product/Variant payloads");
    out.word(static_cast<unsigned>(node.kind)); out.word(node.names.size());
    for (const auto& name : node.names) out.text(name);
    if (finite_dependencies) {
      out.word(std::bit_cast<std::uint64_t>(node.family)); out.word(node.capacity);
      out.word(node.dependent_bundle); out.word(node.domain_bindings.size());
      for (const auto& binding : node.domain_bindings) {
        out.word(binding.value_path.size()); for (auto p : binding.value_path) out.word(p);
        out.word(binding.domain_path.size()); for (auto p : binding.domain_path) out.word(p);
      }
    }
    out.word(node.children.size());
    for (const auto& child : node.children) self(self, child);
  };
  out.text(finite_dependencies ? "pops.continuation-transport-type@2" : "pops.continuation-transport-type@1");
  write(write, type);
  return std::string(reinterpret_cast<const char*>(out.bytes.data()), out.bytes.size());
}

/// Context-owned alias session. No method performs communication: its caller must converge
/// local allocation/codec failures before entering the next collective transport phase.
/// A decoded domain is interned across messages and operations, and a captured domain is
/// returned by exact shared identity, never replaced by an equal-cardinality lookalike.
template<int Dim> class ContinuationTransportSession {
 public:
  using Value = ContinuationValue<Dim>;
  using Ref = ContinuationValueRef<Dim>;
  using Key = std::array<std::uint64_t, 3>;
  using ArrayResolver = std::function<ContinuationDistributedArrayRef<Dim>(distributed::Identity,
      distributed::DomainRef, const ContinuationType&)>;
  ContinuationTransportSession() = default;
  ContinuationTransportSession(const ContinuationTransportSession&) = delete;
  ContinuationTransportSession& operator=(const ContinuationTransportSession&) = delete;
  ~ContinuationTransportSession() noexcept { clear(); }

  void bind_distributed_registry(distributed::Registry& registry, ArrayResolver resolver) {
    if (registry_ && registry_ != &registry)
      throw std::logic_error("transport session cannot switch a live distributed registry");
    if (!resolver) throw std::invalid_argument("transport requires an authenticated Array resolver");
    registry_ = &registry; array_resolver_ = std::move(resolver);
  }

  void clear() noexcept {
    for (const auto& [identity, weak] : foreign_realms_)
      if (auto realm = weak.lock()) realm->revoke();
    foreign_realms_.clear(); known_realms_.clear(); domains_.clear(); registry_ = nullptr; array_resolver_ = {};
  }

  // Rank-local housekeeping after a successful collective liveness frontier.
  // Never revoke a live authority while pruning these weak alias caches.
  void collect_expired_aliases() noexcept {
    for(auto i=domains_.begin();i!=domains_.end();)if(i->second.expired())i=domains_.erase(i);else ++i;
    for(auto i=foreign_realms_.begin();i!=foreign_realms_.end();)if(i->second.expired())i=foreign_realms_.erase(i);else ++i;
    for(auto i=known_realms_.begin();i!=known_realms_.end();)if(i->second.expired())i=known_realms_.erase(i);else ++i;
  }
  void capture(const Ref& value) {
    if (!value) throw std::invalid_argument("transport capture is absent");
    if (const auto& domain = value->domain_ref()) capture_domain_(domain);
    if (const auto& domain = value->distributed_domain_ref()) {
      if (!registry_) throw std::logic_error("distributed transport has no live registry");
      registry_->authenticate(domain);
    }
    if (const auto& array = value->distributed_array_ref(); array && array->index_domain) {
      if (!registry_) throw std::logic_error("distributed transport has no live registry");
      registry_->authenticate(array->index_domain);
    }
    for (const auto& child : value->members()) capture(child);
  }

  std::vector<std::uint8_t> encode(const ContinuationType& type, const Ref& value,
                                 std::uint64_t capacity) {
    auto codec = codec_(type, capacity); codec.validate(value);
    capture(value);
    std::vector<FiniteDomainRef> domains; collect_domains_(value, domains);
    continuation_detail::Writer out(capacity);
    out.text("POPSTRNS2"); out.text(codec.identity()); out.word(Dim); out.word(domains.size());
    for (const auto& domain : domains) {
      const auto origin = domain->realm()->transport_identity();
      out.word(origin[0]); out.word(origin[1]); out.word(domain->realm()->replicated());
      out.word(domain->size); out.word(domain->capacity); out.word(domain->family); out.word(domain->version);
    }
    encode_value_(value, domains, out);
    return std::move(out.bytes);
  }

  Ref decode(const ContinuationType& type, std::span<const std::uint8_t> bytes,
             std::uint64_t capacity) {
    if (bytes.size() > capacity) throw std::length_error("transport image exceeds capacity");
    auto codec = codec_(type, capacity);
    continuation_detail::Reader in(bytes);
    if (in.text() != "POPSTRNS2" || in.text() != codec.identity() || in.word() != Dim)
      throw std::invalid_argument("transport schema, dimension or wire differs");
    const auto count = in.word();
    if (count > (bytes.size()-in.at)/56) throw std::invalid_argument("transport domain table is truncated");
    std::vector<FiniteDomainRef> domains; domains.reserve(static_cast<std::size_t>(count));
    std::vector<Key> keys; keys.reserve(static_cast<std::size_t>(count));
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
      if (domain && !domain->visible()) throw std::logic_error("transport refers to a revoked domain");
      if (domain) {
        if (domain->size != size || domain->capacity != cap || domain->family != family ||
            domain->realm()->replicated() != bool(replicated))
          throw std::invalid_argument("transport changed an existing domain descriptor");
      } else {
        auto realm = known_realms_[origin].lock();
        auto& known_realm = foreign_realms_[origin];
        if (!realm) realm = known_realm.lock();
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
    auto value = decode_value_(type, domains, in);
    if (in.at != bytes.size()) throw std::invalid_argument("transport has trailing bytes");
    codec.validate(value);
    std::vector<FiniteDomainRef> used; collect_domains_(value, used);
    if (used.size() != domains.size()) throw std::invalid_argument("transport includes unused domains");
    return value;
  }

 private:
  std::map<Key, std::weak_ptr<const FiniteDomain>> domains_;
  std::map<std::array<std::uint64_t, 2>, std::weak_ptr<const FiniteDomainRealm>> foreign_realms_;
  std::map<std::array<std::uint64_t, 2>, std::weak_ptr<const FiniteDomainRealm>> known_realms_;
  distributed::Registry* registry_ = nullptr;
  ArrayResolver array_resolver_;
  static void collect_domains_(const Ref& value, std::vector<FiniteDomainRef>& domains) {
    if (value->kind() == ContinuationKind::DistributedArray) return; // authenticated vault reference
    if (const auto& domain = value->domain_ref(); domain &&
        std::find(domains.begin(), domains.end(), domain) == domains.end()) domains.push_back(domain);
    for (const auto& child : value->members()) collect_domains_(child, domains);
  }
  static ContinuationCodec<Dim> codec_(const ContinuationType& type, std::uint64_t capacity) {
    return ContinuationCodec<Dim>(type, continuation_transport_type_identity(type, true), capacity,
        [](int) -> const MultiFab<Dim>& { throw std::logic_error("transport has no field prototype"); });
  }
  void capture_domain_(const FiniteDomainRef& domain) {
    domain->require_live(); const auto origin = domain->realm()->transport_identity();
    auto& known_realm=known_realms_[origin];
    if(auto previous=known_realm.lock();previous && previous!=domain->realm() && previous->visible())
      throw std::logic_error("transport capture conflicts with a live finite realm");
    known_realm=domain->realm();
    const Key key{origin[0], origin[1], static_cast<std::uint64_t>(domain->version)};
    auto& known = domains_[key];
    if (auto previous = known.lock(); previous && previous != domain && previous->visible())
      throw std::logic_error("transport capture conflicts with a live domain authority");
    known = domain;
  }

  void encode_value_(const Ref& value, const std::vector<FiniteDomainRef>& domains,
                     continuation_detail::Writer& out) const {
    out.word(static_cast<unsigned>(value->kind()));
    if (continuation_distributed_kind(value->kind())) {
      if (!registry_) throw std::logic_error("distributed transport has no live registry");
      const auto& domain = value->distributed_domain_ref(); registry_->authenticate(domain);
      const auto id = domain->identity();
      out.word(id.issuer[0]); out.word(id.issuer[1]); out.word(id.ordinal);
      out.word(domain->family()); out.word(domain->capacity());
      out.word(domain->partition_offsets().size());
      for (auto offset : domain->partition_offsets()) out.word(offset);
      if (value->kind() == ContinuationKind::DistributedIndex)
        out.word(std::bit_cast<std::uint64_t>(value->distributed_index_offset()));
      if (value->kind() == ContinuationKind::DistributedArray) {
        const auto& array = value->distributed_array_ref();
        // The resolver authenticates a published immutable global version. The local
        // values never masquerade as a complete global array in the message.
        if (!array_resolver_ || array_resolver_(array->version, domain, array->element) != array)
          throw std::logic_error("distributed Array is not an authenticated published version");
        out.word(array->version.issuer[0]); out.word(array->version.issuer[1]); out.word(array->version.ordinal);
      }
      return;
    }
    switch (value->kind()) {
      case ContinuationKind::Boolean: case ContinuationKind::Int64: case ContinuationKind::Float64:
        out.word(value->word_); return;
      case ContinuationKind::Variant: out.word(value->arm()); break;
      case ContinuationKind::Domain: case ContinuationKind::Index: case ContinuationKind::Array: {
        auto found = std::find(domains.begin(), domains.end(), value->domain_ref());
        if (found == domains.end()) throw std::logic_error("transport lost a finite alias");
        out.word(static_cast<std::uint64_t>(found-domains.begin()));
        if (value->kind() == ContinuationKind::Index) out.word(value->word_);
        if (value->kind() != ContinuationKind::Array) return;
        break;
      }
      case ContinuationKind::Product: break;
      default: throw std::invalid_argument("transport cannot serialize this structural kind");
    }
    out.word(value->members().size());
    for (const auto& child : value->members()) encode_value_(child, domains, out);
  }

  Ref decode_value_(const ContinuationType& type, const std::vector<FiniteDomainRef>& domains,
                    continuation_detail::Reader& in) const {
    if (in.word() != static_cast<unsigned>(type.kind)) throw std::invalid_argument("transport value kind differs");
    if (continuation_distributed_kind(type.kind)) {
      if (!registry_) throw std::logic_error("distributed transport has no live registry");
      const distributed::Identity id{{in.word(), in.word()}, in.word()};
      const auto family = std::bit_cast<std::int64_t>(in.word()), cap = std::bit_cast<std::int64_t>(in.word());
      const auto n = in.word();
      if (n != static_cast<std::size_t>(registry_->lane().size()+1) || n > (in.bytes.size()-in.at)/8)
        throw std::invalid_argument("distributed transport ownership differs from its execution lane");
      std::vector<std::int64_t> offsets; offsets.reserve(static_cast<std::size_t>(n));
      for (std::uint64_t j = 0; j < n; ++j) offsets.push_back(std::bit_cast<std::int64_t>(in.word()));
      if (family != type.family || cap != type.capacity)
        throw std::invalid_argument("distributed transport family differs from installed schema");
      auto domain = registry_->resolve(id, family, cap, offsets);
      if (type.kind == ContinuationKind::DistributedDomain) return Value::distributed_domain(std::move(domain));
      if (type.kind == ContinuationKind::DistributedIndex)
        return Value::distributed_index(std::move(domain), std::bit_cast<std::int64_t>(in.word()));
      const distributed::Identity version{{in.word(), in.word()}, in.word()};
      if (!array_resolver_) throw std::logic_error("distributed Array transport has no resolver");
      return Value::distributed_array(array_resolver_(version, std::move(domain), type.children[0]));
    }
    if (type.kind == ContinuationKind::Boolean) {
      const auto v = in.word(); if (v > 1) throw std::invalid_argument("transport Bool is invalid");
      return Value::boolean(v != 0);
    }
    if (type.kind == ContinuationKind::Int64) return Value::integer(std::bit_cast<std::int64_t>(in.word()));
    if (type.kind == ContinuationKind::Float64) return Value::real(std::bit_cast<double>(in.word()));
    std::uint64_t arm = 0;
    if (type.kind == ContinuationKind::Variant) {
      arm = in.word(); if (arm >= type.children.size()) throw std::invalid_argument("transport Variant tag is invalid");
    }
    FiniteDomainRef domain;
    if (type.kind == ContinuationKind::Domain || type.kind == ContinuationKind::Index || type.kind == ContinuationKind::Array) {
      const auto slot = in.word(); if (slot >= domains.size()) throw std::invalid_argument("transport domain alias is invalid");
      domain = domains[slot]; continuation_detail::domain(domain, type);
      if (type.kind == ContinuationKind::Domain) return Value::domain(std::move(domain));
      if (type.kind == ContinuationKind::Index) return Value::index(std::move(domain), std::bit_cast<std::int64_t>(in.word()));
    }
    const auto count = in.word();
    const auto expected = type.kind == ContinuationKind::Product ? type.children.size() :
        type.kind == ContinuationKind::Variant ? 1 : domain ? static_cast<std::uint64_t>(domain->size) : UINT64_MAX;
    if (count != expected || count > (in.bytes.size()-in.at)/8)
      throw std::invalid_argument("transport member count differs from its installed schema");
    std::vector<Ref> values; values.reserve(static_cast<std::size_t>(count));
    for (std::size_t j = 0; j < count; ++j)
      values.push_back(decode_value_(type.children[type.kind == ContinuationKind::Product ? j :
          type.kind == ContinuationKind::Variant ? arm : 0], domains, in));
    if (type.kind == ContinuationKind::Product) return Value::product(std::move(values));
    if (type.kind == ContinuationKind::Variant) return Value::variant(static_cast<std::uint32_t>(arm), std::move(values[0]));
    return Value::array(std::move(domain), std::move(values));
  }
};

// Deterministic rank-local retained-byte policy shared by publication and restore.
// The enclosing caller owns the collective phase; this routine never communicates.
template<int Dim>
std::uint64_t continuation_distributed_array_charge(
    const ContinuationDistributedArrayRef<Dim>& array, distributed::Registry& registry,
    std::uint64_t capacity) {
  if (!array || !array->domain || array->values.size()!=std::size_t(array->domain->local_size()))
    throw std::invalid_argument("distributed Array charge has inconsistent ownership");
  registry.authenticate(array->domain);
  ContinuationTransportSession<Dim> session;
  session.bind_distributed_registry(registry,
      [&registry](distributed::Identity version, distributed::DomainRef domain, const ContinuationType& element) {
        registry.authenticate(domain);
        auto held=std::static_pointer_cast<const ContinuationDistributedArray<Dim>>(
            registry.resolve_value(version,distributed::array_type_tag<Dim>()));
        if (!held || held->domain!=domain || held->element!=element ||
            held->version.issuer!=version.issuer || held->version.ordinal!=version.ordinal ||
            held->values.size()!=std::size_t(domain->local_size()))
          throw std::invalid_argument("Array charge dependency differs from its immutable vault version");
        return held;
      });
  const auto maximum=std::numeric_limits<std::uint64_t>::max();
  const auto type_bytes=continuation_transport_type_identity(array->element,true).size();
  if(array->domain->partition_offsets().size()>(maximum-256)/8)
    throw std::length_error("distributed Array charge overflow");
  std::uint64_t charge=256+8*array->domain->partition_offsets().size();
  if(type_bytes>maximum-charge) throw std::length_error("distributed Array charge overflow");
  charge+=type_bytes;
  for(const auto& value:array->values) {
    const auto bytes=session.encode(array->element,value,capacity);
    if(charge>maximum-32 || bytes.size()>maximum-charge-32)
      throw std::length_error("distributed Array charge overflow");
    charge+=32+bytes.size();
  }
  return charge;
}

template<int Dim>
std::vector<std::uint8_t> encode_continuation_transport(const ContinuationType& type,
    ContinuationValueRef<Dim> value, std::uint64_t capacity, ContinuationTransportSession<Dim>& session) {
  return session.encode(type, value, capacity);
}
template<int Dim>
ContinuationValueRef<Dim> decode_continuation_transport(const ContinuationType& type,
    std::span<const std::uint8_t> bytes, std::uint64_t capacity, ContinuationTransportSession<Dim>& session) {
  return session.decode(type, bytes, capacity);
}

/// Local-only payload codec for native transports. No collective, pointer, domain or field wire.
template<int Dim>
std::vector<std::uint8_t> encode_continuation_transport(const ContinuationType& type,
    ContinuationValueRef<Dim> value, std::uint64_t capacity) {
  ContinuationCodec<Dim> codec(type, continuation_transport_type_identity(type), capacity,
      [](int) -> const MultiFab<Dim>& { throw std::logic_error("transport has no field prototype"); });
  return codec.encode(value, 0);
}
template<int Dim>
ContinuationValueRef<Dim> decode_continuation_transport(const ContinuationType& type,
    std::span<const std::uint8_t> bytes, std::uint64_t capacity) {
  ContinuationCodec<Dim> codec(type, continuation_transport_type_identity(type), capacity,
      [](int) -> const MultiFab<Dim>& { throw std::logic_error("transport has no field prototype"); });
  auto result = codec.decode(bytes);
  if (result.publication != 0) throw std::invalid_argument("transport payload carries publication authority");
  return std::move(result.root);
}

} // namespace pops::runtime::program
