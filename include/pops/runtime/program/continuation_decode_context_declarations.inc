// Included in ContinuationCodec<Dim>'s public section. One instance belongs to
// one unpublished multi-image restart transaction, never to live model state.
class DecodeContext {
 public:
  explicit DecodeContext(std::uint64_t workspace_bytes,
                         std::function<void(std::uint64_t)> reserve_workspace = {})
      : capacity_(workspace_bytes), reserve_workspace_(std::move(reserve_workspace)) {}

  DecodeContext(const DecodeContext&) = delete;
  DecodeContext& operator=(const DecodeContext&) = delete;
  std::uint64_t workspace_used() const noexcept { return used_; }

  std::shared_ptr<distributed::Registry> registry(const ExecutionLane&,
                                                  std::array<std::uint64_t, 2>, std::uint64_t,
                                                  std::uint64_t);
  distributed::DomainRef domain(distributed::Registry&, distributed::Identity, std::int64_t,
                                std::int64_t, std::span<const std::int64_t>);
  ContinuationDistributedArrayRef<Dim> array(
      distributed::Registry&, ContinuationDistributedArrayRef<Dim>, std::uint64_t, std::uint64_t,
      std::span<const std::shared_ptr<distributed::Registry>>);
  FiniteDomainRealmRef intern_realm(FiniteDomainRealmRef);
  FiniteDomainRef intern_domain(FiniteDomainRef);

 private:
  using Origin = std::array<std::uint64_t, 2>;
  using Key = std::array<std::uint64_t, 3>;
  struct ArrayEntry {
    ContinuationDistributedArrayRef<Dim> value;
    std::uint64_t charge;
  };
  std::map<Origin, std::shared_ptr<distributed::Registry>> registries_;
  std::map<Key, distributed::DomainRef> domains_;
  std::map<Key, ArrayEntry> arrays_;
  std::map<Origin, FiniteDomainRealmRef> realms_;
  std::map<Key, FiniteDomainRef> finite_;
  std::uint64_t capacity_, used_ = 0;

  std::function<void(std::uint64_t)> reserve_workspace_;

  void charge_(std::uint64_t bytes) {
    if (bytes > capacity_ - used_)
      throw std::length_error("continuation restart identity workspace exceeds its capacity");

    if (reserve_workspace_)
      reserve_workspace_(bytes);

    used_ += bytes;
  }
  static Key key_(distributed::Identity id) { return {id.issuer[0], id.issuer[1], id.ordinal}; }
};
