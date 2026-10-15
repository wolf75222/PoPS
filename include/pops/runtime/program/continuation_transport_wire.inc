// Included after ContinuationKind. The same word emission is used by native
// transport and by the inert archive's counting sink. Authority validation is
// deliberately the caller's responsibility; this layer only specifies wire bytes.
namespace continuation_detail {
template <class Output, class Domains>
void transport_header(Output& out, const std::string& identity, int dimension,
                      const Domains& domains) {
  out.text("POPSTRNS2");
  out.text(identity);
  out.word(dimension);
  out.word(domains.size());
  for (const auto& domain : domains) {
    const auto origin = domain->realm()->transport_identity();
    out.word(origin[0]);
    out.word(origin[1]);
    out.word(domain->realm()->replicated());
    out.word(domain->size);
    out.word(domain->capacity);
    out.word(domain->family);
    out.word(domain->version);
  }
}
template <class Output, class Identity, class Offsets>
void transport_distributed_node(Output& out, ContinuationKind kind, const Identity& domain,
                                std::int64_t family, std::int64_t capacity, const Offsets& offsets,
                                std::int64_t index, const Identity& array) {
  out.word(static_cast<unsigned>(kind));
  out.word(domain.issuer[0]);
  out.word(domain.issuer[1]);
  out.word(domain.ordinal);
  out.word(family);
  out.word(capacity);
  out.word(offsets.size());
  for (auto offset : offsets)
    out.word(offset);
  if (kind == ContinuationKind::DistributedIndex)
    out.word(index);
  if (kind == ContinuationKind::DistributedArray) {
    out.word(array.issuer[0]);
    out.word(array.issuer[1]);
    out.word(array.ordinal);
  }
}
template <class Output>
bool transport_finite_node(Output& out, ContinuationKind kind, std::uint64_t word,
                           std::uint64_t finite_slot, std::uint64_t members) {
  out.word(static_cast<unsigned>(kind));
  switch (kind) {
    case ContinuationKind::Boolean:
    case ContinuationKind::Int64:
    case ContinuationKind::Float64:
      out.word(word);
      return false;
    case ContinuationKind::Variant:
      out.word(word);
      break;
    case ContinuationKind::Domain:
    case ContinuationKind::Index:
    case ContinuationKind::Array:
      out.word(finite_slot);
      if (kind == ContinuationKind::Index)
        out.word(word);
      if (kind != ContinuationKind::Array)
        return false;
      break;
    case ContinuationKind::Product:
      break;
    default:
      throw std::invalid_argument("transport cannot serialize this structural kind");
  }
  out.word(members);
  return true;
}
}  // namespace continuation_detail
