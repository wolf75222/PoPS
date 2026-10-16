// Included inside pops::runtime::program, after FieldExchangeExpressions.
namespace field_exchange_detail {
inline void charge(std::size_t& used, std::size_t amount, std::size_t capacity) {
  if (used > capacity || amount > capacity - used)
    throw std::length_error("field exchange byte capacity exceeded");
  used += amount;
}
struct Writer {
  std::vector<std::uint8_t> bytes;
  std::size_t capacity;
  void word(std::uint64_t x) {
    auto used = bytes.size();
    charge(used, 8, capacity);
    for (unsigned i = 0; i < 8; ++i)
      bytes.push_back(static_cast<std::uint8_t>(x >> (8 * i)));
  }
  void blob(std::span<const std::uint8_t> data) {
    auto used = bytes.size();
    charge(used, 8, capacity);
    charge(used, data.size(), capacity);
    word(data.size());
    bytes.insert(bytes.end(), data.begin(), data.end());
  }
  void text(const std::string& text) {
    blob({reinterpret_cast<const std::uint8_t*>(text.data()), text.size()});
  }
};
struct Reader {
  std::span<const std::uint8_t> bytes;
  std::size_t at = 0;
  std::uint64_t word() {
    if (at > bytes.size() || bytes.size() - at < 8)
      throw std::invalid_argument("truncated field exchange image");
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i)
      value |= std::uint64_t(bytes[at++]) << (8 * i);
    return value;
  }
  std::span<const std::uint8_t> blob() {
    const auto n = word();
    if (n > bytes.size() - at)
      throw std::invalid_argument("truncated field exchange payload");
    auto result = bytes.subspan(at, n);
    at += n;
    return result;
  }
  std::string text() {
    auto b = blob();
    return {reinterpret_cast<const char*>(b.data()), b.size()};
  }
  std::size_t count(std::size_t capacity, std::size_t width) {
    const auto n = word();
    if (n > capacity || n > (bytes.size() - at) / width)
      throw std::length_error("field exchange table exceeds its authenticated capacity");
    return n;
  }
};
// Pure size preflight: no strings, keys or records are copied before this succeeds.
inline std::size_t basis_wire_size(const FieldExchangeExpressions::Basis& basis,
                                   const FieldExchangeExpressions::Limits& limits) {
  if (basis.identity.empty() || basis.program_block < 0 ||
      basis.records.size() != basis.cells.size() ||
      basis.records.size() > limits.max_records_per_basis)
    throw std::invalid_argument("field exchange basis lacks exact source/support authority");
  std::size_t bytes = 64;
  charge(bytes, basis.identity.size(), limits.max_bytes);
  for (std::size_t i = 0; i < basis.records.size(); ++i) {
    const auto& record = basis.records[i];
    const auto& cell = basis.cells[i];
    if (cell.empty() || cell.size() > 3)
      throw std::invalid_argument("invalid field exchange cell dimension");
    charge(bytes, 120 + 8 * cell.size(), limits.max_bytes);
    for (const auto* text :
         {&record.operation_identity, &record.occurrence_identity, &record.evaluation_context,
          &record.quadrature_identity, &record.source_evaluation_identity})
      charge(bytes, text->size(), limits.max_bytes);
  }
  return bytes;
}
inline std::size_t basis_bytes(const FieldExchangeExpressions::Basis& basis,
                               const FieldExchangeExpressions::Limits& limits) {
  const auto bytes = basis_wire_size(basis, limits);
  for (std::size_t i = 0; i < basis.records.size(); ++i) {
    const auto& record = basis.records[i];
    record.validate();
    const auto& cell = basis.cells[i];
    if (record.trace_axis < 0 || static_cast<std::size_t>(record.trace_axis) >= cell.size() ||
        record.trace_side < 0 || record.trace_side > 1)
      throw std::invalid_argument("field exchange cell incidence is incomplete");
    std::string incidence = "cell";
    for (auto coordinate : cell)
      incidence += ":" + std::to_string(coordinate);
    incidence +=
        "/axis:" + std::to_string(record.trace_axis) + "/side:" + std::to_string(record.trace_side);
    if (record.quadrature_identity != incidence &&
        record.quadrature_identity !=
            incidence + "/component:" + std::to_string(record.trace_component))
      throw std::invalid_argument("field exchange support disagrees with its quadrature identity");
    if (i && !(basis.records[i - 1].key() < record.key()))
      throw std::invalid_argument("duplicate or noncanonical source exchange incidence order");
  }
  return bytes;
}

inline std::vector<std::uint8_t> basis_wire(const FieldExchangeExpressions::Basis& basis,
                                            const FieldExchangeExpressions::Limits& limits) {
  (void)basis_bytes(basis, limits);
  Writer out{{}, limits.max_bytes};
  out.text(basis.identity);
  out.word(basis.program_block);
  AcceptedExchangeLedger ledger;
  for (const auto& record : basis.records)
    ledger.stage(record);
  out.blob(ledger.checkpoint(true));
  out.word(basis.cells.size());
  for (const auto& cell : basis.cells) {
    out.word(cell.size());
    for (auto coordinate : cell)
      out.word(std::bit_cast<std::uint64_t>(coordinate));
  }
  return std::move(out.bytes);
}
}  // namespace field_exchange_detail

inline void FieldExchangeExpressions::canonicalize_basis(Basis& basis) {
  if (basis.records.size() != basis.cells.size())
    throw std::invalid_argument("field exchange record/cell count differs");
  std::vector<std::size_t> order;
  order.reserve(basis.records.size());
  for (std::size_t i = 0; i < basis.records.size(); ++i)
    order.push_back(i);
  std::sort(order.begin(), order.end(),
            [&](auto a, auto b) { return basis.records[a].key() < basis.records[b].key(); });
  Records records;
  std::vector<Cell> cells;
  records.reserve(order.size());
  cells.reserve(order.size());
  for (auto i : order) {
    records.push_back(std::move(basis.records[i]));
    cells.push_back(std::move(basis.cells[i]));
  }
  basis.records.swap(records);
  basis.cells.swap(cells);
}

inline bool FieldExchangeExpressions::same_basis(const Basis& left, const Basis& right) noexcept {
  if (left.identity != right.identity || left.program_block != right.program_block ||
      left.cells != right.cells || left.records.size() != right.records.size())
    return false;
  for (std::size_t i = 0; i < left.records.size(); ++i) {
    const auto& a = left.records[i];
    const auto& b = right.records[i];
    if (a.key() != b.key() || a.orientation != b.orientation || a.multiplicity != b.multiplicity ||
        a.trace_axis != b.trace_axis || a.trace_side != b.trace_side ||
        a.trace_component != b.trace_component || a.exterior_trace != b.exterior_trace ||
        a.source_evaluation_identity != b.source_evaluation_identity ||
        std::bit_cast<std::uint64_t>(a.face_measure) !=
            std::bit_cast<std::uint64_t>(b.face_measure) ||
        std::bit_cast<std::uint64_t>(a.numerical_flux) !=
            std::bit_cast<std::uint64_t>(b.numerical_flux) ||
        std::bit_cast<std::uint64_t>(a.temporal_weight) !=
            std::bit_cast<std::uint64_t>(b.temporal_weight))
      return false;
  }
  return true;
}

inline void FieldExchangeExpressions::require_expression(const Ref& expression,
                                                         const Limits& limits) {
  if (!expression)
    return;
  if (!limits.max_bases || !limits.max_terms || !limits.max_records_per_basis || !limits.max_bytes)
    throw std::length_error("field exchange expression has no installed resource contract");
  if (expression->terms.size() > limits.max_bases)
    throw std::length_error("field exchange expression basis budget exceeded");
  std::size_t bytes = 32;
  for (std::size_t i = 0; i < expression->terms.size(); ++i) {
    const auto& term = expression->terms[i];
    if (!term.basis || !std::isfinite(term.coefficient) || term.coefficient == 0)
      throw std::invalid_argument("field exchange expression has an invalid basis");
    for (std::size_t j = 0; j < i; ++j)
      if (expression->terms[j].basis->identity == term.basis->identity)
        throw std::invalid_argument("field exchange expression repeats a basis");
    field_exchange_detail::charge(bytes, 24, limits.max_bytes);
    field_exchange_detail::charge(bytes, field_exchange_detail::basis_bytes(*term.basis, limits),
                                  limits.max_bytes);
  }
}

inline std::vector<std::uint8_t> FieldExchangeExpressions::encode_roots(std::span<const Ref> roots,
                                                                        const Limits& limits) {
  using namespace field_exchange_detail;
  std::size_t wire_budget = 32;
  // Bound the unique table without allocating an index. A repeated pointer is an alias;
  // repeated source identities are checked byte-for-byte below, after this preflight.
  std::size_t unique_bases = 0;
  for (std::size_t r = 0; r < roots.size(); ++r) {
    charge(wire_budget, 8, limits.max_bytes);
    const auto& root = roots[r];
    if (!root)
      continue;
    bool prior = false;
    for (std::size_t q = 0; q < r; ++q)
      prior = prior || roots[q].get() == root.get();
    if (prior)
      continue;
    charge(wire_budget, 16, limits.max_bytes);
    for (std::size_t t = 0; t < root->terms.size(); ++t) {
      const auto& term = root->terms[t];
      if (!term.basis)
        throw std::invalid_argument("absent field exchange basis");
      charge(wire_budget, 16, limits.max_bytes);
      bool seen = false;
      for (std::size_t q = 0; q <= r; ++q)
        if (roots[q]) {
          const auto count = q == r ? t : roots[q]->terms.size();
          for (std::size_t k = 0; k < count; ++k)
            if (roots[q]->terms[k].basis &&
                roots[q]->terms[k].basis->identity == term.basis->identity)
              seen = true;
        }
      if (!seen) {
        if (++unique_bases > limits.max_bases)
          throw std::length_error("retained field exchange basis budget exceeded");
        charge(wire_budget, 8, limits.max_bytes);
        charge(wire_budget, basis_wire_size(*term.basis, limits), limits.max_bytes);
      }
    }
  }
  std::size_t workspace = 0;
  for (int copy = 0; copy < 3; ++copy)
    charge(workspace, wire_budget, limits.max_bytes);
  for (std::size_t r = 0; r < roots.size(); ++r) {
    charge(workspace, sizeof(Ref) + 4 * sizeof(void*), limits.max_bytes);
    const auto& root = roots[r];
    if (!root)
      continue;
    bool prior = false;
    for (std::size_t q = 0; q < r; ++q)
      prior = prior || roots[q].get() == root.get();
    if (prior)
      continue;
    charge(workspace, sizeof(Expression) + 4 * sizeof(void*), limits.max_bytes);
    for (const auto& term : root->terms) {
      charge(workspace, sizeof(Term) + 4 * sizeof(void*), limits.max_bytes);
      // One temporary ledger and its uniqueness tree while serializing a basis. This
      // over-bounds aliases without expanding any record payload before admission.
      for (const auto& record : term.basis->records) {
        (void)record;
        charge(workspace, sizeof(ExchangeRecord) + 4 * sizeof(std::string) + 8 * sizeof(void*),
               limits.max_bytes);
      }
    }
  }
  std::vector<Ref> expressions;
  std::map<const Expression*, std::size_t> expression_ids;
  std::vector<std::shared_ptr<const Basis>> bases;
  std::map<std::string, std::size_t> basis_ids;
  std::vector<std::vector<std::uint8_t>> basis_images;
  std::size_t budget = 40;
  for (const auto& root : roots) {
    charge(budget, 8, limits.max_bytes);
    if (!root || expression_ids.contains(root.get()))
      continue;
    require_expression(root, limits);
    charge(budget, 16, limits.max_bytes);
    expression_ids.emplace(root.get(), expressions.size() + 1);
    expressions.push_back(root);
    for (const auto& term : root->terms) {
      charge(budget, 16, limits.max_bytes);
      if (const auto found = basis_ids.find(term.basis->identity); found != basis_ids.end()) {
        if (!same_basis(*bases[found->second], *term.basis))
          throw std::invalid_argument("same field exchange source identity has different records");
        continue;
      }
      if (bases.size() >= limits.max_bases)
        throw std::length_error("retained field exchange basis budget exceeded");
      auto image = basis_wire(*term.basis, limits);
      charge(budget, 8, limits.max_bytes);
      charge(budget, image.size(), limits.max_bytes);
      basis_ids.emplace(term.basis->identity, bases.size());
      bases.push_back(term.basis);
      basis_images.push_back(std::move(image));
    }
  }
  Writer out{{}, limits.max_bytes};
  out.word(UINT64_C(0x3158504553504f50));  // POPSEPX1
  out.word(bases.size());
  for (const auto& bytes : basis_images)
    out.blob(bytes);
  out.word(expressions.size());
  for (const auto& expression : expressions) {
    out.word(expression->exact);
    out.word(expression->terms.size());
    for (const auto& term : expression->terms) {
      out.word(basis_ids.at(term.basis->identity));
      out.word(std::bit_cast<std::uint64_t>(term.coefficient));
    }
  }
  out.word(roots.size());
  for (const auto& root : roots)
    out.word(root ? expression_ids.at(root.get()) : 0);
  return std::move(out.bytes);
}

inline std::string FieldExchangeExpressions::collective_identity(const Ref& expression,
                                                                 const Limits& limits) {
  require_expression(expression, limits);
  field_exchange_detail::Writer out{{}, limits.max_bytes};
  out.word(bool(expression));
  if (expression) {
    out.word(expression->exact);
    out.word(expression->terms.size());
    for (const auto& term : expression->terms) {
      out.text(term.basis->identity);
      out.word(term.basis->program_block);
      out.word(std::bit_cast<std::uint64_t>(term.coefficient));
    }
  }
  return {reinterpret_cast<const char*>(out.bytes.data()), out.bytes.size()};
}

inline std::vector<FieldExchangeExpressions::Ref> FieldExchangeExpressions::decode_roots(
    std::span<const std::uint8_t> bytes, const Limits& limits) {
  using namespace field_exchange_detail;
  if (bytes.size() > limits.max_bytes)
    throw std::length_error("field exchange image exceeds byte capacity");
  (void)decoded_storage_bound(bytes, limits);
  Reader in{bytes};
  if (in.word() != UINT64_C(0x3158504553504f50))
    throw std::invalid_argument("unknown field exchange codec");
  const auto basis_count = in.count(limits.max_bases, 8);
  std::vector<std::shared_ptr<const Basis>> bases;
  std::set<std::string> identities;
  bases.reserve(basis_count);
  for (std::size_t i = 0; i < basis_count; ++i) {
    Reader row{in.blob()};
    Basis basis;
    basis.identity = row.text();
    const auto block = row.word();
    if (block > std::uint64_t(std::numeric_limits<int>::max()))
      throw std::invalid_argument("field exchange owner overflows int");
    basis.program_block = static_cast<int>(block);
    const auto ledger = AcceptedExchangeLedger::from_checkpoint(row.blob());
    basis.records = ledger.records();
    const auto cells = row.count(limits.max_records_per_basis, 8);
    if (cells != basis.records.size())
      throw std::invalid_argument("field exchange cell table differs from records");
    basis.cells.reserve(cells);
    for (std::size_t c = 0; c < cells; ++c) {
      const auto dim = row.count(3, 8);
      Cell cell;
      cell.reserve(dim);
      for (std::size_t axis = 0; axis < dim; ++axis)
        cell.push_back(std::bit_cast<std::int64_t>(row.word()));
      basis.cells.push_back(std::move(cell));
    }
    if (row.at != row.bytes.size() || !identities.insert(basis.identity).second)
      throw std::invalid_argument("noncanonical field exchange basis table");
    (void)basis_bytes(basis, limits);
    bases.push_back(std::make_shared<const Basis>(std::move(basis)));
  }
  const auto count = in.count(limits.max_bytes / 16, 16);
  std::vector<Ref> expressions;
  expressions.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto exact = in.word();
    if (exact > 1)
      throw std::invalid_argument("invalid field exchange proof tag");
    Expression expression;
    expression.exact = bool(exact);
    const auto terms = in.count(limits.max_bases, 16);
    expression.terms.reserve(terms);
    for (std::size_t t = 0; t < terms; ++t) {
      const auto index = in.word();
      if (index >= bases.size())
        throw std::invalid_argument("absent field exchange basis");
      expression.terms.push_back({bases[index], std::bit_cast<double>(in.word())});
    }
    auto ref = std::make_shared<const Expression>(std::move(expression));
    require_expression(ref, limits);
    expressions.push_back(std::move(ref));
  }
  const auto count_roots = in.count(limits.max_bytes / 8, 8);
  std::vector<Ref> roots;
  roots.reserve(count_roots);
  for (std::size_t i = 0; i < count_roots; ++i) {
    const auto index = in.word();
    if (index > expressions.size())
      throw std::invalid_argument("absent field exchange expression");
    roots.push_back(index ? expressions[index - 1] : Ref{});
  }
  if (in.at != bytes.size() ||
      encode_roots(roots, limits) != std::vector<std::uint8_t>(bytes.begin(), bytes.end()))
    throw std::invalid_argument("noncanonical or orphan field exchange metadata");
  return roots;
}

template <class OwnsCell>
FieldExchangeExpressions::Ref FieldExchangeExpressions::rematerialize(
    std::span<const Ref> sources, const Limits& limits, OwnsCell owns_cell,
    RematerializationCache* cache) {
  if (sources.empty())
    throw std::invalid_argument("field exchange migration has no authenticated sources");
  const auto& first = sources.front();
  for (const auto& source : sources) {
    require_expression(source, limits);
    if (bool(source) != bool(first) ||
        (source && (source->exact != first->exact || source->terms.size() != first->terms.size())))
      throw std::invalid_argument("field exchange migration source algebra differs");
  }
  if (!first)
    return {};
  Expression result;
  result.exact = first->exact;
  for (std::size_t term = 0; term < first->terms.size(); ++term) {
    const auto& initial = first->terms[term];
    Basis basis;
    basis.identity = initial.basis->identity;
    basis.program_block = initial.basis->program_block;
    std::set<std::tuple<std::string, std::string, std::string, std::string>> seen;
    for (const auto& source : sources) {
      const auto& selected = source->terms[term];
      if (selected.basis->identity != basis.identity ||
          selected.basis->program_block != basis.program_block ||
          std::bit_cast<std::uint64_t>(selected.coefficient) !=
              std::bit_cast<std::uint64_t>(initial.coefficient))
        throw std::invalid_argument("field exchange migration changes an exact source coefficient");
      for (std::size_t i = 0; i < selected.basis->records.size(); ++i) {
        const auto& record = selected.basis->records[i];
        if (!seen.insert(record.key()).second)
          throw std::invalid_argument("field exchange migration duplicates source ownership");
        if (owns_cell(basis.program_block, selected.basis->cells[i])) {
          if (basis.records.size() >= limits.max_records_per_basis)
            throw std::length_error("migrated field exchange exceeds target record capacity");
          basis.records.push_back(record);
          basis.cells.push_back(selected.basis->cells[i]);
        }
      }
    }
    canonicalize_basis(basis);
    auto materialized = std::make_shared<const Basis>(std::move(basis));
    if (cache) {
      auto found = cache->find(materialized->identity);
      if (found != cache->end()) {
        if (!same_basis(*found->second, *materialized))
          throw std::invalid_argument(
              "aliased field exchange migration has different source ownership");
        materialized = found->second;
      } else
        cache->emplace(materialized->identity, materialized);
    }
    result.terms.push_back({std::move(materialized), initial.coefficient});
  }
  auto result_ref = std::make_shared<const Expression>(std::move(result));
  require_expression(result_ref, limits);
  return result_ref;
}

inline std::size_t FieldExchangeExpressions::decoded_storage_bound(
    std::span<const std::uint8_t> bytes, const Limits& limits) {
  using namespace field_exchange_detail;
  if (bytes.size() > limits.max_bytes)
    throw std::length_error("field exchange image exceeds byte capacity");
  std::size_t result = 0;
  const auto add = [&](std::size_t count, std::size_t width) {
    if (width && count > std::numeric_limits<std::size_t>::max() / width)
      throw std::length_error("field exchange decode workspace overflows native storage");
    charge(result, count * width, std::numeric_limits<std::size_t>::max());
  };
  // Input strings are owned once by records and once by uniqueness keys; canonical validation
  // retains one encoded table and one re-encoded table. No recursive expansion of aliases.
  add(bytes.size(), 4);
  Reader in{bytes};
  if (in.word() != UINT64_C(0x3158504553504f50))
    throw std::invalid_argument("unknown field exchange codec");
  const auto basis_count = in.count(limits.max_bases, 8);
  add(basis_count, sizeof(Basis) + sizeof(std::shared_ptr<const Basis>) + sizeof(std::string) +
                       4 * sizeof(void*));
  for (std::size_t i = 0; i < basis_count; ++i) {
    Reader row{in.blob()};
    (void)row.blob();
    (void)row.word();
    Reader ledger{row.blob()};
    if (ledger.word() != UINT64_C(0x3230584553504f50))
      throw std::invalid_argument("retained field exchange requires the exact support codec");
    const auto ledger_records = ledger.count(limits.max_records_per_basis, 112);
    const auto records = row.count(limits.max_records_per_basis, 8);
    if (records != ledger_records)
      throw std::invalid_argument("field exchange cell table differs from records");
    // Decode keeps the ledger and the copied Basis alive together; ledger key nodes also live
    // until the entire Basis has validated. Account both object arrays before allocation.
    add(records, 2 * sizeof(ExchangeRecord) + sizeof(Cell) +
                     2 * sizeof(std::tuple<std::string, std::string, std::string, std::string>) +
                     8 * sizeof(void*));
    for (std::size_t record = 0; record < records; ++record) {
      const auto dimension = row.count(3, 8);
      add(dimension, sizeof(std::int64_t));
      for (std::size_t d = 0; d < dimension; ++d)
        (void)row.word();
    }
    if (row.at != row.bytes.size())
      throw std::invalid_argument("trailing field exchange basis bytes");
  }
  const auto expressions = in.count(limits.max_bytes / 16, 16);
  add(expressions, sizeof(Expression) + sizeof(Ref) +
                       sizeof(std::pair<const Expression*, std::size_t>) + 4 * sizeof(void*));
  for (std::size_t i = 0; i < expressions; ++i) {
    (void)in.word();
    const auto terms = in.count(limits.max_bases, 16);
    add(terms, sizeof(Term) + sizeof(std::string) + 4 * sizeof(void*));
    for (std::size_t term = 0; term < terms; ++term) {
      (void)in.word();
      (void)in.word();
    }
  }
  const auto roots = in.count(limits.max_bytes / 8, 8);
  add(roots, sizeof(Ref));
  for (std::size_t i = 0; i < roots; ++i)
    (void)in.word();
  if (in.at != bytes.size())
    throw std::invalid_argument("trailing field exchange image bytes");
  return result;
}

inline std::size_t FieldExchangeExpressions::rematerialized_storage_bound(
    std::span<const Ref> sources, const Limits& limits) {
  std::size_t result = sizeof(Expression);
  const auto add = [&](std::size_t amount) {
    field_exchange_detail::charge(result, amount, std::numeric_limits<std::size_t>::max());
  };
  for (const auto& source : sources) {
    if (!source)
      continue;
    if (source->terms.size() > limits.max_bases)
      throw std::length_error("field exchange basis budget exceeded");
    for (const auto& term : source->terms) {
      if (!term.basis)
        throw std::invalid_argument("absent field exchange basis");
      add(sizeof(Basis) + sizeof(Term) + sizeof(std::string) + 4 * sizeof(void*));
      // Includes the one candidate basis, its source-uniqueness index, and both exact byte
      // comparisons against a cached alias. This is a conservative bound before selection.
      const auto bytes = field_exchange_detail::basis_wire_size(*term.basis, limits);
      for (int copy = 0; copy < 4; ++copy)
        add(bytes);
      for (const auto& cell : term.basis->cells) {
        add(sizeof(ExchangeRecord) + sizeof(Cell) +
            sizeof(std::tuple<std::string, std::string, std::string, std::string>) +
            4 * sizeof(void*));
        add(cell.size() * sizeof(std::int64_t));
      }
    }
  }
  return result;
}
