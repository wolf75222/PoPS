public:
void configure_stencil_expressions(std::size_t bases, std::size_t terms,
                                   std::size_t identity_bytes) const {
  continuation_local_phase(
      [&] {
        const auto multiply = [](std::size_t a, std::size_t b) {
          if (b && a > std::numeric_limits<std::size_t>::max() / b)
            throw std::length_error("field exchange resource contract overflows native storage");
          return a * b;
        };
        std::size_t records = 0;
        for (int block = 0; block < n_blocks(); ++block) {
          const auto& value = state(block);
          for (std::size_t patch = 0; patch < value.layout().size(); ++patch) {
            const auto& box = value.layout()[patch];
            const auto count =
                multiply(multiply(static_cast<std::size_t>(box.numPts()), 2 * Dim), value.ncomp());
            if (count > std::numeric_limits<std::size_t>::max() - records)
              throw std::length_error("field exchange record capacity overflows native storage");
            records += count;
          }
        }
        const auto record_bytes = multiply(identity_bytes, 8);
        if (record_bytes > std::numeric_limits<std::size_t>::max() - 2048)
          throw std::length_error("field exchange identity capacity overflows native storage");
        const auto capacity =
            multiply(bases, multiply(std::max<std::size_t>(records, 1), record_bytes + 2048));
        field_exchanges_->configure({bases, terms, std::max<std::size_t>(records, 1), capacity});
        runtime_state().hist_.exchange_limits = field_exchanges_->limits();
      },
      "Program field exchange resource installation");
}

void prepare_stencil_expression_write(const field_type& output,
                                      std::initializer_list<const field_type*> inputs) const {
  std::exception_ptr error;
  try {
    std::vector<const void*> fields(inputs.begin(), inputs.end());
    field_exchanges_->prepare_write(&output, fields);
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, prepared_execution_lane(),
                                 "Program field exchange write preparation");
}

void finish_stencil_expression_write(const field_type& output) const noexcept {
  field_exchanges_->finish_write(&output);
}

template <class Producer>
void capture_stencil_exchanges(const field_type& rhs, int program_block,
                               const std::string& source_identity, Producer&& producer) const {
  const auto& lane = prepared_execution_lane();
  std::uint64_t sequence = 0;
  std::exception_ptr error;
  try {
    sequence = field_exchanges_->next_capture_sequence();
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane, "Program face evaluation ordinal");
  std::vector<FieldExchangeExpressions::Cell> cells;
  std::size_t bytes = 0;
  auto records = prepare_exchange_batch(
      [&](auto&& append) {
        std::forward<Producer>(producer)([&](ExchangeRecord record, const Index<Dim>& cell) {
          if (cells.size() >= field_exchanges_->limits().max_records_per_basis)
            throw std::length_error("captured field exchange record capacity exceeded");
          FieldExchangeExpressions::Cell coordinates;
          coordinates.reserve(Dim);
          for (int axis = 0; axis < Dim; ++axis)
            coordinates.push_back(cell[axis]);
          cells.push_back(std::move(coordinates));
          append(std::move(record));
        });
      },
      [&](ExchangeRecord& record) {
        FieldExchangeExpressions::qualify_capture(record, boundary_evaluation_point(0), sequence);
        field_exchange_detail::charge(bytes, 128 + 8 * Dim, field_exchanges_->limits().max_bytes);
        for (const auto* text :
             {&record.operation_identity, &record.occurrence_identity, &record.evaluation_context,
              &record.quadrature_identity, &record.source_evaluation_identity})
          field_exchange_detail::charge(bytes, text->size(), field_exchanges_->limits().max_bytes);
      },
      lane);
  try {
    ExchangeRecord identity{"capture", "source", source_identity, "identity", 1, 1., 0., 1., 1};
    FieldExchangeExpressions::qualify_capture(identity, boundary_evaluation_point(0), sequence);
    field_exchanges_->capture(&rhs, std::move(records), sequence, identity.evaluation_context,
                              program_block, std::move(cells));
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane, "Program face evaluation retention");
}

void publish_stencil_exchanges(int program_block, const field_type& accepted) const {
  publish_field_exchanges_(sys_block(program_block), accepted);
}

private:
void publish_field_exchanges_(int runtime_block, const field_type& accepted) const {
  const auto& lane = prepared_execution_lane();
  std::optional<FieldExchangeExpressions::Publication> publication;
  std::exception_ptr error;
  try {
    require_same_field_contract_(accepted, system_->block_state(runtime_block),
                                 "Program accepted face expression");
    const auto delivery = boundary_evaluation_point(0);
    publication.emplace(field_exchanges_->prepare_publication(runtime_block, &accepted, &delivery));
  } catch (...) {
    error = std::current_exception();
  }
  collectively_rethrow_exception(error, lane, "Program accepted face expression preparation");
  system_->stage_program_exchanges(publication->records);
  field_exchanges_->commit_publication(std::move(*publication));
}

public:
