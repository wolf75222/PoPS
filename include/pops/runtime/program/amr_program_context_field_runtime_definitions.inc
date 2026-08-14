
struct GeneratedFieldRoute {
  std::string field;
  std::vector<int> program_blocks;
  std::vector<int> runtime_blocks;

  friend bool operator==(const GeneratedFieldRoute&, const GeneratedFieldRoute&) = default;
};

static facade_type* require_facade_(facade_type* facade) {
  if (facade == nullptr)
    throw std::invalid_argument("AmrProgramContext requires a non-null exact-ranked facade");
  return facade;
}
static runtime_type* require_runtime_(facade_type& facade) {
  return require_runtime_(facade.engine());
}
static runtime_type* require_runtime_(runtime_type* runtime) {
  if (runtime == nullptr)
    throw std::invalid_argument("AmrProgramContext requires a materialized exact-ranked runtime");
  return runtime;
}
